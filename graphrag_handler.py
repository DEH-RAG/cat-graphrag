import random
import uuid
import json
import asyncio
from typing import Any, List, Iterable, Dict, Tuple, Optional, AsyncContextManager
from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import Neo4jError
from langchain_core.documents import Document as LangChainDocument
from cat import BaseVectorDatabaseHandler
from cat.services.memory.models import (
    DocumentRecall, PointStruct, Record, ScoredPoint, UpdateResult
)
from cat.log import log

from .settings import Neo4jGraphRAGConfig
from .entity_extractor import EntityExtractor
from .models import EntityType


class GraphRAGHandler(BaseVectorDatabaseHandler):
    """
    Advanced GraphRAG handler with:
    - Neo4j 5.23+ vector indexes (HNSW)
    - Entity extraction with spaCy
    - Knowledge graph with entities and semantic relations
    - Hybrid retrieval (vector + graph + entity expansion)
    """
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        neo4j_database: str = "neo4j",
        document_vector_index: str = "document_embeddings",
        entity_vector_index: str = "entity_embeddings",
        vector_dimensions: int = 384,
        vector_similarity_threshold: float = 0.7,
        enable_entity_extraction: bool = True,
        enable_entity_embeddings: bool = False,
        enable_entity_expansion: bool = True,
        spacy_model: str = "en_core_web_lg",
        extra_technology_patterns: List[str] | None = None,
        save_memory_snapshots: bool = False,
    ):
        super().__init__(save_memory_snapshots=save_memory_snapshots)
        self.config = Neo4jGraphRAGConfig(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database,
            document_vector_index=document_vector_index,
            entity_vector_index=entity_vector_index,
            vector_dimensions=vector_dimensions,
            vector_similarity_threshold=vector_similarity_threshold,
            enable_entity_extraction=enable_entity_extraction,
            enable_entity_embeddings=enable_entity_embeddings,
            enable_entity_expansion=enable_entity_expansion,
            spacy_model=spacy_model,
            extra_technology_patterns=extra_technology_patterns,
        )
        self._driver: Optional[AsyncDriver] = None
        self._entity_extractor: Optional[EntityExtractor] = None
        self._initialized = False
        self._pending_entity_tasks: List[asyncio.Task] = []
        self._user_message = None

    @property
    def user_message(self) -> Optional[str]:
        return self._user_message

    @user_message.setter
    def user_message(self, value: str):
        self._user_message = value
        
    @property
    def client(self):
        return self._driver
        
    def tenant_field_condition(self) -> Dict:
        return {"key": "tenant_id", "match": {"value": self.agent_id}}
        
    def _get_session(self) -> AsyncContextManager:
        if not self._driver:
            raise RuntimeError("Neo4j driver not initialized")
        return self._driver.session(database=self.config.neo4j_database)
            
    async def _ensure_connected(self):
        if not self._driver:
            await self._connect()
            
    async def _connect(self):
        try:
            self._driver = AsyncGraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
                max_connection_pool_size=self.config.connection_pool_size,
                connection_acquisition_timeout=60,
            )
            async with self._driver.session(database=self.config.neo4j_database) as session:
                await session.run("RETURN 1")
            log.info(f"Connected to Neo4j at {self.config.neo4j_uri}")
        except Exception as e:
            log.error(f"Failed to connect to Neo4j: {e}")
            raise

    async def _ensure_vector_indexes_in_session(self, session):
        """Creates vector indexes for Document and Entity, using an already opened session."""
        # Index for Document
        doc_index_query = f"""
        CREATE VECTOR INDEX {self.config.document_vector_index} IF NOT EXISTS
        FOR (d:Document) ON d.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.config.vector_dimensions},
                `vector.similarity_function`: 'cosine',
                `vector.hnsw.ef_construction`: 200,
                `vector.hnsw.m`: 16
            }}
        }}
        """

        # Index for Entity (optional)
        entity_index_query = f"""
        CREATE VECTOR INDEX {self.config.entity_vector_index} IF NOT EXISTS
        FOR (e:Entity) ON e.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.config.vector_dimensions},
                `vector.similarity_function`: 'cosine',
                `vector.hnsw.ef_construction`: 200,
                `vector.hnsw.m`: 16
            }}
        }}
        """

        # Create document index
        try:
            await session.run(doc_index_query)
            log.info(f"Document vector index ensured: {self.config.document_vector_index}")
        except Exception as e:
            if "already exists" not in str(e):
                log.warning(f"Document index creation warning: {e}")

        # Create an entity index (if enabled)
        if self.config.enable_entity_embeddings:
            try:
                await session.run(entity_index_query)
                log.info(f"Entity vector index ensured: {self.config.entity_vector_index}")
            except Exception as e:
                if "already exists" not in str(e):
                    log.warning(f"Entity index creation warning: {e}")

    @staticmethod
    async def _ensure_constraints_in_session(session):
        """Creates integrity constraints using an already opened session."""
        constraints = [
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT collection_unique IF NOT EXISTS FOR (c:Collection) REQUIRE (c.name, c.tenant_id) IS UNIQUE",
        ]

        for constraint in constraints:
            try:
                await session.run(constraint)
            except Neo4jError as e:
                if "already exists" not in str(e):
                    log.warning(f"Constraint creation failed: {e}")
                        
    async def initialize(self, embedder_name: str, embedder_size: int):
        if self._initialized:
            return

        log.info(f"Initializing Advanced GraphRAG for agent {self.agent_id}")

        self.config.vector_dimensions = embedder_size

        await self._connect()
        async with self._get_session() as session:
            await self._ensure_vector_indexes_in_session(session)
            await self._ensure_constraints_in_session(session)

        # Initialize entity extractor
        if self.config.enable_entity_extraction:
            self._entity_extractor = EntityExtractor(
                model_name=self.config.spacy_model,
                extra_technology_patterns=self.config.extra_technology_patterns or None,
            )
            await self._entity_extractor.initialize()
            log.info(f"Entity extractor initialized with model: {self.config.spacy_model}")

        # Create collections in one shared session
        async with self._get_session() as session:
            for collection_name in self._collection_names:
                await self._ensure_collection_exists_in_session(session, collection_name)

        self._initialized = True
        log.info(f"Advanced GraphRAG initialized")
        
    async def close(self):
        # Cancel and clean up all pending entity tasks
        for task in self._pending_entity_tasks:
            if not task.done():
                task.cancel()
        # Wait for cancellations to propagate
        if self._pending_entity_tasks:
            await asyncio.gather(*self._pending_entity_tasks, return_exceptions=True)
        self._pending_entity_tasks.clear()
                
        if self._driver:
            await self._driver.close()
            self._driver = None
            self._initialized = False
            
    def is_db_remote(self) -> bool:
        return True
        
    # ========== COLLECTION METHODS ==========

    async def _ensure_collection_exists(self, collection_name: str):
        async with self._get_session() as session:
            await self._ensure_collection_exists_in_session(session, collection_name)

    async def _ensure_collection_exists_in_session(self, session, collection_name: str):
        query = """
        MERGE (c:Collection {
            name: $name,
            tenant_id: $tenant_id
        })
        ON CREATE SET c.created_at = datetime()
        RETURN c
        """
        await session.run(query, name=collection_name, tenant_id=self.agent_id)
            
    async def create_collection(self, embedder_name: str, embedder_size: int, collection_name: str):
        await self._ensure_collection_exists(collection_name)
        
    async def create_hybrid_collection(self, collection_name: str, dense_config: str, sparse_config: str):
        log.warning("Hybrid collections not supported")
        pass
        
    async def delete_collection(self, collection_name: str, timeout: int | None = None):
        """
        Deletes a collection and its documents.
        Entities are deleted only if they become orphans (no remaining MENTIONS
        from documents in other collections), preserving cross-collection knowledge.
        """
        # Step 1: delete the collection node and all its documents.
        # DETACH DELETE removes all relationships including MENTIONS,
        # so orphan detection in Step 2 is correct.
        delete_docs_query = """
        MATCH (c:Collection {name: $name, tenant_id: $tenant_id})
        OPTIONAL MATCH (c)<-[:BELONGS_TO]-(d:Document)
        DETACH DELETE c, d
        """
        # Step 2: delete entities that are now unreferenced (no more MENTIONS
        # from any document, across all collections for this tenant).
        delete_orphan_entities_query = """
        MATCH (e:Entity {tenant_id: $tenant_id})
        WHERE NOT (e)<-[:MENTIONS]-()
        DELETE e
        """
        async with self._get_session() as session:
            await session.run(delete_docs_query, name=collection_name, tenant_id=self.agent_id)
            await session.run(delete_orphan_entities_query, tenant_id=self.agent_id)
        log.info(f"Collection {collection_name} deleted (orphaned entities pruned)")
        
    async def check_collection_existence(self, collection_name: str) -> bool:
        query = """
        MATCH (c:Collection {name: $name, tenant_id: $tenant_id})
        RETURN count(c) > 0 AS exists
        """
        async with self._get_session() as session:
            result = await session.run(query, name=collection_name, tenant_id=self.agent_id)
            record = await result.single()
            return record["exists"] if record else False
            
    async def get_collection_names(self) -> List[str]:
        query = """
        MATCH (c:Collection {tenant_id: $tenant_id})
        RETURN c.name AS name
        """
        async with self._get_session() as session:
            result = await session.run(query, tenant_id=self.agent_id)
            records = await result.data()
            return [r["name"] for r in records]

    async def check_embedding_size(self, embedder_name: str, embedder_size: int, collection_name: str) -> bool:
        """
        Validates that the running embedder dimension is compatible with the
        vector index already created for this collection.

        vector_dimensions is persisted in the plugin settings and reflects the
        dimension used when the Neo4j index was originally created.
        A mismatch means Neo4j will reject new vectors at insert time.
        """
        if self.config.vector_dimensions != embedder_size:
            log.error(
                f"[GraphRAG] Embedding dimension mismatch for collection '{collection_name}': "
                f"the Neo4j index was created with {self.config.vector_dimensions} dims but "
                f"the current embedder '{embedder_name}' produces {embedder_size} dims. "
                f"Drop and re-create the collection (or switch to a compatible embedder)."
            )
            return False
        return True
        
    async def save_dump(self, collection_name: str, folder: str = "dormouse/"):
        log.info(f"Save dump not implemented, use neo4j-admin dump")
        pass
        
    # ========== POINT METHODS ==========
    
    async def add_point_to_tenant(
        self,
        collection_name: str,
        content: str,
        vector: Iterable,
        metadata: Dict = None,
        id_point: str | None = None,
        **kwargs,
    ) -> PointStruct | None:
        """
        Adds a document:
        - Creates a Document node with embedding
        - Starts entity extraction in the background
        """
        await self._ensure_connected()
        
        point_id = id_point or str(uuid.uuid4())
        vector_list = list(vector)
        metadata = metadata or {}
        metadata["tenant_id"] = self.agent_id
        
        create_query = """
        MATCH (c:Collection {name: $collection_name, tenant_id: $tenant_id})
        CREATE (d:Document {
            id: $id,
            content: $content,
            embedding: $embedding,
            metadata: $metadata,
            tenant_id: $tenant_id,
            created_at: datetime()
        })
        CREATE (d)-[:BELONGS_TO]->(c)
        RETURN d.id AS id
        """
        
        async with self._get_session() as session:
            await session.run(
                create_query,
                collection_name=collection_name,
                tenant_id=self.agent_id,
                id=point_id,
                content=content,
                embedding=vector_list,
                metadata=metadata  # stored as native Neo4j map (no json.dumps)
            )
            
        # Start entity extraction in the background
        if self.config.enable_entity_extraction and self._entity_extractor:
            task = asyncio.create_task(self._extract_and_link_entities(point_id, content, metadata))
            self._pending_entity_tasks.append(task)
            # Clean up completed tasks
            self._pending_entity_tasks = [t for t in self._pending_entity_tasks if not t.done()]
            
        # Create SIMILAR_TO relationships in the background
        asyncio.create_task(self._create_similarity_relationships(point_id, vector_list, collection_name))
        
        return PointStruct(
            id=point_id,
            payload={
                "id": point_id,
                "page_content": content,
                "metadata": metadata,
                "tenant_id": self.agent_id,
            },
            vector=vector_list,
        )

    async def _extract_and_link_entities(
        self,
        document_id: str,
        content: str,
        metadata: Dict
    ) -> None:
        """
        Extracts entities from the document and links them to the graph.
        Runs in the background.
        """
        create_entity_query = """
        MERGE (e:Entity {
            id: $id,
            tenant_id: $tenant_id
        })
        ON CREATE SET 
            e.name = $name,
            e.type = $type,
            e.created_at = datetime(),
            e.metadata = $metadata
        ON MATCH SET 
            e.last_seen = datetime()
        RETURN e.id AS id
        """

        mention_query = """
        MATCH (d:Document {id: $doc_id, tenant_id: $tenant_id})
        MATCH (e:Entity {id: $entity_id, tenant_id: $tenant_id})
        MERGE (d)-[r:MENTIONS]->(e)
        ON CREATE SET r.created_at = datetime(), r.confidence = $confidence
        ON MATCH SET r.last_seen = datetime(), r.confidence = $confidence
        RETURN r
        """

        rel_query = """
        MATCH (s:Entity {id: $source_id, tenant_id: $tenant_id})
        MATCH (t:Entity {id: $target_id, tenant_id: $tenant_id})
        MERGE (s)-[r:RELATED_TO {type: $rel_type}]->(t)
        ON CREATE SET r.weight = $weight, r.created_at = datetime()
        ON MATCH SET r.weight = (r.weight + $weight) / 2
        """

        try:
            extracted = await self._entity_extractor.extract(
                content, document_id, metadata
            )

            if not extracted.entities:
                return

            # Build name → type map for correct relation hashing (fix: was always UNKNOWN)
            entity_type_map = {e.name: e.type for e in extracted.entities}

            async with self._get_session() as session:
                for entity in extracted.entities:
                    entity_id = self._entity_extractor.get_entity_hash(
                        entity.name, entity.type, self.agent_id
                    )

                    await session.run(
                        create_entity_query,
                        id=entity_id,
                        tenant_id=self.agent_id,
                        name=entity.name,
                        type=entity.type.value,
                        metadata={"source_document": document_id, "confidence": entity.confidence}  # native map
                    )

                    await session.run(
                        mention_query,
                        doc_id=document_id,
                        entity_id=entity_id,
                        tenant_id=self.agent_id,
                        confidence=entity.confidence
                    )

                # Extract relationships among entities
                for relation in extracted.relations:
                    # Use an actual entity type for the correct hash (not EntityType.UNKNOWN)
                    source_type = entity_type_map.get(relation.source_entity, EntityType.UNKNOWN)
                    target_type = entity_type_map.get(relation.target_entity, EntityType.UNKNOWN)
                    source_id = self._entity_extractor.get_entity_hash(
                        relation.source_entity, source_type, self.agent_id
                    )
                    target_id = self._entity_extractor.get_entity_hash(
                        relation.target_entity, target_type, self.agent_id
                    )

                    if source_id and target_id:
                        await session.run(
                            rel_query,
                            source_id=source_id,
                            target_id=target_id,
                            tenant_id=self.agent_id,
                            rel_type=relation.relation_type,
                            weight=relation.weight
                        )

            log.debug(
                f"Extracted {len(extracted.entities)} entities and {len(extracted.relations)} relations for {document_id}"
            )
        except Exception as e:
            log.error(f"Failed to extract entities for {document_id}: {e}")

    async def _create_similarity_relationships(self, point_id: str, vector: List[float], collection_name: str):
        """Creates SIMILAR_TO relationships between similar documents."""
        find_similar_query = """
        MATCH (c:Collection {name: $collection_name, tenant_id: $tenant_id})
        CALL db.index.vector.queryNodes($index_name, 20, $vector)
        YIELD node, score
        WHERE (node)-[:BELONGS_TO]->(c)
          AND node.id <> $point_id
          AND score >= $threshold
        RETURN node.id AS id, score
        ORDER BY score DESC
        """

        create_rel_query = """
        MATCH (a:Document {id: $source_id})
        MATCH (b:Document {id: $target_id})
        MERGE (a)-[r:SIMILAR_TO]->(b)
        SET r.score = $score, r.created_at = datetime()
        """

        try:
            async with self._get_session() as session:
                result = await session.run(
                    find_similar_query,
                    collection_name=collection_name,
                    tenant_id=self.agent_id,
                    index_name=self.config.document_vector_index,
                    vector=vector,
                    point_id=point_id,
                    threshold=self.config.vector_similarity_threshold
                )
                similar = await result.data()

                for sim in similar:
                    await session.run(
                        create_rel_query,
                        source_id=point_id,
                        target_id=sim["id"],
                        score=sim["score"]
                    )

            log.debug(f"Created {len(similar)} similarity relationships for {point_id}")

        except Exception as e:
            log.error(f"Failed to create similarity relationships: {e}")
            
    async def add_points_to_tenant(
        self, collection_name: str, points: List[PointStruct]
    ) -> UpdateResult:
        operation_id = random.randint(1, 100000)
        for point in points:
            await self.add_point_to_tenant(
                collection_name,
                point.payload.get("page_content", ""),
                point.vector,
                point.payload.get("metadata", {}),
                point.id
            )
        return UpdateResult(status="completed", operation_id=operation_id)
        
    async def delete_tenant_points(self, collection_name: str, metadata: Dict | None = None) -> UpdateResult:
        operation_id = random.randint(1, 100000)
        
        where_clause = "WHERE d.tenant_id = $tenant_id"
        params = {"tenant_id": self.agent_id, "collection_name": collection_name}
        
        if metadata:
            for k, v in metadata.items():
                # Use meta_ prefix to avoid collision with reserved param names
                safe_param = f"meta_{k.replace('-', '_').replace('.', '_')}"
                where_clause += f" AND d.metadata['{k}'] = ${safe_param}"
                params[safe_param] = v
                
        query = f"""
        MATCH (c:Collection {{name: $collection_name, tenant_id: $tenant_id}})<-[:BELONGS_TO]-(d:Document)
        {where_clause}
        DETACH DELETE d
        """
        
        async with self._get_session() as session:
            await session.run(query, **params)
            
        return UpdateResult(status="completed", operation_id=operation_id)
        
    async def delete_tenant_points_by_ids(self, collection_name: str, points_ids: List) -> UpdateResult:
        operation_id = random.randint(1, 100000)

        query = """
        MATCH (c:Collection {name: $collection_name})<-[:BELONGS_TO]-(d:Document)
        WHERE d.id IN $ids AND d.tenant_id = $tenant_id
        OPTIONAL MATCH (d)-[r:MENTIONS]->(e:Entity)
        DETACH DELETE d
        """
        async with self._get_session() as session:
            await session.run(
                query,
                collection_name=collection_name,
                ids=points_ids,
                tenant_id=self.agent_id
            )
        return UpdateResult(status="completed", operation_id=operation_id)
        
    async def retrieve_tenant_points(self, collection_name: str, points: List) -> List[Record]:
        query = """
        MATCH (d:Document)
        WHERE d.id IN $ids AND d.tenant_id = $tenant_id
        RETURN d.id AS id, d.content AS content, d.metadata AS metadata, d.embedding AS embedding
        """
        async with self._get_session() as session:
            result = await session.run(query, ids=points, tenant_id=self.agent_id)
            records = await result.data()
            
        return [
            Record(
                id=r["id"],
                payload={
                    "id": r["id"],
                    "page_content": r["content"],
                    "metadata": json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"],
                },
                vector=r["embedding"]
            )
            for r in records
        ]
        
    # ========== MAIN METHOD: HYBRID RECALL ==========

    async def recall_tenant_memory_from_embedding(
        self,
        collection_name: str,
        embedding: List[float],
        metadata: Dict | None = None,
        k: int | None = 5,
        threshold: float | None = None,
    ) -> List[DocumentRecall]:
        """
        GraphRAG hybrid retrieval — two parallel phases, then a smart merge.

        Phase A (entity-first, only when enable_entity_expansion=True):
          ① Extract named entities from the *user's raw message* (_USER_MESSAGE)
             using the same spaCy pipeline used for ingestion.
          ② Direct lookup — find documents that explicitly MENTION those entities.
             Score = (matched entities) / (total query entities), so docs that
             cover more of the query entities rank higher.
          ③ Related lookup — walk the RELATED_TO graph up to `graph_retrieval_depth`
             hops from the query entities and retrieve documents that mention the
             reached entities.  Score decays with hop distance.

        Phase B (always active):
          ④ Standard HNSW vector search on the document embeddings.

        Merge:
          Documents found by both Phase A and Phase B receive a confidence boost
          (they are both semantically similar AND topically relevant).
          Documents found only by one phase are included with their own score.
          The final list is sorted by score and capped at k.
        """
        async def retrieve() -> Tuple[List[Dict], List[Dict], List[Dict]]:
            # Entity expansion disabled → pure vector search only
            if not self.config.enable_entity_expansion or not self._entity_extractor:
                vr = await self._recall_by_vector(collection_name, embedding, k_fetch, threshold)
                return [], [], vr
            # ── Phase A ──────────────────────────────────────────────────────
            query_entity_names = await self._extract_query_entities()
            # No recognisable entities in the query → pure vector
            if not query_entity_names:
                vr = await self._recall_by_vector(collection_name, embedding, k_fetch, threshold)
                return [], [], vr
            # Run A② + A③ + B in parallel — they are fully independent
            ed, er, vr = await asyncio.gather(
                self._recall_entity_direct(collection_name, query_entity_names, k),
                self._recall_entity_related(collection_name, query_entity_names, k, depth, decay),
                self._recall_by_vector(collection_name, embedding, k_fetch, threshold),
            )
            return ed, er, vr

        await self._ensure_connected()

        threshold = threshold or self.config.vector_similarity_threshold
        k = k or 5
        depth = self.config.graph_retrieval_depth
        decay = self.config.graph_decay_factor
        k_fetch = k * 2  # CALL does not support $param arithmetic

        entity_direct, entity_related, vector_raw = await retrieve()
        return self._merge_and_rerank(entity_direct, entity_related, vector_raw, k, decay)

    # ── Phase A helpers ───────────────────────────────────────────────────────

    async def _extract_query_entities(self) -> List[str]:
        """
        Extracts entity names from the current user message using the spaCy
        pipeline already loaded for document ingestion.

        Returns an empty list if the extractor is not ready or no entities
        are found, so callers can safely skip Phase A without failing.
        """
        if not self.user_message or not self._entity_extractor or not self._entity_extractor.initialized:
            return []

        extractor = self._entity_extractor  # narrow type: EntityExtractor (not Optional)
        doc = await asyncio.to_thread(extractor.nlp, self.user_message)
        entities = extractor.extract_entities(doc)
        entities += extractor.extract_technologies_regex(self.user_message)
        entities = EntityExtractor.deduplicate_entities(entities)

        names = [e.name for e in entities]
        log.debug(f"[GraphRAG] Query entities: {names}")
        return names

    async def _recall_entity_direct(
        self,
        collection_name: str,
        entity_names: List[str],
        k: int,
    ) -> List[Dict]:
        """
        Phase A②: finds documents that directly MENTION at least one query entity.

        Score = (number of query entities mentioned in the document) / (total query
        entities).  A document mentioning all query entities scores 1.0; one that
        mentions half scores 0.5. This naturally surfaces the most topically
        complete answers.
        """
        query = """
        UNWIND $entity_names AS q_name
        MATCH (q_e:Entity {tenant_id: $tenant_id})
        WHERE toLower(q_e.name) = toLower(q_name)
        WITH DISTINCT q_e

        MATCH (d:Document {tenant_id: $tenant_id})-[:MENTIONS]->(q_e)
        WHERE EXISTS {
            MATCH (d)-[:BELONGS_TO]->(:Collection {name: $collection_name, tenant_id: $tenant_id})
        }

        WITH d, count(DISTINCT q_e) AS matched_count
        RETURN d.id        AS id,
               d.content   AS content,
               d.metadata  AS metadata,
               d.embedding AS embedding,
               toFloat(matched_count) / $num_entities AS score
        ORDER BY score DESC
        LIMIT $k
        """
        async with self._get_session() as session:
            result = await session.run(
                query,
                entity_names=entity_names,
                tenant_id=self.agent_id,
                collection_name=collection_name,
                num_entities=len(entity_names),
                k=k,
            )
            return await result.data()

    async def _recall_entity_related(
        self,
        collection_name: str,
        entity_names: List[str],
        k: int,
        depth: int,
        decay: float,
    ) -> List[Dict]:
        """
        Phase A③: walks the RELATED_TO graph from the query entities and
        retrieves documents that mention the reached entities.

        Score decays with hop distance: decay^1 for 1-hop, decay^2 for 2-hop, etc.
        Documents mentioning entities that are already directly in the query are
        excluded (they are already returned by _recall_entity_direct).

        depth is injected as a literal — Neo4j does not allow parameters in
        variable-length relationship bounds (*min..max).
        """
        query = f"""
        UNWIND $entity_names AS q_name
        MATCH (q_e:Entity {{tenant_id: $tenant_id}})
        WHERE toLower(q_e.name) = toLower(q_name)

        MATCH path = (q_e)-[:RELATED_TO*1..{depth}]-(r_e:Entity {{tenant_id: $tenant_id}})
        WHERE NOT toLower(r_e.name) IN [n IN $entity_names | toLower(n)]

        MATCH (d:Document {{tenant_id: $tenant_id}})-[:MENTIONS]->(r_e)
        WHERE EXISTS {{
            MATCH (d)-[:BELONGS_TO]->(:Collection {{name: $collection_name, tenant_id: $tenant_id}})
        }}

        WITH d, min(length(path)) AS min_hops
        RETURN d.id        AS id,
               d.content   AS content,
               d.metadata  AS metadata,
               d.embedding AS embedding,
               $decay ^ min_hops AS score
        ORDER BY score DESC
        LIMIT $k
        """
        async with self._get_session() as session:
            result = await session.run(
                query,
                entity_names=entity_names,
                tenant_id=self.agent_id,
                collection_name=collection_name,
                decay=decay,
                k=k,
            )
            return await result.data()

    # ── Phase B helper ────────────────────────────────────────────────────────

    async def _recall_by_vector(
        self,
        collection_name: str,
        embedding: List[float],
        k_fetch: int,
        threshold: float,
    ) -> List[Dict]:
        """
        Phase B: standard HNSW vector search on document embeddings.
        Returns raw Cypher records (dicts) so they can be merged with Phase A results
        before the final conversion to DocumentRecall.
        """
        query = """
        CALL db.index.vector.queryNodes($index_name, $k_fetch, $vector)
        YIELD node AS doc, score AS doc_score
        WHERE doc_score >= $threshold
          AND EXISTS {
              MATCH (doc)-[:BELONGS_TO]->(:Collection {name: $collection_name, tenant_id: $tenant_id})
          }
        RETURN doc.id        AS id,
               doc.content   AS content,
               doc.metadata  AS metadata,
               doc.embedding AS embedding,
               doc_score     AS score
        ORDER BY score DESC
        LIMIT $k_fetch
        """
        async with self._get_session() as session:
            result = await session.run(
                query,
                index_name=self.config.document_vector_index,
                k_fetch=k_fetch,
                vector=embedding,
                threshold=threshold,
                collection_name=collection_name,
                tenant_id=self.agent_id,
            )
            return await result.data()

    # ── Merge ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _merge_and_rerank(
        entity_direct: List[Dict],
        entity_related: List[Dict],
        vector_results: List[Dict],
        k: int,
        decay: float,
        boost: float = 1.3,
    ) -> List[DocumentRecall]:
        """
        Merges Phase A and Phase B results into a single ranked list.

        Scoring rules (applied in order of priority):
          1. Doc found by entity_direct AND vector  → max(es, vs) × boost   (jackpot)
          2. Doc found by entity_direct only        → entity_score
          3. Doc found by entity_related AND vector → max(es, vs) × (boost * decay)
          4. Doc found by entity_related only       → entity_score
          5. Doc found by vector only               → vector_score

        The boost (default 1.3, capped at 1.0) rewards documents that are both
        semantically similar to the query AND topically grounded in the graph.
        """
        def get_final_score(info) -> float:
            es = info["entity_score"]
            vs = info["vector_score"]
            is_direct = info["is_direct"]
            if es > 0 and vs > 0:
                applied_boost = boost if is_direct else boost * decay
                return min(1.0, max(es, vs) * applied_boost)
            return es or vs

        def load_metadata(metadata: Dict | str) -> Dict:
            if isinstance(metadata, dict):
                return metadata
            try:
                return json.loads(metadata)
            except json.JSONDecodeError:
                return {}

        # registry: doc_id → {data, entity_score, vector_score, is_direct}
        registry: Dict[str, Dict] = {
            r["id"]: {
                "data": r,
                "entity_score": r["score"],
                "vector_score": 0.0,
                "is_direct": True,
            } for r in entity_direct
        }

        registry.update({
            r["id"]: {
                "data": r,
                "entity_score": r["score"],
                "vector_score": 0.0,
                "is_direct": False,
            } for r in entity_related if r["id"] not in registry # entity_direct always takes priority
        })

        for r in vector_results:
            if r["id"] in registry:
                registry[r["id"]]["vector_score"] = r["score"]
            else:
                registry[r["id"]] = {
                    "data": r,
                    "entity_score": 0.0,
                    "vector_score": r["score"],
                    "is_direct": False,
                }

        final: List[Tuple[str, float, Dict]] = [
            (doc_id, get_final_score(info), info) for doc_id, info in registry.items()
        ]
        final.sort(key=lambda x: x[1], reverse=True)

        documents = [
            DocumentRecall(
                document=LangChainDocument(
                    page_content=info.get("data", {}).get("content"),
                    metadata=load_metadata(info.get("data", {}).get("metadata")),
                    id=doc_id,
                ),
                vector=info.get("data", {}).get("embedding"),
                id=doc_id,
                score=final_score,
            ) for doc_id, final_score, info in final[:k]
        ]

        log.debug(
            f"[GraphRAG] Merge: {len(entity_direct)} direct + "
            f"{len(entity_related)} related + {len(vector_results)} vector "
            f"→ {len(documents)} final"
        )
        return documents

    async def recall_tenant_memory(self, collection_name: str) -> List[DocumentRecall]:
        """Retrieves all memory points."""
        query = """
        MATCH (c:Collection {name: $collection_name, tenant_id: $tenant_id})<-[:BELONGS_TO]-(d:Document)
        RETURN d.id AS id, d.content AS content, d.metadata AS metadata, d.embedding AS embedding
        """
        async with self._get_session() as session:
            result = await session.run(query, collection_name=collection_name, tenant_id=self.agent_id)
            records = await result.data()
            
        documents = []
        for r in records:
            metadata_dict = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"]
            documents.append(DocumentRecall(
                document=LangChainDocument(
                    page_content=r["content"],
                    metadata=metadata_dict,
                    id=r["id"],
                ),
                vector=r["embedding"],
                id=r["id"],
            ))
        return documents
        
    # ========== GET ALL METHODS ==========
    
    async def get_all_tenant_points(
        self,
        collection_name: str,
        limit: int | None = None,
        offset: str | None = None,
        metadata: Dict | None = None,
        with_vectors: bool = True,
    ) -> Tuple[List[Record], int | str | None]:
        skip = int(offset) if offset and offset.isdigit() else 0
        query_limit = limit or 1000
        
        where_clauses = ["d.tenant_id = $tenant_id", "c.name = $collection_name"]
        params = {
            "tenant_id": self.agent_id,
            "collection_name": collection_name,
            "skip": skip,
            "limit": query_limit
        }
        
        if metadata:
            for k, v in metadata.items():
                # Use meta_ prefix to avoid collision with reserved param names
                # and sanitize key for valid Cypher identifier
                safe_param = f"meta_{k.replace('-', '_').replace('.', '_')}"
                where_clauses.append(f"d.metadata['{k}'] = ${safe_param}")
                params[safe_param] = v
                
        where_str = " AND ".join(where_clauses)
        
        query = f"""
        MATCH (c:Collection)<-[:BELONGS_TO]-(d:Document)
        WHERE {where_str}
        RETURN d.id AS id, d.content AS content, d.metadata AS metadata, d.embedding AS embedding
        SKIP $skip
        LIMIT $limit
        """
        
        async with self._get_session() as session:
            result = await session.run(query, **params)
            records = await result.data()
            
        points = []
        for r in records:
            metadata_dict = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"]
            points.append(Record(
                id=r["id"],
                payload={
                    "id": r["id"],
                    "page_content": r["content"],
                    "metadata": metadata_dict,
                },
                vector=r["embedding"] if with_vectors else None
            ))
            
        next_offset = str(skip + len(points)) if len(points) == query_limit else None
        return points, next_offset
        
    async def get_all_tenant_points_from_web(
        self, collection_name: str, limit: int | None = None, offset: str | None = None
    ) -> Tuple[List[Record], int | str | None]:
        return await self.get_all_tenant_points(
            collection_name, limit, offset, {"source": "http"}, with_vectors=False
        )
        
    async def get_all_tenant_points_from_files(
        self, collection_name: str, limit: int | None = None, offset: str | None = None
    ) -> Tuple[List[Record], int | str | None]:
        return await self.get_all_tenant_points(
            collection_name, limit, offset, {"source": "file"}, with_vectors=False
        )
        
    async def get_tenant_vectors_count(self, collection_name: str) -> int:
        query = """
        MATCH (c:Collection {name: $collection_name, tenant_id: $tenant_id})<-[:BELONGS_TO]-(d:Document)
        RETURN count(d) AS count
        """
        async with self._get_session() as session:
            result = await session.run(query, collection_name=collection_name, tenant_id=self.agent_id)
            record = await result.single()
            return record["count"] if record else 0
            
    # ========== SEARCH METHODS ==========
    
    async def search_in_tenant(
        self,
        collection_name: str,
        query_vector: List[float],
        query_filter: Any = None,
        with_payload: bool = True,
        with_vectors: bool = True,
        limit: int = 10,
        score_threshold: float | None = None,
    ) -> List[ScoredPoint]:
        """Direct vector search (without expansion)."""
        search_query = """
        MATCH (c:Collection {name: $collection_name, tenant_id: $tenant_id})
        CALL db.index.vector.queryNodes($index_name, $limit, $vector)
        YIELD node, score
        WHERE (node)-[:BELONGS_TO]->(c)
          AND score >= $threshold
        RETURN node.id AS id, node.content AS content, node.metadata AS metadata, 
               node.embedding AS embedding, score
        ORDER BY score DESC
        LIMIT $limit
        """
        
        async with self._get_session() as session:
            result = await session.run(
                search_query,
                collection_name=collection_name,
                tenant_id=self.agent_id,
                index_name=self.config.document_vector_index,
                vector=query_vector,
                limit=limit,
                threshold=score_threshold or 0.0
            )
            records = await result.data()
            
        scored_points = []
        for r in records:
            metadata_dict = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"]
            scored_points.append(ScoredPoint(
                id=r["id"],
                score=r["score"],
                payload={
                    "id": r["id"],
                    "page_content": r["content"],
                    "metadata": metadata_dict,
                },
                vector=r["embedding"] if with_vectors else None,
                version=r.get("version", 0),
            ))
        return scored_points
        
    async def search_prefetched_in_tenant(
        self,
        collection_name: str,
        query: str,
        query_vector: List[float],
        query_filter: Any,
        k: int,
        k_prefetched: int,
        threshold: float,
    ) -> List[ScoredPoint]:
        return await self.search_in_tenant(
            collection_name, query_vector, query_filter, True, True, k, threshold
        )
        
    # ========== UTILITY METHODS ==========
    
    def build_condition(self, key: str, value: Any) -> List:
        return [{"key": key, "match": {"value": value}}]
        
    def filter_from_dict(self, filter_dict: Dict) -> Any:
        if not filter_dict:
            return None
        return {"must": [{"key": k, "match": {"value": v}} for k, v in filter_dict.items()]}
