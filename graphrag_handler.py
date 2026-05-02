import random
import math
import uuid
import json
import asyncio
from typing import Any, List, Iterable, Dict, Tuple, Optional, AsyncContextManager, cast, LiteralString, Type
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import Neo4jError
from langchain_core.documents import Document as LangChainDocument
from pydantic import Field, ConfigDict

from cat import BaseVectorDatabaseHandler, Embeddings, VectorDatabaseSettings
from cat.services.memory.models import (
    DocumentRecall, PointStruct, Record, ScoredPoint, UpdateResult
)
from cat.log import log

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
        neo4j_kwargs: Dict = None,
        document_vector_index: str = "document_embeddings",
        entity_vector_index: str = "entity_embeddings",
        vector_similarity_threshold: float = 0.7,
        enable_entity_extraction: bool = True,
        enable_entity_embeddings: bool = False,
        enable_entity_expansion: bool = True,
        spacy_models: Dict[str, str] = None,
        extra_technology_patterns: List[str] | None = None,
        graph_retrieval_depth: int = 2,
        graph_decay_factor: float = 0.5,
        connection_pool_size: int = 50,
        save_memory_snapshots: bool = False,
    ):
        super().__init__(save_memory_snapshots=save_memory_snapshots)

        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password
        self._neo4j_database = neo4j_database
        self._neo4j_kwargs = neo4j_kwargs or {}
        self._document_vector_index = document_vector_index
        self._entity_vector_index = entity_vector_index
        self._vector_similarity_threshold = vector_similarity_threshold
        self._enable_entity_extraction = enable_entity_extraction
        self._enable_entity_embeddings = enable_entity_embeddings
        self._enable_entity_expansion = enable_entity_expansion
        self._spacy_models = spacy_models or {"en": "en_core_web_sm"}
        self._extra_technology_patterns=extra_technology_patterns
        self._graph_retrieval_depth=graph_retrieval_depth
        self._graph_decay_factor=graph_decay_factor
        self._connection_pool_size=connection_pool_size

        self._driver: Optional[AsyncDriver] = None
        self._pending_entity_tasks: List[asyncio.Task] = []
        # Semaphore: caps concurrent Neo4j write transactions to reduce lock
        # contention and deadlock probability during bulk PDF ingestion.
        # Shared between entity extraction and similarity writes.
        self._neo4j_write_semaphore = asyncio.Semaphore(4)
        self._user_message = None
        self._embedder: Optional[Embeddings] = None

        # Initialize entity extractor
        self._entity_extractor: Optional[EntityExtractor] = EntityExtractor(
            models=self._spacy_models,
            extra_technology_patterns=self._extra_technology_patterns or None,
        ) if self._enable_entity_extraction else None
    
    def to_dict(self):
        return {
            "neo4j_uri": self._neo4j_uri,
            "neo4j_user": self._neo4j_user,
            "neo4j_password": self._neo4j_password,
            "neo4j_database": self._neo4j_database,
            "neo4j_kwargs": self._neo4j_kwargs,
            "document_vector_index": self._document_vector_index,
            "entity_vector_index": self._entity_vector_index,
            "vector_similarity_threshold": self._vector_similarity_threshold,
            "enable_entity_extraction": self._enable_entity_extraction,
            "enable_entity_embeddings": self._enable_entity_embeddings,
            "enable_entity_expansion": self._enable_entity_expansion,
            "spacy_models": self._spacy_models,
            "extra_technology_patterns": self._extra_technology_patterns,
            "graph_retrieval_depth": self._graph_retrieval_depth,
            "graph_decay_factor": self._graph_decay_factor,
            "connection_pool_size": self._connection_pool_size,
        }

    def _eq(self, other: "GraphRAGHandler") -> bool:
        return self.to_dict() == other.to_dict()

    @property
    def user_message(self) -> Optional[str]:
        return self._user_message

    @user_message.setter
    def user_message(self, value: str):
        self._user_message = value

    @property
    def embedder(self) -> Optional[Embeddings]:
        return self._embedder

    @embedder.setter
    def embedder(self, value: Embeddings):
        self._embedder = value
        
    @property
    def client(self):
        return self._driver

    @property
    def entity_extractor(self) -> EntityExtractor | None:
        return self._entity_extractor
        
    def tenant_field_condition(self) -> Dict:
        return {"key": "tenant_id", "match": {"value": self.agent_id}}
        
    def _get_session(self) -> AsyncContextManager[AsyncSession]:
        if not self._driver:
            raise RuntimeError("Neo4j driver not initialized")
        return self._driver.session(database=self._neo4j_database)
            
    async def _ensure_connected(self):
        if not self._driver:
            await self._connect()
            
    async def _connect(self):
        try:
            self._driver: AsyncDriver = AsyncGraphDatabase.driver(
                self._neo4j_uri,
                auth=(self._neo4j_user, self._neo4j_password),
                max_connection_pool_size=self._connection_pool_size,
                connection_acquisition_timeout=60,
                # Suppress GQL warnings 01N51 / 01N52 ("relationship type / property
                # key does not exist") that Neo4j emits on a fresh database before any
                # schema elements have been written.  These are harmless — queries that
                # match nothing simply return zero rows — but pollute the log on startup.
                notifications_disabled_categories=["UNRECOGNIZED"],
                **self._neo4j_kwargs,
            )
            assert isinstance(self._driver, AsyncDriver)
            async with self._driver.session(database=self._neo4j_database) as session:
                await session.run("RETURN 1")
            log.info(f"Connected to Neo4j at {self._neo4j_uri}")
        except Exception as e:
            log.error(f"Failed to connect to Neo4j: {e}")
            raise

    async def _ensure_vector_indexes_in_session(self, session, vector_dimensions: int):
        """Creates vector indexes for Document and Entity, using an already opened session."""
        # Index for Document
        doc_index_query = f"""
        CREATE VECTOR INDEX {self._document_vector_index} IF NOT EXISTS
        FOR (d:Document) ON d.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {vector_dimensions},
                `vector.similarity_function`: 'cosine',
                `vector.hnsw.ef_construction`: 200,
                `vector.hnsw.m`: 16
            }}
        }}
        """

        # Index for Entity (optional)
        entity_index_query = f"""
        CREATE VECTOR INDEX {self._entity_vector_index} IF NOT EXISTS
        FOR (e:Entity) ON e.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {vector_dimensions},
                `vector.similarity_function`: 'cosine',
                `vector.hnsw.ef_construction`: 200,
                `vector.hnsw.m`: 16
            }}
        }}
        """

        # Create document index
        try:
            await session.run(doc_index_query)
            log.info(f"Document vector index ensured: {self._document_vector_index}")
        except Exception as e:
            if "already exists" not in str(e):
                log.error(f"Document index creation warning: {e}")
                raise e

        # Create an entity index (if enabled)
        if self._enable_entity_embeddings:
            try:
                await session.run(entity_index_query)
                log.info(f"Entity vector index ensured: {self._entity_vector_index}")
            except Exception as e:
                if "already exists" not in str(e):
                    log.error(f"Entity index creation warning: {e}")
                    raise e

    @staticmethod
    async def _ensure_constraints_in_session(session):
        """Creates integrity constraints using an already opened session."""
        # noinspection SqlNoDataSourceInspection
        constraints = [
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT collection_unique IF NOT EXISTS FOR (c:Collection) REQUIRE (c.name, c.tenant_id) IS UNIQUE",
            # Composite index to speed up entity lookups by (tenant, name) without toLower() overhead
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id, e.name)",
        ]

        for constraint in constraints:
            try:
                await session.run(constraint)
            except Neo4jError as e:
                if "already exists" not in str(e):
                    log.error(f"Constraint creation failed: {e}")
                    raise e

    async def _get_index_dimensions(self, session, index_name: str) -> Optional[int]:
        """
        Returns the current `vector.dimensions` of an existing Neo4j vector index,
        or None if the index does not exist yet.

        Uses SHOW INDEXES (Neo4j 5.x) and filters in Python to avoid
        parameter-support limitations in SHOW commands.
        """
        result = await session.run("SHOW INDEXES YIELD name, type, options WHERE type = 'VECTOR'")
        records = await result.data()
        for record in records:
            if record.get("name") == index_name:
                index_config = (record.get("options") or {}).get("indexConfig", {})
                dims = index_config.get("vector.dimensions")
                if dims is not None:
                    return int(dims)
        return None

    async def _get_collection_embedder_config(
        self, session, collection_name: str
    ) -> Optional[Tuple[str, int]]:
        """
        Returns the (embedder_name, embedder_size) stored on a Collection node,
        or None if the collection does not exist or was created before this field
        was introduced.
        """
        # Avoid IS NOT NULL filters on properties that may not yet exist in the
        # database schema — Neo4j 5.x emits GQL warning 01N52 in that case.
        # Instead, fetch the raw values and perform the null check in Python.
        query = """
        MATCH (c:Collection {name: $name, tenant_id: $tenant_id})
        RETURN c.embedder_name AS embedder_name, c.embedder_size AS embedder_size
        """
        result = await session.run(query, name=collection_name, tenant_id=self.agent_id)
        record = await result.single()
        if record and record["embedder_name"] is not None and record["embedder_size"] is not None:
            return record["embedder_name"], int(record["embedder_size"])
        return None

    async def _drop_vector_indexes_in_session(self, session) -> None:
        """Drops the document (and optional entity) vector index so they can be
        recreated with the new embedder dimensions."""
        for index_name in [
            self._document_vector_index,
            self._entity_vector_index,
        ]:
            try:
                # noinspection SqlNoDataSourceInspection
                await session.run(f"DROP INDEX {index_name} IF EXISTS")
                log.info(f"[GraphRAG] Dropped vector index: {index_name}")
            except Exception as e:
                log.error(f"[GraphRAG] Failed to drop index {index_name}: {e}")

    async def _drop_tenant_data_in_session(self, session) -> None:
        """
        Deletes all Document and Collection nodes belonging to this tenant.
        Orphaned Entity nodes (no remaining MENTIONS) are pruned as well.

        Called when an embedder change is detected — all existing embeddings
        are stale and must be discarded before the indexes are rebuilt.
        """
        await session.run(
            """
            MATCH (c:Collection {tenant_id: $tenant_id})<-[:BELONGS_TO]-(d:Document)
            DETACH DELETE d
            """,
            tenant_id=self.agent_id,
        )
        await session.run(
            """
            MATCH (c:Collection {tenant_id: $tenant_id})
            DETACH DELETE c
            """,
            tenant_id=self.agent_id,
        )
        await session.run(
            """
            MATCH (e:Entity {tenant_id: $tenant_id})
            WHERE NOT (e)<-[:MENTIONS]-()
            DELETE e
            """,
            tenant_id=self.agent_id,
        )
        log.info(f"[GraphRAG] Tenant data wiped for agent_id={self.agent_id}")

    async def initialize(self, embedder_name: str, embedder_size: int):
        await self._connect()

        async with self._get_session() as session:
            # Constraints are always idempotent — create them first.
            await self._ensure_constraints_in_session(session)

            # ── Detect embedder change ────────────────────────────────────────
            # 1. Dimension mismatch → the HNSW index must be dropped and
            #    recreated (Neo4j vector indexes are immutable once created).
            index_dims = await self._get_index_dimensions(
                session, self._document_vector_index
            )
            index_needs_rebuild = index_dims is not None and index_dims != embedder_size

            # 2. Same dimension but different model → embeddings are in a
            #    different vector space; stale data must be discarded too.
            name_mismatch = False
            if not index_needs_rebuild:
                for collection_name in self._collection_names:
                    stored = await self._get_collection_embedder_config(
                        session, collection_name
                    )
                    if stored is not None:
                        stored_name, stored_size = stored
                        if stored_name != embedder_name or stored_size != embedder_size:
                            name_mismatch = True
                            break

            if index_needs_rebuild or name_mismatch:
                log.warning(
                    f"[GraphRAG] Embedder change detected "
                    f"(index_dims={index_dims} → {embedder_size}, "
                    f"name_mismatch={name_mismatch}). "
                    "Wiping tenant data and rebuilding indexes."
                )
                if self.save_memory_snapshots:
                    for collection_name in self._collection_names:
                        await self.save_dump(collection_name)

                await self._drop_tenant_data_in_session(session)

                if index_needs_rebuild:
                    await self._drop_vector_indexes_in_session(session)

            # Always ensure indexes exist with the correct dimensions.
            # If they were just dropped, this recreates them;
            # if they already match, IF NOT EXISTS is a no-op.
            await self._ensure_vector_indexes_in_session(session, embedder_size)

        # Create / update collections — always store current embedder metadata.
        async with self._get_session() as session:
            for collection_name in self._collection_names:
                await self._ensure_collection_exists_in_session(
                    session, collection_name, embedder_name, embedder_size
                )

        log.info(
            f"Advanced GraphRAG initialized "
            f"(embedder={embedder_name}, dims={embedder_size})"
        )

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

    def is_db_remote(self) -> bool:
        return True

    # ========== COLLECTION METHODS ==========

    async def _ensure_collection_exists_in_session(
        self,
        session,
        collection_name: str,
        embedder_name: str | None = None,
        embedder_size: int | None = None,
    ):
        """
        Creates the Collection node if it does not exist yet.
        `embedder_name` and `embedder_size` are stored (or updated) on the node
        so that future calls to `initialize` can detect an embedder change.
        """
        query = """
        MERGE (c:Collection {name: $name, tenant_id: $tenant_id})
        ON CREATE SET
            c.created_at     = datetime(),
            c.embedder_name  = $embedder_name,
            c.embedder_size  = $embedder_size
        ON MATCH SET
            c.embedder_name  = $embedder_name,
            c.embedder_size  = $embedder_size
        RETURN c
        """
        await session.run(
            query,
            name=collection_name,
            tenant_id=self.agent_id,
            embedder_name=embedder_name,
            embedder_size=embedder_size,
        )

    async def create_collection(self, embedder_name: str, embedder_size: int, collection_name: str):
        await self._ensure_connected()

        async with self._get_session() as session:
            await self._ensure_collection_exists_in_session(
                session, collection_name, embedder_name, embedder_size
            )

    async def create_hybrid_collection(self, collection_name: str, dense_config: str, sparse_config: str):
        log.warning("Hybrid collections not supported")
        pass

    async def delete_collection(self, collection_name: str, timeout: int | None = None):
        """
        Deletes a collection and its documents.
        Entities are deleted only if they become orphans (no remaining MENTIONS
        from documents in other collections), preserving cross-collection knowledge.
        """
        await self._ensure_connected()

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
            await session.run(cast(LiteralString, delete_docs_query), name=collection_name, tenant_id=self.agent_id)
            await session.run(cast(LiteralString, delete_orphan_entities_query), tenant_id=self.agent_id)
        log.info(f"Collection {collection_name} deleted (orphaned entities pruned)")

    async def check_collection_existence(self, collection_name: str) -> bool:
        await self._ensure_connected()

        query = """
        MATCH (c:Collection {name: $name, tenant_id: $tenant_id})
        RETURN count(c) > 0 AS exists
        """
        async with self._get_session() as session:
            result = await session.run(cast(LiteralString, query), name=collection_name, tenant_id=self.agent_id)
            record = await result.single()
            return record["exists"] if record else False

    async def get_collection_names(self) -> List[str]:
        await self._ensure_connected()

        query = """
        MATCH (c:Collection {tenant_id: $tenant_id})
        RETURN c.name AS name
        """
        async with self._get_session() as session:
            result = await session.run(cast(LiteralString, query), tenant_id=self.agent_id)
            records = await result.data()
            return [r["name"] for r in records]

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

        # ── Guard: empty content ──────────────────────────────────────────────
        if not content or not content.strip():
            log.warning(
                f"[GraphRAG] Skipping point {point_id}: content is empty or whitespace-only. "
                "Check the document splitter / loader upstream."
            )
            return None

        vector_list = list(vector)

        # ── Guard: zero / non-finite embedding vector ─────────────────────────
        _l2_sq = sum(x * x for x in vector_list)
        if _l2_sq == 0.0 or not math.isfinite(math.sqrt(_l2_sq)):
            log.warning(
                f"[GraphRAG] Skipping point {point_id}: embedding vector has zero or "
                "non-finite L2-norm. The embedder may have returned a fallback zero "
                "tensor (e.g. empty input, cold-start failure, or unreachable model)."
            )
            return None

        metadata = metadata or {}
        metadata["tenant_id"] = self.agent_id
        # Neo4j does not support Map-type node properties (only primitives /
        # arrays of primitives are allowed).  Serialise to a JSON string so the
        # CREATE never raises ClientError.Statement.TypeError.  All retrieve
        # helpers already call json.loads() when they get back a string, so this
        # is fully backward-compatible.
        metadata_json = json.dumps(metadata)

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
            result = await session.run(
                cast(LiteralString, create_query),
                collection_name=collection_name,
                tenant_id=self.agent_id,
                id=point_id,
                content=content,
                embedding=vector_list,
                metadata=metadata_json,
            )
            # Consuming the result surfaces any server-side error immediately
            # (otherwise the async driver silently discards it on session close).
            record = await result.single()
            if record is None:
                log.warning(
                    f"[GraphRAG] Document {point_id} was NOT created: "
                    f"collection '{collection_name}' not found for tenant '{self.agent_id}'. "
                    "Make sure initialize() was called before ingesting documents."
                )
                return None

        # Start entity extraction in the background
        if self._enable_entity_extraction and self._entity_extractor:
            task = asyncio.create_task(self._extract_and_link_entities(point_id, content, metadata))
            self._pending_entity_tasks.append(task)

        # Create SIMILAR_TO relationships in the background (tracked for clean shutdown)
        sim_task = asyncio.create_task(self._create_similarity_relationships(point_id, vector_list, collection_name))
        self._pending_entity_tasks.append(sim_task)

        # Clean up completed tasks
        self._pending_entity_tasks = [t for t in self._pending_entity_tasks if not t.done()]

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
        metadata: Dict,
    ) -> None:
        """
        Extracts entities from the document and links them to the graph.
        Runs in the background.

        Uses three batched UNWIND queries instead of N sequential calls:
        one for entity nodes, one for MENTIONS edges, one for RELATED_TO edges.
        Relations with the same (source, target, type) key are deduplicated
        in Python before being sent, averaging their weights.
        """
        batch_entity_query = """
        UNWIND $entities AS ent
        MERGE (e:Entity {id: ent.id, tenant_id: $tenant_id})
        ON CREATE SET
            e.name       = ent.name,
            e.type       = ent.type,
            e.created_at = datetime(),
            e.metadata   = ent.metadata,
            e.embedding  = ent.embedding
        ON MATCH SET
            e.last_seen  = datetime(),
            e.embedding  = CASE WHEN ent.embedding IS NOT NULL THEN ent.embedding ELSE e.embedding END
        """

        batch_mention_query = """
        MATCH (d:Document {id: $doc_id, tenant_id: $tenant_id})
        WITH d
        UNWIND $mentions AS m
        MATCH (e:Entity {id: m.entity_id, tenant_id: $tenant_id})
        MERGE (d)-[r:MENTIONS]->(e)
        ON CREATE SET r.created_at  = datetime(), r.confidence = m.confidence
        ON MATCH SET  r.last_seen   = datetime(), r.confidence = m.confidence
        """

        batch_relation_query = """
        UNWIND $relations AS rel
        MATCH (s:Entity {id: rel.source_id, tenant_id: $tenant_id})
        MATCH (t:Entity {id: rel.target_id, tenant_id: $tenant_id})
        MERGE (s)-[r:RELATED_TO {type: rel.rel_type}]->(t)
        ON CREATE SET r.weight = rel.weight, r.created_at = datetime()
        ON MATCH SET  r.weight = (r.weight + rel.weight) / 2
        """

        try:
            extracted = await self._entity_extractor.extract(content, document_id, metadata)
            if not extracted.entities:
                return

            entity_type_map = {e.name: e.type for e in extracted.entities}

            # Build batch payload for entities and mentions in one pass
            entities_batch = []
            mentions_batch = []
            for entity in extracted.entities:
                entity_id = self._entity_extractor.get_entity_hash(
                    entity.name, entity.type, self.agent_id
                )
                entities_batch.append({
                    "id":        entity_id,
                    "name":      entity.name.lower().strip(),
                    "type":      entity.type.value,
                    # Serialise to JSON string: Neo4j does not support Map-type
                    # node properties (only primitives / arrays are allowed).
                    "metadata":  json.dumps({"source_document": document_id, "confidence": entity.confidence}),
                    "embedding": None,  # populated below when enable_entity_embeddings=True
                })
                mentions_batch.append({
                    "entity_id":  entity_id,
                    "confidence": entity.confidence,
                })

            # Batch-embed all entity names in one call (non-blocking via thread)
            if self._enable_entity_embeddings and self._embedder is not None:
                try:
                    names = [ent["name"] for ent in entities_batch]
                    embeddings = await asyncio.to_thread(
                        self._embedder.embed_documents, names
                    )
                    for ent, emb in zip(entities_batch, embeddings):
                        ent["embedding"] = emb
                except Exception as emb_err:
                    log.warning(f"[GraphRAG] Entity embedding skipped: {emb_err}")

            # Build and deduplicate relation payload (average weights on collision)
            rel_map: Dict[Tuple, Dict] = {}
            for relation in extracted.relations:
                source_type = entity_type_map.get(relation.source_entity, EntityType.UNKNOWN)
                target_type = entity_type_map.get(relation.target_entity, EntityType.UNKNOWN)
                source_id = self._entity_extractor.get_entity_hash(
                    relation.source_entity, source_type, self.agent_id
                )
                target_id = self._entity_extractor.get_entity_hash(
                    relation.target_entity, target_type, self.agent_id
                )
                if not source_id or not target_id or source_id == target_id:
                    continue
                key = (source_id, target_id, relation.relation_type)
                if key in rel_map:
                    rel_map[key]["weight"] = (rel_map[key]["weight"] + relation.weight) / 2
                else:
                    rel_map[key] = {
                        "source_id": source_id,
                        "target_id": target_id,
                        "rel_type":  relation.relation_type,
                        "weight":    relation.weight,
                    }
            relations_batch = list(rel_map.values())

            # Sort all batches by ID so every concurrent transaction acquires
            # Neo4j node locks in the same order — breaks circular wait chains.
            entities_batch.sort(key=lambda e: e["id"])
            mentions_batch.sort(key=lambda m: m["entity_id"])
            relations_batch.sort(key=lambda r: (r["source_id"], r["target_id"]))

            # execute_write wraps all three queries in a single managed write
            # transaction that the Neo4j driver automatically retries on
            # TransientError (including DeadlockDetected).
            # The semaphore caps concurrent write transactions to further
            # reduce lock contention during bulk PDF ingestion.
            async def _write_entities(tx):
                await tx.run(
                    cast(LiteralString, batch_entity_query),
                    entities=entities_batch,
                    tenant_id=self.agent_id,
                )
                await tx.run(
                    cast(LiteralString, batch_mention_query),
                    mentions=mentions_batch,
                    doc_id=document_id,
                    tenant_id=self.agent_id,
                )
                if relations_batch:
                    await tx.run(
                        cast(LiteralString, batch_relation_query),
                        relations=relations_batch,
                        tenant_id=self.agent_id,
                    )

            async with self._neo4j_write_semaphore:
                async with self._get_session() as session:
                    await session.execute_write(_write_entities)

            log.debug(
                f"Linked {len(entities_batch)} entities and {len(relations_batch)} relations "
                f"for document {document_id}"
            )
        except Exception as e:
            log.error(f"Failed to extract entities for {document_id}: {e}")

    async def _create_similarity_relationships(self, point_id: str, vector: List[float], collection_name: str):
        """
        Creates bidirectional SIMILAR_TO relationships between similar documents.

        Both directions (a→b and b→a) are stored so graph traversal never misses
        a link regardless of the direction used by future queries.
        A single UNWIND query replaces the previous one-round-trip-per-document loop.
        """
        # ── Guard: reject zero / non-finite vectors before hitting Neo4j ──────
        _l2_sq = sum(x * x for x in vector)
        if _l2_sq == 0.0 or not math.isfinite(math.sqrt(_l2_sq)):
            log.warning(
                f"[GraphRAG] Skipping similarity search for {point_id}: "
                "vector has zero or non-finite L2-norm."
            )
            return

        find_similar_query = """
        MATCH (c:Collection {name: $collection_name, tenant_id: $tenant_id})
        CALL db.index.vector.queryNodes($index_name, 20, $vector)
        YIELD node, score
        WHERE EXISTS { MATCH (node)-[:BELONGS_TO]->(c) }
          AND node.id <> $point_id
          AND score >= $threshold
        RETURN node.id AS id, score
        ORDER BY score DESC
        """

        create_rel_query = """
        UNWIND $similar AS sim
        MATCH (a:Document {id: $point_id})
        MATCH (b:Document {id: sim.id})
        MERGE (a)-[r1:SIMILAR_TO]->(b)
        SET r1.score = sim.score, r1.updated_at = datetime()
        MERGE (b)-[r2:SIMILAR_TO]->(a)
        SET r2.score = sim.score, r2.updated_at = datetime()
        """

        try:
            # Read phase — auto-commit, read-only, no write locks acquired.
            async with self._get_session() as session:
                result = await session.run(
                    cast(LiteralString, find_similar_query),
                    collection_name=collection_name,
                    tenant_id=self.agent_id,
                    index_name=self._document_vector_index,
                    vector=vector,
                    point_id=point_id,
                    threshold=self._vector_similarity_threshold,
                )
                similar = await result.data()

            if not similar:
                log.debug(f"No similar documents found for {point_id}")
                return

            # Sort by document id so every concurrent transaction acquires
            # node relationship-group locks in the same order, breaking
            # circular wait chains between transactions.
            similar.sort(key=lambda s: s["id"])

            # Write phase — execute_write uses a managed transaction that the
            # Neo4j driver automatically retries on TransientError (deadlock).
            # The semaphore caps concurrent writers to reduce contention.
            async def _write_similarity(tx):
                await tx.run(
                    cast(LiteralString, create_rel_query),
                    similar=similar,
                    point_id=point_id,
                )

            async with self._neo4j_write_semaphore:
                async with self._get_session() as session:
                    await session.execute_write(_write_similarity)

            log.debug(f"Created {len(similar) * 2} similarity relationships for {point_id}")

        except Exception as e:
            log.error(f"Failed to create similarity relationships: {e}")

    async def add_points_to_tenant(
        self, collection_name: str, points: List[PointStruct]
    ) -> UpdateResult:
        await self._ensure_connected()

        operation_id = random.randint(1, 100000)
        for point in points:
            await self.add_point_to_tenant(
                collection_name,
                point.payload.get("page_content", ""),  # type: ignore[arg-type]
                point.vector,
                point.payload.get("metadata", {}),  # type: ignore[arg-type]
                str(point.id),
            )
        return UpdateResult(status="completed", operation_id=operation_id)

    async def delete_tenant_points(self, collection_name: str, metadata: Dict | None = None) -> UpdateResult:
        await self._ensure_connected()

        operation_id = random.randint(1, 100000)

        conditions: List[str] = []
        params: Dict = {"tenant_id": self.agent_id, "collection_name": collection_name}

        if metadata:
            for k, v in metadata.items():
                safe_param = f"meta_{k.replace('-', '_').replace('.', '_')}"
                # Same CONTAINS strategy as get_all_tenant_points: match the
                # exact JSON fragment that json.dumps produces for this pair.
                conditions.append(f"d.metadata CONTAINS ${safe_param}")
                params[safe_param] = f'"{k}": {json.dumps(v)}'

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        query = f"""
        MATCH (c:Collection {{name: $collection_name, tenant_id: $tenant_id}})<-[:BELONGS_TO]-(d:Document)
        {where_clause}
        DETACH DELETE d
        """

        async with self._get_session() as session:
            await (await session.run(cast(LiteralString, query), **params)).consume()

        return UpdateResult(status="completed", operation_id=operation_id)

    async def delete_tenant_points_by_ids(self, collection_name: str, points_ids: List) -> UpdateResult:
        await self._ensure_connected()

        operation_id = random.randint(1, 100000)

        query = """
        MATCH (c:Collection {name: $collection_name})<-[:BELONGS_TO]-(d:Document)
        WHERE d.id IN $ids AND d.tenant_id = $tenant_id
        DETACH DELETE d
        """
        orphan_query = """
        MATCH (e:Entity {tenant_id: $tenant_id})
        WHERE NOT (e)<-[:MENTIONS]-()
        DELETE e
        """
        async with self._get_session() as session:
            await session.run(
                cast(LiteralString, query),
                collection_name=collection_name,
                ids=points_ids,
                tenant_id=self.agent_id
            )
            await session.run(cast(LiteralString, orphan_query), tenant_id=self.agent_id)
        return UpdateResult(status="completed", operation_id=operation_id)

    async def retrieve_tenant_points(self, collection_name: str, points: List) -> List[Record]:
        await self._ensure_connected()

        query = """
        MATCH (d:Document)
        WHERE d.id IN $ids AND d.tenant_id = $tenant_id
        RETURN d.id AS id, d.content AS content, d.metadata AS metadata, d.embedding AS embedding
        """
        async with self._get_session() as session:
            result = await session.run(cast(LiteralString, query), ids=points, tenant_id=self.agent_id)
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
        GraphRAG hybrid retrieval — four parallel signals, then a smart merge.

        Phase A (entity-first):
          ② Direct lookup — find documents that explicitly MENTION entities
             extracted from the raw user message (spaCy pipeline).
             Score = matched_entities / total_query_entities.
             Only when `enable_entity_expansion=True` and query entities found.
          ③ Related lookup — walk the RELATED_TO graph up to
             `graph_retrieval_depth` hops from the query entities.
             Score decays with hop distance.
             Only when `enable_entity_expansion=True` and query entities found.
          ④ Entity vector search — query the entity embedding index with the
             query embedding; retrieve documents that mention the closest entities.
             Score = max entity-similarity score across matched entities.
             Only when `enable_entity_embeddings=True` and embedder injected.

        Phase B (always active):
          ⑤ Standard HNSW vector search on document embeddings.

        Merge:
          A③ and A④ results are combined into one "indirect evidence" pool
          (max score when the same document appears in both).
          Documents found by both the entity pool and Phase B receive a boost.
          The final list is sorted by composite score and capped at k.
        """
        async def _empty() -> List[Dict]:
            return []

        async def retrieve() -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
            # A④ and B are always candidates — build their coroutines now
            ev_coro = (
                self._recall_entity_by_vector(collection_name, embedding, k_fetch)
                if self._enable_entity_embeddings and self._embedder is not None
                else _empty()
            )
            vr_coro = self._recall_by_vector(collection_name, embedding, k_fetch, threshold)

            # Entity name expansion (A② + A③) disabled → only A④ + B
            if not self._enable_entity_expansion or not self._entity_extractor:
                ev, vr = await asyncio.gather(ev_coro, vr_coro)
                return [], [], ev, vr

            # ── Phase A ──────────────────────────────────────────────────────
            query_entity_names = await self._extract_query_entities()

            # No recognisable entities in the query → A④ + B only
            if not query_entity_names:
                ev, vr = await asyncio.gather(ev_coro, vr_coro)
                return [], [], ev, vr

            # Run A② + A③ + A④ + B all in parallel — fully independent
            ed, er, ev, vr = await asyncio.gather(
                self._recall_entity_direct(collection_name, query_entity_names, k_fetch),
                self._recall_entity_related(collection_name, query_entity_names, k_fetch, depth, decay),
                ev_coro,
                vr_coro,
            )
            return ed, er, ev, vr

        await self._ensure_connected()

        threshold = threshold or self._vector_similarity_threshold
        k = k or 5
        depth = self._graph_retrieval_depth
        decay = self._graph_decay_factor
        # Fetch more than k to compensate for post-hoc collection filtering;
        # $param arithmetic is not supported inside Cypher, so pre-compute in Python.
        k_fetch = k * 2

        entity_direct, entity_related, entity_vector, vector_raw = await retrieve()

        # Merge A③ and A④ into one "indirect evidence" pool.
        # Both phases surface documents through associated entities rather than
        # direct name matches; treat them symmetrically and keep the best score
        # when the same document is found by both.
        indirect_map: Dict[str, Dict] = {}
        for r in entity_related:
            indirect_map[r["id"]] = r
        for r in entity_vector:
            if r["id"] not in indirect_map or r["score"] > indirect_map[r["id"]]["score"]:
                indirect_map[r["id"]] = r
        entity_indirect = list(indirect_map.values())

        return self._merge_and_rerank(entity_direct, entity_indirect, vector_raw, k, decay)  # type: ignore[arg-type]

    # ── Phase A helpers ───────────────────────────────────────────────────────

    async def _extract_query_entities(self) -> List[str]:
        """
        Extracts entity names from the current user message using the spaCy
        pipeline already loaded for document ingestion.

        Returns an empty list if the extractor is not ready or no entities
        are found, so callers can safely skip Phase A without failing.
        """
        if not self.user_message or not self._entity_extractor:
            return []

        doc = await self._entity_extractor.extract_doc(self.user_message)

        entities = self._entity_extractor.extract_entities(doc)
        entities += self._entity_extractor.extract_technologies_regex(self.user_message)  # type: ignore[arg-type]
        entities = self._entity_extractor.deduplicate_entities(entities)

        names = [e.name.lower().strip() for e in entities]
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
        WHERE q_e.name = q_name
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
                cast(LiteralString, query),
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
        WHERE q_e.name = q_name

        MATCH path = (q_e)-[:RELATED_TO*1..{depth}]-(r_e:Entity {{tenant_id: $tenant_id}})
        WHERE NOT r_e.name IN $entity_names

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
                cast(LiteralString, query),
                entity_names=entity_names,
                tenant_id=self.agent_id,
                collection_name=collection_name,
                decay=decay,
                k=k,
            )
            return await result.data()

    # ── Phase A④ helper ──────────────────────────────────────────────────────

    async def _recall_entity_by_vector(
        self,
        collection_name: str,
        embedding: List[float],
        k: int,
    ) -> List[Dict]:
        """
        Phase A④: searches the entity vector index with the query embedding.

        Finds entity nodes whose stored embedding is semantically close to the
        query, then returns documents that MENTION those entities.  The score
        for each document is the maximum entity-similarity score across all
        matched entities.

        This phase is complementary to A② (direct name match): it catches
        entities that are semantically related to the query even when spaCy
        did not extract them explicitly (paraphrases, abbreviations, synonyms).

        Only active when `enable_entity_embeddings=True` and entity embeddings
        have been stored during ingestion (requires the embedder to be injected).
        """
        query = """
        CALL db.index.vector.queryNodes($index_name, $k, $vector)
        YIELD node AS ent, score AS ent_score
        WHERE ent.tenant_id = $tenant_id
        MATCH (d:Document {tenant_id: $tenant_id})-[:MENTIONS]->(ent)
        WHERE EXISTS {
            MATCH (d)-[:BELONGS_TO]->(:Collection {name: $collection_name, tenant_id: $tenant_id})
        }
        WITH d, max(ent_score) AS score
        RETURN d.id        AS id,
               d.content   AS content,
               d.metadata  AS metadata,
               d.embedding AS embedding,
               score
        ORDER BY score DESC
        LIMIT $k
        """
        async with self._get_session() as session:
            result = await session.run(
                cast(LiteralString, query),
                index_name=self._entity_vector_index,
                k=k,
                vector=embedding,
                tenant_id=self.agent_id,
                collection_name=collection_name,
            )
            return await result.data()

    # ── Phase B helper ────────────────────────────────────────────────────────

    async def _recall_by_vector(
        self,
        collection_name: str,
        embedding: List[float],
        k_fetch: int,
        threshold: float | None = None,
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
                cast(LiteralString, query),
                index_name=self._document_vector_index,
                k_fetch=k_fetch,
                vector=embedding,
                threshold=threshold or 0.0,
                collection_name=collection_name,
                tenant_id=self.agent_id,
            )
            return await result.data()

    # ── Merge ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _merge_and_rerank(
        entity_direct: List[Dict],
        entity_indirect: List[Dict],
        vector_results: List[Dict],
        k: int,
        decay: float,
        boost: float = 1.3,
    ) -> List[DocumentRecall]:
        """
        Merges Phase A and Phase B results into a single ranked list.

        `entity_indirect` is the pre-merged pool of A③ (graph traversal) and
        A④ (entity vector search) results — both surfaces documents through
        associated entities rather than direct name matches.

        Scoring rules (applied in order of priority):
          1. Doc found by entity_direct AND vector   → max(es, vs) × boost        (jackpot)
          2. Doc found by entity_direct only         → entity_score
          3. Doc found by entity_indirect AND vector → max(es, vs) × (boost × decay)
          4. Doc found by entity_indirect only       → entity_score
          5. Doc found by vector only                → vector_score

        The boost (default 1.3, capped at 1.0) rewards documents that are both
        semantically similar to the query AND topically grounded in the graph.
        """
        def get_final_score(info) -> float:
            es = float(info["entity_score"])
            vs = float(info["vector_score"])
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
            } for r in entity_indirect if r["id"] not in registry  # entity_direct always takes priority
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
                    page_content=info.get("data", {}).get("content", ""),
                    metadata=load_metadata(info.get("data", {}).get("metadata", {})),
                    id=doc_id,
                ),
                vector=info.get("data", {}).get("embedding", []),
                id=doc_id,
                score=final_score,
            ) for doc_id, final_score, info in final[:k]
        ]

        log.debug(
            f"[GraphRAG] Merge: {len(entity_direct)} direct + "
            f"{len(entity_indirect)} indirect (graph+vector) + {len(vector_results)} vector "
            f"→ {len(documents)} final"
        )
        return documents

    async def recall_tenant_memory(self, collection_name: str) -> List[DocumentRecall]:
        await self._ensure_connected()

        """Retrieves all memory points."""
        query = """
        MATCH (c:Collection {name: $collection_name, tenant_id: $tenant_id})<-[:BELONGS_TO]-(d:Document)
        RETURN d.id AS id, d.content AS content, d.metadata AS metadata, d.embedding AS embedding
        """
        async with self._get_session() as session:
            result = await session.run(
                cast(LiteralString, query), collection_name=collection_name, tenant_id=self.agent_id,
            )
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
        await self._ensure_connected()

        skip = int(offset) if offset and offset.isdigit() else 0
        query_limit = limit or 1000

        where_clauses = ["d.tenant_id = $tenant_id", "c.name = $collection_name"]
        params: Dict = {
            "tenant_id": self.agent_id,
            "collection_name": collection_name,
            "skip": skip,
            "limit": query_limit,
        }

        if metadata:
            for k, v in metadata.items():
                safe_param = f"meta_{k.replace('-', '_').replace('.', '_')}"
                # metadata is stored as a JSON string (not a Map property).
                # Use CONTAINS with the exact JSON fragment that json.dumps
                # always produces for this key-value pair so the filter runs
                # server-side without requiring APOC or map-index syntax.
                # e.g.  {"source": "file", ...}  CONTAINS  '"source": "file"'
                where_clauses.append(f"d.metadata CONTAINS ${safe_param}")
                params[safe_param] = f'"{k}": {json.dumps(v)}'

        where_str = " AND ".join(where_clauses)

        query = f"""
        MATCH (c:Collection)<-[:BELONGS_TO]-(d:Document)
        WHERE {where_str}
        RETURN d.id AS id, d.content AS content, d.metadata AS metadata, d.embedding AS embedding
        SKIP $skip
        LIMIT $limit
        """

        async with self._get_session() as session:
            result = await session.run(cast(LiteralString, query), **params)
            records = await result.data()

        points = []
        for r in records:
            metadata_dict = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else (r["metadata"] or {})
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
        await self._ensure_connected()

        query = """
        MATCH (c:Collection {name: $collection_name, tenant_id: $tenant_id})<-[:BELONGS_TO]-(d:Document)
        RETURN count(d) AS count
        """
        async with self._get_session() as session:
            result = await session.run(
                cast(LiteralString, query), collection_name=collection_name, tenant_id=self.agent_id,
            )
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
        await self._ensure_connected()

        search_query = """
        MATCH (c:Collection {name: $collection_name, tenant_id: $tenant_id})
        CALL db.index.vector.queryNodes($index_name, $limit, $vector)
        YIELD node, score
        WHERE EXISTS { MATCH (node)-[:BELONGS_TO]->(c) }
          AND score >= $threshold
        RETURN node.id AS id, node.content AS content, node.metadata AS metadata, 
               node.embedding AS embedding, score
        ORDER BY score DESC
        LIMIT $limit
        """

        async with self._get_session() as session:
            result = await session.run(
                cast(LiteralString, search_query),
                collection_name=collection_name,
                tenant_id=self.agent_id,
                index_name=self._document_vector_index,
                vector=query_vector,
                limit=limit,
                threshold=score_threshold or 0.0,
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
        await self._ensure_connected()

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


class Neo4jGraphRAGConfig(VectorDatabaseSettings):
    # Neo4j connection
    neo4j_uri: str = Field(default="neo4j://localhost:7687", description="Neo4j URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str | None = Field(default=None, description="Neo4j password")
    neo4j_database: str = Field(default="neo4j", description="Neo4j database name")
    neo4j_kwargs: Dict = Field(default={}, description="Neo4j extra arguments, as a dictionary")

    # Vector indexes
    document_vector_index: str = Field(default="document_embeddings", description="Name of the document vector index")
    entity_vector_index: str = Field(default="entity_embeddings", description="Name of the entity vector index")
    vector_similarity_threshold: float = Field(default=0.7, description="Minimum similarity score for vector search")

    # Entity extraction
    enable_entity_extraction: bool = Field(default=True, description="Enable entity extraction with spaCy")
    enable_entity_embeddings: bool = Field(
        default=False,
        description="Enable vector embeddings for entities (increases storage)",
    )
    enable_entity_expansion: bool = Field(default=True, description="Enable entity expansion during retrieval")
    spacy_models: Dict[str, str] = Field(
        default={"en": "en_core_web_lg"},
        description="spaCy model names for different languages (e.g. {'en': 'en_core_web_lg', 'de': 'de_core_news_lg'})",
    )
    extra_technology_patterns: List[str] | None = Field(
        default=[],
        description=(
            "Additional regex patterns for technology entity extraction. "
            "Useful for domain-specific keywords or non-English tech terms not "
            "covered by the built-in list (e.g. [r'\\b(MioFramework|AltroTool)\\b'])."
        ),
    )

    # Graph retrieval
    graph_retrieval_depth: int = Field(default=2, description="Max depth for graph traversal", ge=1, le=5)
    graph_decay_factor: float = Field(default=0.8, description="Score decay factor per hop", ge=0.5, le=1.0)

    # Performance
    connection_pool_size: int = Field(default=50,description="Neo4j connection pool size")

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Neo4j GraphRAG Advanced",
            "description": "Advanced GraphRAG with entity extraction, knowledge graph, and native vector indexes",
            "link": "https://neo4j.com/docs/vector-indexes/",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[GraphRAGHandler]:
        return GraphRAGHandler
