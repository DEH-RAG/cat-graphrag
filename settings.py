from cat import VectorDatabaseSettings
from pydantic import ConfigDict, Field
from typing import Type, List

from .graphrag_handler import GraphRAGHandler


class Neo4jGraphRAGConfig(VectorDatabaseSettings):
    # Neo4j connection
    neo4j_uri: str = Field(
        default="neo4j://localhost:7687",
        description="Neo4j URI"
    )
    neo4j_user: str = Field(
        default="neo4j",
        description="Neo4j username"
    )
    neo4j_password: str | None = Field(
        default=None,
        description="Neo4j password",
    )
    neo4j_database: str = Field(
        default="neo4j",
        description="Neo4j database name"
    )
    
    # Vector indexes
    document_vector_index: str = Field(
        default="document_embeddings",
        description="Name of the document vector index"
    )
    entity_vector_index: str = Field(
        default="entity_embeddings",
        description="Name of the entity vector index"
    )
    vector_dimensions: int = Field(
        default=384,
        description="Embedding dimensions"
    )
    vector_similarity_threshold: float = Field(
        default=0.7,
        description="Minimum similarity score for vector search"
    )
    
    # Entity extraction
    enable_entity_extraction: bool = Field(
        default=True,
        description="Enable entity extraction with spaCy"
    )
    enable_entity_embeddings: bool = Field(
        default=False,
        description="Enable vector embeddings for entities (increases storage)"
    )
    enable_entity_expansion: bool = Field(
        default=True,
        description="Enable entity expansion during retrieval"
    )
    spacy_model: str = Field(
        default="en_core_web_lg",
        description="spaCy model name (en_core_web_sm, en_core_web_md, en_core_web_lg, en_core_web_trf)"
    )
    extra_technology_patterns: List[str] = Field(
        default_factory=list,
        description=(
            "Additional regex patterns for technology entity extraction. "
            "Useful for domain-specific keywords or non-English tech terms not "
            "covered by the built-in list (e.g. [r'\\b(MioFramework|AltroTool)\\b'])."
        )
    )

    # Graph retrieval
    graph_retrieval_depth: int = Field(
        default=2,
        description="Max depth for graph traversal",
        ge=1,
        le=5
    )
    graph_decay_factor: float = Field(
        default=0.8,
        description="Score decay factor per hop",
        ge=0.5,
        le=1.0
    )
    
    # Performance
    connection_pool_size: int = Field(
        default=50,
        description="Neo4j connection pool size"
    )
    save_memory_snapshots: bool = Field(
        default=False,
        description="Enable snapshot backup"
    )
    
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
