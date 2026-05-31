from typing import List, Dict, Any
from langchain_core.documents import Document

from cat import hook, RecallSettings, VectorDatabaseSettings
from cat.looking_glass.stray_cat import StrayCat
from cat.services.memory.models import PointStruct

from .graphrag_handler import Neo4jGraphRAGConfig, GraphRAGHandler


@hook(priority=10)
def factory_allowed_vector_databases(allowed: List[VectorDatabaseSettings], cat) -> List:
    allowed.append(Neo4jGraphRAGConfig)
    return allowed


@hook(priority=10)
async def before_cat_recalls_memories(config: RecallSettings, cat: StrayCat) -> RecallSettings:
    """
    Injects the current user message and embedder into the GraphRAGHandler
    before any memory retrieval takes place.

    - `user_message` lets the handler extract named entities from the raw query
      and perform direct graph lookups (Phase A② and A③).
    - `embedder` enables entity vector search (Phase A④) and allows entity
      embeddings to be stored during background ingestion tasks.

    Priority 10 ensures this hook runs before the default (priority 0).
    """
    if hasattr(cat.vector_memory_handler, "user_message"):
        cat.vector_memory_handler.user_message = cat.working_memory.user_message.text

    if hasattr(cat.vector_memory_handler, "embedder"):
        cat.vector_memory_handler.embedder = await cat.embedder()

    return config


@hook(priority=10)
async def before_rabbithole_stores_documents(docs: List[Document], cat) -> List[Document]:
    if hasattr(cat.vector_memory_handler, "embedder"):
        cat.vector_memory_handler.embedder = await cat.embedder()

    if isinstance(cat.vector_memory_handler, GraphRAGHandler):
        handler = cat.vector_memory_handler
        if handler.entity_extractor:
            await handler.entity_extractor.ensure_initialized()
        for i, doc in enumerate(docs):
            doc.metadata.setdefault("chunk_index", i)

    return docs


@hook
async def after_rabbithole_stored_documents(source: str, stored_points: List[PointStruct], cat) -> None:
    handler = cat.vector_memory_handler
    if not isinstance(handler, GraphRAGHandler):
        return
    settings = await cat.mad_hatter.get_plugin().load_settings()
    if not settings.get("enable_derived_graph", True):
        return
    await handler.create_derived_graph_for_source(source, stored_points, cat)


@hook(priority=10)
async def after_plugin_settings_update(plugin_id: str, settings: Dict[str, Any], cat) -> None:
    if isinstance(cat.vector_memory_handler, GraphRAGHandler) and cat.vector_memory_handler.entity_extractor:
        await cat.vector_memory_handler.entity_extractor.ensure_downloaded()
