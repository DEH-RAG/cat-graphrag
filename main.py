from typing import List
from langchain_core.documents import Document

from cat import hook, RecallSettings
from cat.looking_glass.stray_cat import StrayCat


@hook(priority=10)
def before_cat_recalls_memories(config: RecallSettings, cat: StrayCat) -> RecallSettings:
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
        cat.vector_memory_handler.embedder = cat.embedder

    return config


@hook(priority=10)
def before_rabbithole_stores_documents(docs: List[Document], cat) -> List[Document]:
    if hasattr(cat.vector_memory_handler, "embedder"):
        cat.vector_memory_handler.embedder = cat.embedder

    return docs
