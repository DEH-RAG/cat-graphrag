from cat import hook, RecallSettings
from cat.looking_glass.stray_cat import StrayCat


@hook(priority=10)
def before_cat_recalls_memories(config: RecallSettings, cat: StrayCat) -> RecallSettings:
    """
    Captures the user's raw message text before any memory retrieval takes place.

    The value is stored in the module-level `_USER_MESSAGE` global so that
    GraphRAGHandler.recall_tenant_memory_from_embedding can extract named entities
    from the query and perform a direct graph lookup – rather than expanding
    only from whatever vector search happens to return.

    Priority 10 ensures this hook runs before the default (priority 0).
    """

    if hasattr(cat.vector_memory_handler, "user_message"):
        cat.vector_memory_handler.user_message = cat.working_memory.user_message.text

    return config
