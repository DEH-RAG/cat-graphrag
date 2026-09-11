from typing import List, Dict, Any
import asyncio
from langchain_core.documents import Document

from cat import hook, RecallSettings, VectorDatabaseSettings
from cat.log import log
from cat.looking_glass.stray_cat import StrayCat
from cat.services.memory.models import PointStruct

from .graphrag_handler import Neo4jGraphRAGConfig, GraphRAGHandler
from .entity_extractor import EntityExtractor

# Concept-recompute guard state (todo 12, concurrency lifecycle). Module-level
# (not per-handler) because the handler is re-instantiated across settings
# saves — the PERSISTED ``concept_gen_active`` marker is the real source of
# truth for single-flight; the lock only serializes access to the pending map
# and the dict holds the LATEST-WINS target generation per tenant. Both are
# harmless at import time (no side effects) and therefore import-safe.
_concept_recompute_lock = asyncio.Lock()
_concept_recompute_pending: Dict[str, str] = {}


@hook(priority=10)
def factory_allowed_vector_databases(allowed: List[VectorDatabaseSettings], cat) -> List:
    allowed.append(Neo4jGraphRAGConfig)
    return allowed


@hook(priority=10)
async def after_cheshire_cat_creation(cat) -> None:
    """
    Boot-time provenance migration.

    Every agent's graph that predates the PROVENANCE / ``source_files``
    tracking (nodes and relations created before this feature) is reconciled
    in the background, so the file-deletion cascade works for historical data
    too. ``recompute_provenance`` self-skips tenants that are already
    reconciled (a single ``provenance_reconciled`` marker count), so this is a
    cheap no-op on every boot after the first migration.
    """
    handler = getattr(cat, "vector_memory_handler", None)
    if not isinstance(handler, GraphRAGHandler):
        return
    task = asyncio.create_task(handler.recompute_provenance())
    handler._pending_entity_tasks.append(task)
    log.info(
        f"[GraphRAG] Scheduled provenance reconciliation for "
        f"{getattr(handler, 'agent_id', 'unknown')}"
    )


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
        if hasattr(cat.vector_memory_handler, "_align_embedder_lazy"):
            await cat.vector_memory_handler._align_embedder_lazy()

    return config


@hook(priority=10)
async def before_rabbithole_stores_documents(docs: List[Document], cat) -> List[Document]:
    if hasattr(cat.vector_memory_handler, "embedder"):
        cat.vector_memory_handler.embedder = await cat.embedder()
        if hasattr(cat.vector_memory_handler, "_align_embedder_lazy"):
            await cat.vector_memory_handler._align_embedder_lazy()

    if isinstance(cat.vector_memory_handler, GraphRAGHandler):
        handler = cat.vector_memory_handler
        if handler.entity_extractor:
            await handler.entity_extractor.ensure_initialized()
        for i, doc in enumerate(docs):
            doc.metadata.setdefault("chunk_index", i)

    return docs


@hook
async def after_rabbithole_stored_documents(source: str, stored_points: List[PointStruct], cat) -> None:
    # Lazy import (FX-7): the plugin loader reloads each file in glob order and
    # main.py comes BEFORE graphrag_handler.py, so a module-level import binds
    # the PRE-reload class and isinstance() below ALWAYS fails (the factory
    # instantiates the handler from the CURRENT, post-reload class). Import at
    # call time, when graphrag_handler is the current module, so the check
    # matches the class the factory actually created.
    from .graphrag_handler import GraphRAGHandler as _GH
    handler = cat.vector_memory_handler
    if not isinstance(handler, _GH):
        return
    # Single source of truth: the handler's config (vector-DB settings), NOT the
    # plugin settings store (which is empty unless explicitly saved — a stale
    # default there silently disabled the whole derived-graph + LLM path, FX-1).
    if not getattr(handler, "_enable_derived_graph", False):
        return
    await handler.create_derived_graph_for_source(source, stored_points, cat)


@hook(priority=10)
async def after_plugin_settings_update(plugin_id: str, settings: Dict[str, Any], cat) -> None:
    if isinstance(cat.vector_memory_handler, GraphRAGHandler) and cat.vector_memory_handler.entity_extractor:
        await cat.vector_memory_handler.entity_extractor.ensure_downloaded()


@hook(priority=10)
async def after_vector_database_settings_update(
    vector_database_name: str,
    previous_config: Dict[str, Any],
    new_config: Dict[str, Any],
    cat,
) -> None:
    """
    Reacts to ``Neo4jGraphRAGConfig`` saves with two INDEPENDENT refresh paths
    (neither sits behind the other's early return):

    1. Technology terminology: when ``extra_technology_patterns`` changed,
       rebuild the handler's EntityExtractor with the new patterns and re-run
       the pure-regex technology extraction over the stored Documents.
    2. Concept relations: when ``concept_definitions`` / ``relation_definitions``
       / ``concept_relations_prompt`` changed, compute the new config
       fingerprint and launch a single-flight background recompute that
       re-extracts concept relations from EVERY source of this agent tagging
       with the new generation, then atomically flips the tenant's
       ``concept_gen_active`` marker (``GraphRAGHandler._flip_concept_gen``).
       Retrieval (_recall_entity_related) gates on that marker, so old
       concepts disappear exactly when the new ones are in place.

    Idempotence/single-flight comes from the persisted marker + the latest-
    wins pending map + the gen-guard inside the flip — NOT from any in-memory
    handler state (handlers are re-instantiated on every save).
    """
    if vector_database_name != "Neo4jGraphRAGConfig":
        return

    handler = cat.vector_memory_handler
    if not isinstance(handler, GraphRAGHandler):
        return

    # 1 — Technology terminology refresh (unchanged semantics).
    if new_config.get("extra_technology_patterns") != previous_config.get("extra_technology_patterns"):
        if handler.entity_extractor:
            handler._entity_extractor = EntityExtractor(
                models=handler._spacy_models,
                extra_technology_patterns=new_config.get("extra_technology_patterns") or None,
            )
            await handler.refresh_technology_entities(tenant_id=cat.agent_key)

    # 2 — Concept-relation generation flip (todos 9-13).
    concept_changed = any(
        new_config.get(k) != previous_config.get(k)
        for k in ("concept_definitions", "relation_definitions", "concept_relations_prompt")
    )
    if not concept_changed:
        return

    new_gen = handler._concept_fingerprint()
    active = await handler._read_concept_gen(cat.agent_key)
    if active == new_gen:
        return  # the marker already enforces this generation: no-op

    async with _concept_recompute_lock:
        _concept_recompute_pending[cat.agent_key] = new_gen

    task = asyncio.create_task(
        _concept_recompute_job(handler, cat, cat.agent_key, new_gen)
    )
    # Tracked so the handler's close() can await it during agent shutdown
    # (pattern from after_cheshire_cat_creation) and so it is not GC'd.
    handler._pending_entity_tasks.append(task)
    log.info(
        f"[GraphRAG] Concept-relation config changed for {cat.agent_key}: "
        f"scheduled recompute to generation {new_gen[:12]}"
    )


async def _concept_recompute_job(
    handler: GraphRAGHandler, cat: StrayCat, tenant: str, new_gen: str
) -> None:
    """
    Single-flight concept recompute for one tenant.

    Latest-wins: every pass re-reads ``_concept_recompute_pending`` and the
    persisted marker. The in-process lock only serializes pending-map access;
    single-flight ACROSS handler re-instantiations holds because a losing flip
    (gen-guard inside ``_flip_concept_gen``) aborts and this loop re-derives
    from the marker read and the newest pending generation. A flip that commits
    is followed by the deferred GC of stale old-generation concept nodes.
    """
    attempts = 0
    while attempts < 4:
        attempts += 1
        async with _concept_recompute_lock:
            pending = _concept_recompute_pending.get(tenant)
        if pending is None:
            return  # superseded work: a newer job took over
        active = await handler._read_concept_gen(tenant)
        if active == pending:
            async with _concept_recompute_lock:
                if _concept_recompute_pending.get(tenant) == pending:
                    _concept_recompute_pending.pop(tenant, None)
            return

        try:
            await handler.recompute_concept_relations(cat, gen=pending)
            flipped = await handler._flip_concept_gen(tenant, active, pending)
        except Exception as e:  # noqa: BLE001
            log.error(f"[GraphRAG] Concept recompute failed for {tenant}: {e}")
            return  # keep pending: the next settings save re-triggers it

        if flipped:
            await handler._gc_stale_concept_nodes(tenant)
            async with _concept_recompute_lock:
                if _concept_recompute_pending.get(tenant) == pending:
                    _concept_recompute_pending.pop(tenant, None)
            return

        # Gen-guard aborted: a newer save won the race -> loop re-reads the
        # latest pending generation (latest-wins).
    log.error(
        f"[GraphRAG] Concept recompute for {tenant} gave up after repeated "
        "gen-guard aborts (newer saves keep winning)"
    )
