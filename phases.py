"""Class-based ingestion phases for the Neo4j GraphRAG plugin.

The two graph phases (``graph_spacy_extraction`` / ``graph_llm_extraction``)
implement the CORE ingestion contract (:class:`cat.rabbit_hole.phases.IngestionPhase`)
and are registered through the single ``rabbithole_ingestion_phases`` hook.
Their bodies delegate to the SAME graph work the plugin has always run
(``run_spacy_extraction_phase`` / ``run_llm_extraction_phase``): this module
adds the ownership contract (material settings, applicability, cleanup), it
never re-implements extraction.

The RabbitHole instantiates a phase class without arguments for every run and
calls it with an :class:`~cat.rabbit_hole.phases.IngestionSource`, so a phase
keeps no state of its own between runs.

Tenant write lock (plugin-owned, not a core abstraction)
--------------------------------------------------------
Neo4j writes for one tenant are serialised by the plugin's own
``tenant_write_lock`` helper (``graph-write:<tenant>`` via the sanctioned
``cat.db.crud.distributed_lock``). The core only serialises SOURCES
(``ingestion-source:<agent>:<source>``); the tenant-wide graph write lock is
graph-specific and stays in the plugin. The phase bodies run under the per-source
lock the RabbitHole already holds and acquire the tenant write lock INSIDE the
handler's write methods (source lock -> tenant write lock, never the reverse),
via the ``tenant_write_locked`` decorator. This module never takes the lock
itself.

Import-safe: ``cat.rabbit_hole.phases`` is imported lazily in a ``try/except``
with a local fallback, so the module stays importable when the host does not
ship the new ingestion protocol. ``graphrag_handler`` is imported lazily (FX-9:
the plugin loader reloads ``main.py`` before ``graphrag_handler.py``, so a
module-level binding would compare against the PRE-reload class and every
``isinstance`` guard would fail).
"""

from __future__ import annotations

from typing import Any

try:
    from cat.rabbit_hole.phases import IngestionPhase, PhaseResult
except Exception:  # noqa: BLE001 - keep the module import-safe on older hosts
    from dataclasses import dataclass
    from typing import List, Literal, Tuple

    @dataclass(frozen=True)
    class PhaseResult:
        status: Literal["done", "not_ready", "skipped", "error"]
        reason: str | None = None
        retry_after: float | None = None
        processed: int | None = None
        expected: int | None = None
        warnings: List[dict] | None = None

    class IngestionPhase:  # type: ignore[no-redef]
        """Minimal stand-in mirroring the core ingestion-phase contract."""

        id: str = ""
        label: str = ""
        depends_on: Tuple[str, ...] = ()
        not_ready_budget_seconds: float | None = None

        async def settings(self, cat) -> Any:
            return None

        async def applies_to(self, source, cat) -> Tuple[bool, str | None]:
            return True, None

        async def process(self, source, cat) -> PhaseResult:
            raise NotImplementedError

        async def delete(self, source, cat) -> None:
            raise NotImplementedError


_STATUSES = ("done", "not_ready", "error", "skipped")


def graphrag_handler(cat):
    """Resolve the active GraphRAG handler for ``cat``, or ``None``.

    The presence of a :class:`GraphRAGHandler` as ``cat.vector_memory_handler``
    IS the "graph backend active for this agent" condition: the factory only
    instantiates it when the Neo4j backend is the agent's selected vector
    memory. A pure-vector agent has a different handler, so both graph phases
    report themselves inapplicable (``skipped``) instead of ``pending``.

    Lazy import (FX-9) + the current class keeps the guard correct across the
    plugin loader's file-by-file reload.
    """
    from .graphrag_handler import GraphRAGHandler

    handler = getattr(cat, "vector_memory_handler", None)
    if isinstance(handler, GraphRAGHandler):
        return handler
    return None


async def _prepare_handler(handler, cat) -> None:
    """Make the handler self-sufficient for a phase run.

    A phase can run without the ``embedding`` phase (a resumed diary, a phase
    retry after a restart), so the agent embedder and the entity extractor are
    injected here too: the same idempotent operations the
    ``before_rabbithole_stores_documents`` hook performs during embedding.
    """
    if hasattr(handler, "embedder"):
        handler.embedder = await cat.embedder()
        align = getattr(handler, "_align_embedder_lazy", None)
        if align is not None:
            await align()
    extractor = getattr(handler, "entity_extractor", None)
    if extractor is not None:
        await extractor.ensure_initialized()


def _as_count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _to_result(phase_id: str, raw: Any) -> PhaseResult:
    """Normalize the handler's status dict into :class:`PhaseResult`."""
    if isinstance(raw, PhaseResult):
        return raw
    if isinstance(raw, dict):
        status = raw.get("status")
        if status in _STATUSES:
            retry_after = raw.get("retry_after")
            return PhaseResult(
                status=status,
                reason=raw.get("reason"),
                retry_after=(
                    float(retry_after)
                    if isinstance(retry_after, (int, float))
                    and not isinstance(retry_after, bool)
                    else None
                ),
                processed=_as_count(raw.get("processed")),
                expected=_as_count(raw.get("expected")),
                warnings=raw.get("warnings") or None,
            )
        return PhaseResult(
            status="error",
            reason=f"phase '{phase_id}' returned invalid status {status!r}",
        )
    return PhaseResult(
        status="error",
        reason=f"phase '{phase_id}' returned {raw!r} (expected a status dict)",
    )


class _GraphPhase(IngestionPhase):
    """Shared settings/applicability/cleanup contract for both graph phases."""

    depends_on: tuple[str, ...] = ("parsing_chunking", "embedding")

    async def settings(self, cat) -> Any:
        """The phase's OWN material configuration (no LLM call).

        The core folds this into the phase tag: when it changes, the phase (and
        its dependents) run again. Reuses the existing fingerprints, so the
        behaviour is unchanged.
        """
        handler = graphrag_handler(cat)
        if handler is None:
            return None
        return await self._fingerprint(handler)

    async def applies_to(self, source, cat) -> tuple[bool, str | None]:
        if graphrag_handler(cat) is None:
            return False, "graph backend is not active for this agent"
        return True, None

    async def process(self, source, cat) -> PhaseResult:
        handler = graphrag_handler(cat)
        if handler is None:
            return PhaseResult(status="skipped", reason="graph backend inactive")
        await _prepare_handler(handler, cat)
        raw = await self._execute(handler, source.name, cat)
        return _to_result(self.id, raw)

    async def delete(self, source, cat) -> None:
        """Remove ONLY this source's graph artefacts (idempotent).

        Delegates to ``purge_source_graph_artifacts``: PROVENANCE/MENTIONS and
        SIMILAR_TO edges of the source's Documents, the source's contribution
        to RELATED_TO ``source_files``, entities left orphaned by that removal
        and the source's SourceFile node. Chunk Documents, other sources and
        other tenants are untouched. Re-running it is a no-op.
        """
        handler = graphrag_handler(cat)
        if handler is None:
            return
        await handler.purge_source_graph_artifacts(source.name)

    async def _fingerprint(self, handler) -> str:
        raise NotImplementedError

    async def _execute(self, handler, source_name: str, cat) -> Any:
        """Delegate the phase body to the existing ingestion code."""
        raise NotImplementedError


class GraphSpacyPhase(_GraphPhase):
    """spaCy extraction phase (NER + link + similarity)."""

    id = "graph_spacy_extraction"
    label = "Graph spaCy extraction"

    async def _execute(self, handler, source_name: str, cat) -> Any:
        return await handler.run_spacy_extraction_phase(source_name, cat=cat)

    async def _fingerprint(self, handler) -> str:
        return handler._spacy_fingerprint()


class GraphConceptPhase(_GraphPhase):
    """LLM concept-relations phase."""

    id = "graph_llm_extraction"
    label = "Graph LLM extraction"
    # An LLM-backed phase must not hold a source for the generic hour.
    not_ready_budget_seconds = 300.0

    async def _execute(self, handler, source_name: str, cat) -> Any:
        return await handler.run_llm_extraction_phase(source_name, cat=cat)

    async def _fingerprint(self, handler) -> str:
        return await handler._concept_fingerprint()


__all__ = [
    "GraphSpacyPhase",
    "GraphConceptPhase",
    "IngestionPhase",
    "PhaseResult",
    "graphrag_handler",
]
