#!/usr/bin/env python3
"""
Maintenance agent CLI for the Neo4j GraphRAG plugin.

Standalone, import-safe maintenance script that drives Grinning Cat's
phase-based ingestion machine (the efficient ingestion engine) and the
GraphRAG ``graph_building`` phase from INSIDE the Cat container.

All ops are gen-based recomputes on the new lifecycle: a phase is stale
exactly when its recorded generation (in the status doc's ``completed_phases``
diary) differs from the recomputed one — or is absent. The maintenance agent
marks a phase stale by surgically dropping its diary entry, then drives the
ONE phase machine (``reembed_sources``), which probes
``ingestion_phase_pending`` and runs every stale phase serially
(``parsing_chunking`` / ``embedding`` built-in; ``graph_building`` dispatched
through ``ingestion_phase_run`` to this plugin's handler).

Usage (inside the Cat container):
    # Re-embed only (embedding phase, chunk reuse) for one agent
    docker exec cheshire_cat_core python /app/maintenance_agent.py --agent <id> --reembed

    # Re-ingest from scratch (parsing_chunking clean-sweep, then re-parse +
    # re-embed + graph_building) for all agents
    docker exec cheshire_cat_core python /app/maintenance_agent.py --all --reingest --yes

    # Rebuild the GraphRAG graph via the graph_building phase (idempotent
    # recompute of similarity + concept relations) for one agent
    docker exec cheshire_cat_core python /app/maintenance_agent.py --agent <id> --graph

    # Wipe the tenant's graph edges + orphan entities (deliberate destructive
    # step; rebuild with --graph or --reingest)
    docker exec cheshire_cat_core python /app/maintenance_agent.py --agent <id> --wipe-graph --yes

    # Wipe then rebuild via the phase machine
    docker exec cheshire_cat_core python /app/maintenance_agent.py --agent <id> --wipe-graph --graph --yes

    # Combine ops and target the episodic collection
    docker exec cheshire_cat_core python /app/maintenance_agent.py --agent <id> --reingest --graph --collection episodic --yes

    # Show the plan without touching anything (exits 2)
    docker exec cheshire_cat_core python /app/maintenance_agent.py --all --graph --dry-run

    # Run the acceptance checks after the ops (read-only)
    docker exec cheshire_cat_core python /app/maintenance_agent.py --agent <id> --reembed --verify

Exit codes:
    0 = all agents/ops ok (and verify ok when --verify)
    1 = partial failure (some agent/op failed, or a --verify check failed)
    2 = aborted (dry-run plan printed, or --yes missing for a destructive step)

Destructive ops require --yes: --reingest (the parsing_chunking clean-sweep
deletes the source's text/image points and saved image files) and
--wipe-graph (A0 Cypher: tenant's graph edges + orphan entities). --reembed
and --graph are idempotent recomputes (nothing is deleted) and run without
confirmation; preview them with --dry-run.

IMPORT-SAFETY: this file lives in the plugin folder, which the Cat plugin
loader imports recursively at activation. It must have ZERO top-level side
effects: only stdlib imports at module level; every ``cat`` / ``neo4j``
import happens inside functions.
"""

import argparse
import asyncio
import importlib.util
import sys
from typing import Any, NotRequired, TypedDict

# Destructive steps per op — what `--yes` confirms. Only the ops that DELETE
# data are listed: `--reingest` (the parsing_chunking phase's clean-sweep
# removes the source's text/image points and saved image files before
# re-parsing) and `--wipe-graph` (the A0 Cypher wipes the tenant's graph
# edges + orphan entities). `--reembed` and `--graph` are idempotent
# recomputes (status-diary writes + MERGEs only) and do not require `--yes`.
_DESTRUCTIVE_STEPS: dict[str, list[str]] = {
    "reingest": [
        "parsing_chunking clean-sweep per source (deletes text/image points "
        "and saved image files), then re-parse + re-embed + graph_building",
    ],
    "wipe_graph": [
        "A0 wipe Cypher (5 tenant-filtered statements: RELATED_TO, MENTIONS, "
        "PROVENANCE, SIMILAR_TO_<gen>, orphan entities)",
    ],
}

# The admin's default agent (cat.db.database.DEFAULT_SYSTEM_KEY) is always
# skipped by the enumeration; the legacy `default` agent is kept with a
# one-time warning (module-level flag: warn once per process, not per call).
_default_agent_warned = False


class _PlanEntry(TypedDict):
    """One per-agent plan entry: target, ops, collection, optional skip."""

    agent_id: str
    ops: list[str]
    collection: str
    # Present when the agent is excluded from the run (e.g. the GraphRAG
    # plugin is not active for it while `--graph` was requested).
    skip_reason: NotRequired[str]


def _parse_args() -> argparse.Namespace:
    """Parse the CLI: scope (--agent|--all), ops, collection, safety flags."""
    parser = argparse.ArgumentParser(
        prog="maintenance_agent",
        description=(
            "Agent maintenance for the Neo4j GraphRAG plugin: re-ingest, "
            "re-embed, rebuild or wipe the graph for one agent or all agents. "
            "Must run inside the Cat container."
        ),
    )

    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--agent",
        metavar="ID",
        help="Target a single agent by id.",
    )
    scope.add_argument(
        "--all",
        action="store_true",
        help="Target all agents (the system agent is always skipped).",
    )

    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Re-ingest from scratch: mark the parsing_chunking phase stale "
             "per source and drive the phase machine (clean-sweep of the "
             "source's points, then re-parse + re-embed + graph_building). "
             "DESTRUCTIVE (requires --yes).",
    )
    parser.add_argument(
        "--reembed",
        action="store_true",
        help="Re-embed only: mark the embedding phase stale per source and "
             "drive the phase machine (chunk reuse — stored chunks are kept, "
             "vectors are recomputed; graph_building follows). Idempotent.",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Rebuild the GraphRAG graph via the graph_building phase: mark "
             "it stale per source and drive the phase machine (idempotent "
             "recompute of similarity + concept relations; nothing is "
             "deleted).",
    )
    parser.add_argument(
        "--wipe-graph",
        action="store_true",
        help="Wipe the tenant's graph edges + orphan entities (A0 Cypher). "
             "Deliberate destructive step (requires --yes); rebuild with "
             "--graph or --reingest.",
    )

    parser.add_argument(
        "--collection",
        choices=["declarative", "episodic"],
        default="declarative",
        help="Memory collection to operate on (default: declarative).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan (per agent+op steps) and exit 2 without any write.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm destructive steps (--reingest clean-sweep, "
             "--wipe-graph). --reembed/--graph are idempotent recomputes "
             "and do not require it.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run acceptance assertions after the ops (read-only; never "
             "against the system agent).",
    )

    args = parser.parse_args()

    # At least one op required; combinable. wipe_graph precedes graph so a
    # combined `--wipe-graph --graph` wipes FIRST, then rebuilds via the
    # phase machine.
    args.ops = [
        op
        for op in ("reingest", "reembed", "wipe_graph", "graph")
        if getattr(args, op)
    ]
    if not args.ops:
        parser.error(
            "at least one operation is required: --reingest, --reembed, "
            "--graph, --wipe-graph"
        )

    return args


def _runtime_guard() -> bool:
    """Check we are running inside the Cat runtime.

    The script drives the Cat ingestion engine and the GraphRAG handler, so
    it must run inside the container (``docker exec cheshire_cat_core``).
    Returns True when ``cat`` / ``cat.db.database`` are importable; otherwise
    prints the required invocation and returns False.
    """
    try:
        if importlib.util.find_spec("cat") is None:
            raise ImportError("cat not importable")
        if importlib.util.find_spec("cat.db.database") is None:
            raise ImportError("cat.db.database not importable")
    except (ImportError, ValueError, AttributeError):
        print(
            "ERROR: the Cat runtime is not importable — this script must run "
            "inside the Cat container.\n"
            "Run it as:\n"
            "    docker exec cheshire_cat_core python /app/maintenance_agent.py "
            "--agent <id> --reembed --yes"
        )
        return False
    return True


async def _list_agents() -> list[str]:
    """Enumerate agent ids from Redis (read-only).

    ``get_agents_main_keys()`` scans ``agents:*`` and returns the unique
    second key segment. The ``system`` agent (the admin's default,
    ``cat.db.database.DEFAULT_SYSTEM_KEY``) is always skipped; the legacy
    ``default`` agent is kept with a one-time warning.
    """
    global _default_agent_warned

    from cat.db.cruds import settings as crud_settings

    ids = await crud_settings.get_agents_main_keys()

    agents: list[str] = []
    for agent_id in ids:
        if agent_id == "system":
            continue
        if agent_id == "default" and not _default_agent_warned:
            print(
                "[maintenance] warn: agent 'default' is the legacy default "
                "agent — included in the plan"
            )
            _default_agent_warned = True
        agents.append(agent_id)
    return agents


def _resolve_plugin_id() -> str | None:
    """Resolve the deployed GraphRAG plugin folder name at runtime.

    Scans the Cat plugins dir (``cat/plugins``) plus the optional
    ``CAT_PLUGINS_DIR`` override for a folder containing a
    ``graphrag_handler.py`` that defines ``GraphRAGHandler``. Returns the
    folder name (the deployed plugin id, e.g. ``cat_graphrag``) or None when
    the plugin is not deployed. Never hardcoded: the repo folder is
    ``cat-graphrag.my`` but the deployed id is ``cat_graphrag``.
    """
    import os

    from cat.env import get_env
    from cat.utils import get_plugins_path

    candidates = []
    try:
        candidates.append(get_plugins_path())
    except Exception:  # noqa: BLE001 - a broken cat.utils must not kill the scan
        pass
    env_dir = get_env("CAT_PLUGINS_DIR")
    if env_dir:
        candidates.append(env_dir)

    for plugins_dir in candidates:
        if not os.path.isdir(plugins_dir):
            continue
        for folder in sorted(os.listdir(plugins_dir)):
            handler_path = os.path.join(plugins_dir, folder, "graphrag_handler.py")
            if not os.path.isfile(handler_path):
                continue
            try:
                with open(handler_path, encoding="utf-8", errors="ignore") as fh:
                    if "class GraphRAGHandler" in fh.read():
                        return folder
            except OSError:
                continue
    return None


async def _build_ops_plan(args: argparse.Namespace) -> list[_PlanEntry]:
    """Build the per-agent ops plan.

    ``--agent <id>`` validates the id against the live agent list (aborts
    with exit 2 when unknown); ``--all`` expands to every agent minus
    ``system``. When ``--graph`` or ``--wipe-graph`` is requested, the agent
    set is intersected with the agents that have the GraphRAG plugin active;
    the others get a ``skip_reason="plugin-not-active"`` entry.
    """
    agents = await _list_agents()

    if args.agent:
        if args.agent not in agents:
            print(f"ABORTED: unknown agent {args.agent}")
            sys.exit(2)
        agent_ids = [args.agent]
    else:
        agent_ids = agents

    active: set[str] = set()
    if "graph" in args.ops or "wipe_graph" in args.ops:
        plugin_id = _resolve_plugin_id()
        if plugin_id is None:
            print(
                "ABORTED: GraphRAG plugin not found in the plugins dir "
                "(--graph/--wipe-graph require it)"
            )
            sys.exit(2)
        from cat.db.cruds import plugins as crud_plugins

        active = set(await crud_plugins.get_agents_plugin_keys(plugin_id))

    plan: list[_PlanEntry] = []
    for agent_id in agent_ids:
        entry: _PlanEntry = {
            "agent_id": agent_id,
            "ops": list(args.ops),
            "collection": args.collection,
        }
        if ("graph" in args.ops or "wipe_graph" in args.ops) and agent_id not in active:
            entry["skip_reason"] = "plugin-not-active"
        plan.append(entry)
    return plan


def _print_plan(plan: list[_PlanEntry]) -> None:
    """Print the dry-run plan: per agent+op the concrete steps.

    Pure plan (Metis #26): NO bootstrap, NO writes, NO ``CheshireCat.create``.
    The A0 wipe statements are printed verbatim (with a ``<gen>`` placeholder
    for the run-time generation); the per-source diary edits are printed as
    templates — the actual source names are only known after bootstrap, which
    dry-run must NOT do.
    """
    print("DRY-RUN PLAN (no writes performed)")
    for entry in plan:
        agent_id = entry["agent_id"]
        if entry.get("skip_reason"):
            print(f"  agent={agent_id} SKIPPED (reason={entry.get('skip_reason')})")
            continue
        collection = entry["collection"]
        for op in entry["ops"]:
            destructive = "yes" if op in _DESTRUCTIVE_STEPS else "no"
            print(
                f"  agent={agent_id} op={op} collection={collection} "
                f"destructive={destructive}"
            )
            if op == "reingest":
                print("    steps:")
                print(
                    "      1. for each source <name>: drop the "
                    "parsing_chunking entry from the completed_phases diary "
                    "(mark stale)"
                )
                print(
                    "      2. reembed_sources(ccat, collection, sources)  "
                    "# phase machine: parsing_chunking clean-sweep + "
                    "re-parse, then embedding + graph_building follow"
                )
            elif op == "reembed":
                print("    steps:")
                print(
                    "      1. for each source <name>: drop the embedding "
                    "entry from the completed_phases diary (mark stale)"
                )
                print(
                    "      2. reembed_sources(ccat, collection, sources)  "
                    "# phase machine: embedding (chunk reuse), then "
                    "graph_building follows"
                )
            elif op == "graph":
                print("    steps:")
                print(
                    "      1. for each source <name>: drop the "
                    "graph_building entry from the completed_phases diary "
                    "(mark stale)"
                )
                print(
                    "      2. reembed_sources(ccat, collection, sources)  "
                    "# phase machine: graph_building via ingestion_phase_run "
                    "(similarity + concept relations)"
                )
            elif op == "wipe_graph":
                print("    steps:")
                print("      1. A0 wipe Cypher (5 tenant-filtered statements):")
                for stmt in _graph_wipe_statements("<gen>"):
                    print(f"         {stmt}")
    print("(dry-run: exiting 2, nothing was written)")


async def _bootstrap_agent(agent_id: str) -> tuple[object, object | None, str | None]:
    """Bootstrap one agent for the maintenance ops.

    Creates the agent's ``CheshireCat`` (NOT ``BillTheLizard`` — the lizard
    fires the resume-sweep hooks we want to skip), checks that the vector
    memory handler is the plugin's ``GraphRAGHandler``, connects it, and runs
    ``initialize`` (required for the vector index; on an embedder change it
    launches the seamless shadow-swap ``reembed_tenant``, which flips the
    Epoch token — the flip is logged). Snapshots the current generation on
    the handler as ``_walk_gen`` (consumed by the ``--graph`` A-helper) and
    records the GraphRAG detection (``_graphrag_detected`` /
    ``_concept_relations_enabled``) from the ``Neo4jGraphRAGConfig``
    vector-db setting entry.

    Returns ``(ccat, handler, None)`` on success, or
    ``(ccat, None, "handler-not-graphrag")`` when the agent's vector memory
    handler is not a ``GraphRAGHandler``.
    """
    from cat.looking_glass.cheshire_cat import CheshireCat

    ccat = await CheshireCat.create(agent_id)
    handler = ccat.vector_memory_handler

    # The deployed plugin id is resolved at runtime (never hardcoded); the
    # handler class is imported from the plugin module so isinstance checks
    # the deployed code, not this script's copy.
    plugin_id = _resolve_plugin_id()
    if plugin_id is None:
        return (ccat, None, "handler-not-graphrag")
    graphrag_module = importlib.import_module(f"cat.plugins.{plugin_id}.graphrag_handler")
    if not isinstance(handler, graphrag_module.GraphRAGHandler):
        return (ccat, None, "handler-not-graphrag")

    # The versioned decorators probe _get_session() BEFORE the decorated body
    # runs, so the driver must be connected first (AGENTS.md known pitfall).
    await handler._ensure_connected()

    # Generation baseline: initialize() may detect an embedder change and run
    # the shadow-swap reembed_tenant, which flips the Epoch token — the flip
    # is the observable side effect we log (Metis #3).
    gen_before = await handler._read_generation(agent_id)

    embedder = await ccat.embedder()
    await handler.initialize(embedder.name, embedder.size)

    gen = await handler._read_generation(agent_id)
    handler._walk_gen = gen
    if gen != gen_before:
        print(
            f"[maintenance] agent={agent_id} detail=initialize-triggered-reembed "
            f"gen={gen_before}->{gen}"
        )

    # GraphRAG detection: the vector-db setting entry named after the config
    # class (same semantics as main.py:137). The --graph op additionally
    # requires enable_knowledge_graph AND enable_concept_relations (D1: the
    # knowledge-graph flag is the master switch for LLM concept extraction);
    # the per-op skip is decided in _run_agent.
    from cat.db.cruds import settings as crud_settings

    entry = await crud_settings.get_setting_by_name(agent_id, "Neo4jGraphRAGConfig")
    handler._graphrag_detected = entry is not None
    kg = bool((entry or {}).get("value", {}).get("enable_knowledge_graph", False))
    cr = bool((entry or {}).get("value", {}).get("enable_concept_relations", False))
    handler._concept_relations_enabled = bool(kg and cr)

    return (ccat, handler, None)


async def _resolve_ingestion_engine(ccat, op: str = "reembed") -> str | None:
    """Resolve the active ingestion engine; refuse anything but the efficient one.

    Reads the ``ingestion`` settings category (the saved config name) and
    resolves it against the allowed classes via ``cat.services.factory.ingestion``
    (``resolved_config_name``: saved entry, else first plugin class, else the
    base). Only the efficient engine (``EfficientIngestionConfiguration`` /
    ``EfficientIngestionEngine``) is supported by ``--reembed``/``--reingest``
    AND ``--graph`` (all three drive the phase machine via
    ``reembed_sources``): the base engine's re-embed path
    (``embed_all_in_cheshire_cats``) is destructive and out of scope, so it
    is refused with a message naming the active engine (Metis #23).
    ``--wipe-graph`` is handler-level and never consults the engine.

    Returns the active config name when the efficient engine is active, else
    None (the caller marks the op failed → exit 1).
    """
    from cat.services.factory.ingestion import resolved_config_name

    agent_id = getattr(ccat, "agent_key", None) or getattr(ccat, "_id", None)
    name = await resolved_config_name(ccat)
    if name != "EfficientIngestionConfiguration":
        print(
            f"[maintenance] agent={agent_id} op={op} result=fail "
            f"detail=engine-not-efficient active={name}"
        )
        return None
    return name


async def _mark_phase_stale(ccat, collection: str, phase: str) -> int:
    """Remove one phase entry from each source's ``completed_phases`` diary.

    The gen-based staleness model: a phase is pending exactly when its
    recorded generation differs from the recomputed one — or is absent.
    Dropping the entry therefore makes the next probe report the phase
    stale, and ``reembed_sources`` re-runs it (plus any follower phases
    whose generation is recomputed against it). This is the surgical
    alternative to deleting the whole status doc, which would force EVERY
    phase stale (a full re-ingest).

    The registry's ``set_status`` only MERGES diary entries by phase-id and
    cannot remove one, so the doc is read via ``get_status``, filtered, and
    stored back through the same official ``cat.db.crud.store`` the registry
    itself uses (no raw Redis). Error rows keep their ``status=error`` (only
    the diary entry is dropped); the phase machine's ``should_stop_for_error``
    guard prevents any resurrection.

    Returns the number of sources whose diary was modified.
    """
    from cat.core_plugins.ingestion_status.registry import get_status, status_key
    from cat.db import crud
    from cat.services.memory.models import VectorMemoryType

    agent_id = getattr(ccat, "agent_key", None) or getattr(ccat, "_id", None)
    all_sources = await ccat.get_stored_sources_with_metadata()
    sources = all_sources.get(VectorMemoryType(collection), [])

    marked = 0
    for source in sources:
        doc = await get_status(agent_id, "agent", source.name)
        if doc is None:
            continue
        completed = doc.get("completed_phases") or []
        filtered = [
            e
            for e in completed
            if not (isinstance(e, dict) and e.get("phase") == phase)
        ]
        if len(filtered) == len(completed):
            continue  # phase not recorded: nothing to mark stale
        doc["completed_phases"] = filtered
        await crud.store(status_key(agent_id, "agent", source.name), doc)
        marked += 1
    return marked


async def _op_reembed(ccat, handler, collection: str) -> bool:
    """Re-embed only: mark the ``embedding`` phase stale, then drive the machine.

    Gen-based recompute (new lifecycle): removing the ``embedding`` entry
    from each source's ``completed_phases`` diary makes the probe report it
    stale; ``reembed_sources`` then runs the embedding phase (chunk reuse —
    the stored chunks are untouched, only the vectors are recomputed) and
    the ``graph_building`` phase follows (its generation is recomputed
    against the new embedding generation). Nothing is deleted: no points, no
    status docs. ``handler`` is accepted for signature symmetry with the
    other ops.
    """
    from cat.core_plugins.efficient_ingestion.reembed import reembed_sources
    from cat.services.memory.models import VectorMemoryType

    agent_id = getattr(ccat, "agent_key", None) or getattr(ccat, "_id", None)

    all_sources = await ccat.get_stored_sources_with_metadata()
    sources = all_sources.get(VectorMemoryType(collection), [])
    marked = await _mark_phase_stale(ccat, collection, "embedding")
    print(
        f"[maintenance] agent={agent_id} op=reembed detail=marked-stale "
        f"sources={marked}"
    )
    await reembed_sources(ccat, VectorMemoryType(collection), sources)
    return True


async def _op_reingest(ccat, handler, collection: str) -> bool:
    """Re-ingest from scratch: mark ``parsing_chunking`` stale, then drive.

    Gen-based recompute (new lifecycle): removing the ``parsing_chunking``
    entry from each source's ``completed_phases`` diary makes the probe
    report it stale; ``reembed_sources`` then runs the parsing phase (its
    clean-sweep deletes the source's text/image points and saved image
    files, then re-parses + re-chunks), the ``embedding`` phase follows
    (recomputed against the new parsing generation) and ``graph_building``
    follows it. The old points-first manual wipe is gone: the parsing
    phase's clean-sweep IS the wipe, and URL sources are re-downloaded by
    the phase itself.
    """
    from cat.core_plugins.efficient_ingestion.reembed import reembed_sources
    from cat.services.memory.models import VectorMemoryType

    agent_id = getattr(ccat, "agent_key", None) or getattr(ccat, "_id", None)

    all_sources = await ccat.get_stored_sources_with_metadata()
    sources = all_sources.get(VectorMemoryType(collection), [])
    marked = await _mark_phase_stale(ccat, collection, "parsing_chunking")
    print(
        f"[maintenance] agent={agent_id} op=reingest detail=marked-stale "
        f"sources={marked}"
    )
    await reembed_sources(ccat, VectorMemoryType(collection), sources)
    return True


def _graph_wipe_statements(gen) -> list[str]:
    """The 5 tenant-filtered A0 wipe statements (single source of truth).

    Shared by ``_graph_wipe`` (execution) and the dry-run plan printer
    (``_print_plan``, which passes the literal ``"<gen>"`` placeholder — the
    real generation is only known after bootstrap, which dry-run must NOT
    do, Metis #26). Every statement is tenant-filtered; no ``MATCH
    (e:Entity)`` without ``{tenant_id}``.
    """
    similar_rel = f"SIMILAR_TO_{gen}"
    return [
        # 1. RELATED_TO edges — undirected match catches both directions.
        "MATCH (:Entity {tenant_id: $tenant_id})-[r:RELATED_TO]-() DELETE r",
        # 2. MENTIONS edges (Document -> Entity).
        "MATCH (:Document {tenant_id: $tenant_id})-[r:MENTIONS]->() DELETE r",
        # 3. PROVENANCE edges (Document -> Entity).
        "MATCH (:Document {tenant_id: $tenant_id})-[r:PROVENANCE]->() DELETE r",
        # 4. SIMILAR_TO edges of the walk generation (Document <-> Document).
        f"MATCH (:Document {{tenant_id: $tenant_id}})-[r:{similar_rel}]->"
        f"(:Document {{tenant_id: $tenant_id}}) DELETE r",
        # 5. Orphan entities: once the edges above are gone, entities with NO
        #    edges at all are orphans — including stale ones (PROVENANCE was
        #    deleted in step 3, so the cascade-prune condition matches,
        #    Metis #7).
        "MATCH (e:Entity {tenant_id: $tenant_id}) WHERE NOT (e)--() DELETE e",
    ]


async def _graph_wipe(handler, tenant_id, gen) -> None:
    """Phase A0: wipe the tenant's fixed-graph edges + orphan entities.

    The fixed part is additive-only (``_extract_and_link_entities`` MERGEs,
    never deletes stale edges), so a plain re-walk would accumulate ghost
    entities and stale relations (Metis #1). No handler helper exists for
    this — the Cypher is defined HERE (``_graph_wipe_statements``),
    tenant-filtered on every statement. The Document/SourceFile/Collection
    structure is kept: only edges and orphan entities are deleted. The
    SIMILAR_TO relation name is the versioned one of the walk generation
    (``SIMILAR_TO_<gen>``).
    """
    async with handler._get_session() as session:
        for stmt in _graph_wipe_statements(gen):
            await session.run(stmt, tenant_id=tenant_id)


async def _op_graph(ccat, handler, collection: str) -> bool:
    """Rebuild the graph via the ``graph_building`` phase (phase machine).

    Marks the phase stale (removes the ``graph_building`` entry from each
    source's ``completed_phases`` diary) and runs ``reembed_sources``: the
    probe reports ``graph_building`` stale (its dependency phases
    ``parsing_chunking`` + ``embedding`` are recorded), and the dispatcher
    runs it through the ``ingestion_phase_run`` hook →
    ``GraphRAGHandler.run_graph_building_phase`` per source (joins the
    source's background NER/similarity tasks, re-runs the similarity
    relationships for the stored points, and re-extracts the LLM concept
    relations when knowledge-graph + concept relations are enabled). All
    writes are idempotent MERGEs — nothing is deleted. The concept-gen
    marker flip (``concept_gen_active``) is the settings path's job
    (``after_vector_database_settings_update``), not this op's.
    """
    from cat.core_plugins.efficient_ingestion.reembed import reembed_sources
    from cat.services.memory.models import VectorMemoryType

    agent_id = getattr(ccat, "agent_key", None) or getattr(ccat, "_id", None)

    all_sources = await ccat.get_stored_sources_with_metadata()
    sources = all_sources.get(VectorMemoryType(collection), [])
    marked = await _mark_phase_stale(ccat, collection, "graph_building")
    print(
        f"[maintenance] agent={agent_id} op=graph detail=marked-stale "
        f"sources={marked}"
    )
    await reembed_sources(ccat, VectorMemoryType(collection), sources)
    return True


async def _op_wipe_graph(ccat, handler, collection: str) -> bool:
    """Deliberate destructive step: wipe the tenant's graph edges + orphans.

    The old wipe-then-rebuild A0 step, kept ONLY as an explicit destructive
    option (requires ``--yes``). Deletes the tenant's RELATED_TO / MENTIONS /
    PROVENANCE / SIMILAR_TO_<gen> edges and the orphan entities
    (``_graph_wipe_statements``, tenant-filtered on every statement); the
    Document/SourceFile/Collection structure is kept. After a wipe the graph
    is rebuilt by the phase machine: ``--graph`` re-runs the ``graph_building``
    phase (similarity + concept relations), while the NER entity/edge
    extraction runs at ingestion time — so a FULL wipe-then-rebuild needs
    ``--reingest`` (re-parsing re-spawns the NER tasks the phase joins).
    """
    tenant_id = getattr(ccat, "agent_key", None) or getattr(ccat, "_id", None)

    gen = getattr(handler, "_walk_gen", None)
    if gen is None:
        gen = await handler._read_generation(tenant_id)
        handler._walk_gen = gen

    await _graph_wipe(handler, tenant_id, gen)
    return True


async def _run_agent(agent_id: str, ops: list[str], collection: str = "declarative") -> bool:
    """Run the requested ops for one agent.

    Bootstraps the agent (``CheshireCat.create`` + handler checks +
    ``initialize``) and dispatches each op. ``--reembed``, ``--reingest`` and
    ``--graph`` resolve the active ingestion engine first (refusing the base
    engine, Metis #23 — all three drive the phase machine via
    ``reembed_sources``) and then mark the target phase stale + run the
    machine (``_op_reembed`` / ``_op_reingest`` / ``_op_graph``);
    ``--wipe-graph`` runs the A0 wipe Cypher directly (``_op_wipe_graph``,
    no engine needed). Skip reasons: ``handler-not-graphrag`` (no GraphRAG
    handler), ``graphrag-not-detected`` (no ``Neo4jGraphRAGConfig`` setting
    entry) and ``graphrag-not-enabled`` (``--graph`` additionally requires
    ``enable_knowledge_graph`` AND ``enable_concept_relations``).
    Returns True when every op succeeded or was skipped, False otherwise.
    """
    ccat, handler, reason = await _bootstrap_agent(agent_id)
    if reason is not None:
        for op in ops:
            print(f"[maintenance] agent={agent_id} op={op} result=skip reason={reason}")
        return True

    ok = True
    for op in ops:
        if op in ("graph", "wipe_graph") and not getattr(
            handler, "_graphrag_detected", False
        ):
            print(
                f"[maintenance] agent={agent_id} op={op} result=skip "
                f"reason=graphrag-not-detected"
            )
            continue
        if op == "graph" and not getattr(handler, "_concept_relations_enabled", False):
            print(
                f"[maintenance] agent={agent_id} op={op} result=skip "
                f"reason=graphrag-not-enabled"
            )
            continue
        if op in ("reembed", "reingest", "graph"):
            engine = await _resolve_ingestion_engine(ccat, op=op)
            if engine is None:
                ok = False
                continue
        try:
            if op == "reembed":
                await _op_reembed(ccat, handler, collection)
                print(f"[maintenance] agent={agent_id} op={op} result=ok detail=reembedded")
            elif op == "reingest":
                await _op_reingest(ccat, handler, collection)
                print(f"[maintenance] agent={agent_id} op={op} result=ok detail=reingested")
            elif op == "graph":
                await _op_graph(ccat, handler, collection)
                print(f"[maintenance] agent={agent_id} op={op} result=ok detail=graph")
            elif op == "wipe_graph":
                await _op_wipe_graph(ccat, handler, collection)
                print(f"[maintenance] agent={agent_id} op={op} result=ok detail=wiped")
        except Exception as exc:  # noqa: BLE001 - per-op isolation
            print(f"[maintenance] agent={agent_id} op={op} result=fail detail={exc}")
            ok = False
    return ok


async def _recompute_phase_gens(ccat, handler) -> dict[str, Any] | None:
    """Recompute the agent's phase generations from the current config.

    Mirrors the probe logic (``efficient_ingestion.phases`` + the plugin's
    ``ingestion_phase_pending``): ``parsing_chunking`` hashes the chunker
    fingerprint, ``embedding`` the embedder fingerprint + the RECORDED
    parsing generation, ``graph_building`` the graphrag settings fingerprint
    + the concept fingerprint + all recorded dependency generations. Returns
    a context dict consumed by ``_doc_phases_current``, or None when the
    core phase-machine helpers are not deployed (the caller falls back to
    the name-based check).
    """
    try:
        from cat.core_plugins.ingestion_status.fingerprints import (
            build_chunker_fingerprint,
            build_embedder_fingerprint,
            build_graphrag_fingerprint,
            phase_generation,
        )
    except Exception:  # noqa: BLE001 - core patch not deployed
        return None

    agent_id = getattr(ccat, "agent_key", None) or getattr(ccat, "_id", None)
    return {
        "chunker_fp": await build_chunker_fingerprint(agent_id),
        "embedder_fp": await build_embedder_fingerprint(agent_id),
        "graphrag_fp": await build_graphrag_fingerprint(agent_id),
        "concept_fp": handler._concept_fingerprint(),
        "phase_generation": phase_generation,
    }


def _doc_phases_current(doc: dict[str, Any], ctx: dict[str, Any], handler) -> bool:
    """Whether a status doc's recorded generations match the recomputed ones.

    Same staleness rules as the probes: ``parsing_chunking`` must match its
    chunker-hash; ``embedding`` (checked only when parsing is recorded) must
    match its embedder-hash over the recorded parsing generation;
    ``graph_building`` (checked only when both deps are recorded AND the
    handler has the derived graph enabled) must match its graphrag-hash over
    the concept fingerprint and all recorded dependency generations.
    """
    completed = {
        e.get("phase"): e.get("gen")
        for e in (doc.get("completed_phases") or [])
        if isinstance(e, dict) and e.get("phase") is not None
    }
    pg = ctx["phase_generation"]
    if completed.get("parsing_chunking") != pg("parsing_chunking", ctx["chunker_fp"], {}):
        return False
    if "parsing_chunking" in completed and completed.get("embedding") != pg(
        "embedding",
        ctx["embedder_fp"],
        {"parsing_chunking": completed["parsing_chunking"]},
    ):
        return False
    if (
        getattr(handler, "_enable_derived_graph", False)
        and {"parsing_chunking", "embedding"}.issubset(completed)
        and completed.get("graph_building")
        != pg(
            "graph_building",
            {"settings": ctx["graphrag_fp"], "concept": ctx["concept_fp"]},
            {p: g for p, g in completed.items() if p != "graph_building"},
        )
    ):
        return False
    return True


async def _verify_agent(ccat, handler, collection: str, gen: str | None = None) -> list[tuple[str, bool]]:
    """Run the acceptance checks (plan Success criteria, Metis #25).

    Read-only assertions against the agent's status docs and the Neo4j
    graph (via ``handler._get_session()``):

      1. status docs completed with current phase generations (no stale
         phases — the recorded ``completed_phases`` gens match the
         recomputed ones; falls back to the embedder/chunker-name check when
         the core phase-machine helpers are not deployed);
      2. zero ``Document.embedding_<gen> IS NULL``;
      3. ``MENTIONS`` edge count > 0 and ``SIMILAR_TO_<gen>`` edge count > 0;
      4. ``Collection.concept_gen_active == handler._concept_fingerprint()``
         (same read as ``_read_concept_gen``, graphrag_handler.py:2995) —
         checked only when concept relations are enabled (the marker is
         flipped by the settings path, not by the phase machine);
      5. no orphan entities (no MENTIONS/PROVENANCE) — count == 0;
      6. ``SourceFile`` count == source count.

    Returns a list of ``(check_name, ok)`` tuples; the caller prints the
    ``result=verify-ok|verify-fail`` summary. Never destructive.
    """
    from cat.core_plugins.ingestion_status.registry import list_statuses
    from cat.services.memory.models import VectorMemoryType

    tenant_id = getattr(ccat, "agent_key", None) or getattr(ccat, "_id", None)
    if gen is None:
        gen = getattr(handler, "_walk_gen", None) or await handler._read_generation(
            tenant_id
        )

    checks: list[tuple[str, bool]] = []

    # 1. status docs completed with current phase generations.
    statuses = await list_statuses(tenant_id)
    ctx = await _recompute_phase_gens(ccat, handler)
    if ctx is None:
        # Core phase-machine helpers not deployed: fall back to the
        # embedder/chunker-name check.
        embedder = await ccat.embedder()
        chunker = ccat.chunker
        checks.append(
            (
                "status-completed-active-embedder-chunker",
                bool(statuses)
                and all(
                    doc.get("status") == "completed"
                    and doc.get("embedder_name") == embedder.name
                    and doc.get("chunker_name") == chunker.name
                    for doc in statuses
                ),
            )
        )
    else:
        checks.append(
            (
                "status-completed-no-stale-phases",
                bool(statuses)
                and all(
                    doc.get("status") == "completed"
                    and _doc_phases_current(doc, ctx, handler)
                    for doc in statuses
                ),
            )
        )

    async def _scalar(query: str) -> Any:
        async with handler._get_session() as session:
            result = await session.run(query, tenant_id=tenant_id)
            record = await result.single()
            return record["n"] if record is not None else None

    embedding_prop = f"embedding_{gen}"
    similar_rel = f"SIMILAR_TO_{gen}"

    # 2. zero Document.embedding_<gen> IS NULL.
    null_emb = await _scalar(
        f"MATCH (d:Document {{tenant_id: $tenant_id}}) "
        f"WHERE d.{embedding_prop} IS NULL RETURN count(d) AS n"
    )
    checks.append(("no-null-embeddings", null_emb == 0))

    # 3. MENTIONS > 0 and SIMILAR_TO_<gen> > 0.
    mentions = await _scalar(
        "MATCH (:Document {tenant_id: $tenant_id})-[r:MENTIONS]->() "
        "RETURN count(r) AS n"
    )
    similar = await _scalar(
        f"MATCH (:Document {{tenant_id: $tenant_id}})-[r:{similar_rel}]-"
        f">(:Document {{tenant_id: $tenant_id}}) RETURN count(r) AS n"
    )
    checks.append(("mentions-edges", mentions > 0))
    checks.append(("similar-to-edges", similar > 0))

    # 4. Collection.concept_gen_active == fingerprint (only when concept
    #    relations are enabled — the marker is the settings path's flip).
    if getattr(handler, "_concept_relations_enabled", False):
        active_gen = await _scalar(
            "MATCH (c:Collection {tenant_id: $tenant_id}) "
            "RETURN c.concept_gen_active AS n LIMIT 1"
        )
        checks.append(("concept-gen-active", active_gen == handler._concept_fingerprint()))

    # 5. no orphan entities (no MENTIONS/PROVENANCE) — count == 0.
    orphans = await _scalar(
        "MATCH (e:Entity {tenant_id: $tenant_id}) "
        "WHERE NOT (e)-[:MENTIONS]-() AND NOT (e)-[:PROVENANCE]-() "
        "RETURN count(e) AS n"
    )
    checks.append(("no-orphan-entities", orphans == 0))

    # 6. SourceFile count == source count.
    all_sources = await ccat.get_stored_sources_with_metadata()
    sources = all_sources.get(VectorMemoryType(collection), [])
    source_files = await _scalar(
        "MATCH (s:SourceFile {tenant_id: $tenant_id}) RETURN count(s) AS n"
    )
    checks.append(("sourcefile-count", source_files == len(sources)))

    return checks


async def main() -> None:
    """CLI entrypoint: parse -> runtime guard -> plan -> dry-run/yes gates -> run."""
    args = _parse_args()

    if not _runtime_guard():
        sys.exit(2)

    plan = await _build_ops_plan(args)

    if args.dry_run:
        _print_plan(plan)
        sys.exit(2)

    # Skipped agents (e.g. plugin-not-active) are reported, not run.
    runnable = []
    for entry in plan:
        if entry.get("skip_reason"):
            print(
                f"[maintenance] agent={entry['agent_id']} result=skip "
                f"reason={entry.get('skip_reason')}"
            )
        else:
            runnable.append(entry)

    if not args.yes:
        destructive = [
            (entry["agent_id"], op)
            for entry in runnable
            for op in entry["ops"]
            if op in _DESTRUCTIVE_STEPS
        ]
        if destructive:
            print("ABORTED: destructive operations require --yes:")
            for agent_id, op in destructive:
                print(f"  agent={agent_id} op={op} steps:")
                for step in _DESTRUCTIVE_STEPS[op]:
                    print(f"    - {step}")
            print("Re-run with --yes to confirm (or --dry-run to preview).")
            sys.exit(2)

    results = []
    for entry in runnable:
        # Sequential per-agent execution (Metis #18): agents are processed one
        # at a time, so no two ops ever run concurrently on the same agent.
        print(f"[maintenance] agent={entry['agent_id']} progress=start")
        try:
            ok = await _run_agent(
                entry["agent_id"], entry["ops"], entry["collection"]
            )
        except Exception as exc:  # noqa: BLE001 - per-agent isolation
            print(f"[maintenance] agent={entry['agent_id']} result=fail detail={exc}")
            ok = False
        print(f"[maintenance] agent={entry['agent_id']} progress=done")
        results.append(ok)

    # --verify: acceptance checks per agent (read-only; the system agent is
    # never in the plan, so it can never be verified either).
    verify_ok = True
    if args.verify:
        for entry in runnable:
            try:
                ccat, handler, reason = await _bootstrap_agent(entry["agent_id"])
                if reason is not None:
                    print(
                        f"[maintenance] agent={entry['agent_id']} "
                        f"result=verify-skip reason={reason}"
                    )
                    continue
                checks = await _verify_agent(ccat, handler, entry["collection"])
            except Exception as exc:  # noqa: BLE001 - per-agent isolation
                print(
                    f"[maintenance] agent={entry['agent_id']} "
                    f"result=verify-fail detail={exc}"
                )
                verify_ok = False
                continue
            failed = [name for name, ok_ in checks if not ok_]
            if failed:
                verify_ok = False
                print(
                    f"[maintenance] agent={entry['agent_id']} result=verify-fail "
                    f"checks={len(checks) - len(failed)}/{len(checks)} "
                    f"failed={','.join(failed)}"
                )
            else:
                print(
                    f"[maintenance] agent={entry['agent_id']} result=verify-ok "
                    f"checks={len(checks)}/{len(checks)}"
                )
        print(f"[maintenance] result=verify-{'ok' if verify_ok else 'fail'}")

    if all(results) and verify_ok:
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())