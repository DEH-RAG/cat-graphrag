#!/usr/bin/env python3
"""
Maintenance agent CLI for the Neo4j GraphRAG plugin.

Standalone, import-safe maintenance script that drives Grinning Cat's
phase-based ingestion machine (feat/ingestion-phase-machine) and the
GraphRAG ``graph_building`` phase from INSIDE the Cat container.

Usage (inside the Cat container):
    # Re-embed via the gen-based phase machine (stale phases recomputed)
    docker exec cheshire_cat_core python /app/maintenance_agent.py --agent <id> --reembed --yes

    # Re-ingest from scratch (points-first wipe, then re-parse + re-embed) for all agents
    docker exec cheshire_cat_core python /app/maintenance_agent.py --all --reingest --yes

    # Drive the graph_building phase (mark it stale, then the phase machine re-runs it)
    docker exec cheshire_cat_core python /app/maintenance_agent.py --agent <id> --graph --yes

    # Explicit destructive wipe of the tenant's graph edges + orphan entities,
    # then drive the graph_building phase
    docker exec cheshire_cat_core python /app/maintenance_agent.py --agent <id> --wipe-graph --yes

    # Combine ops and target the episodic collection
    docker exec cheshire_cat_core python /app/maintenance_agent.py --agent <id> --reingest --graph --collection episodic --yes

    # Show the plan without touching anything (exits 2)
    docker exec cheshire_cat_core python /app/maintenance_agent.py --all --graph --dry-run

    # Run the acceptance checks after the ops (read-only)
    docker exec cheshire_cat_core python /app/maintenance_agent.py --agent <id> --reembed --yes --verify

Exit codes:
    0 = all agents/ops ok (and verify ok when --verify)
    1 = partial failure (some agent/op failed, or a --verify check failed)
    2 = aborted (dry-run plan printed, or --yes missing for a destructive step)

Lifecycle model (feat/ingestion-phase-machine): status docs carry a
``completed_phases`` diary (``[{phase, gen}]``) plus fingerprint dicts
(``chunker`` / ``embedder`` / ``graphrag``). ``reembed_sources`` is the ONE
probe-driven phase machine: per source it probes ``ingestion_phase_pending``
and runs the stale phases serially. A phase is stale when its recorded
generation differs from the recomputed one (settings fingerprint + dependency
generations): a chunker change re-runs ``parsing_chunking`` (and followers),
an embedder-only change re-runs ``embedding`` (and followers), and the graph
work is the first-class ``graph_building`` phase (deps: ``parsing_chunking``
+ ``embedding`` — SIMILAR_TO needs valid vectors). Error rows are absorbing:
once a row reaches ``status=error`` the phase loop stops for it and it is
re-entered only by a fresh claim.

Every op requires --yes: --reembed runs the gen-based phase machine (stale
phases recomputed), --reingest deletes the source's points + status docs and
re-parses from scratch (raw deletion is the only way to force a full re-parse
the gen-based probe would not trigger), --graph marks the ``graph_building``
phase stale and invokes the phase path, and --wipe-graph additionally wipes
the tenant's graph edges + orphan entities (A0 Cypher) first.

IMPORT-SAFETY: this file lives in the plugin folder, which the Cat plugin
loader imports recursively at activation. It must have ZERO top-level side
effects: only stdlib imports at module level; every ``cat`` / ``neo4j``
import happens inside functions.
"""

import argparse
import asyncio
import importlib.util
import json
import sys
from typing import Any, NotRequired, TypedDict

# Destructive steps per op — what `--yes` confirms. `--reembed` runs the
# gen-based phase machine (stale phases recomputed from the settings
# fingerprints + dependency generations); `--reingest` deletes the source's
# points FIRST (points-first wipe) and its status doc, so the probe sees no
# completed_phases and re-parses from scratch; `--graph` marks the
# ``graph_building`` phase stale (strips the recorded generation from the
# diary) and invokes the phase path; `--wipe-graph` additionally wipes the
# tenant's graph edges + orphan entities via the A0 Cypher. Every op therefore
# requires `--yes`; the missing-`--yes` message lists these steps per
# agent+op.
_DESTRUCTIVE_STEPS: dict[str, list[str]] = {
    "reingest": [
        "delete_tenant_points per source (points-first wipe)",
        "delete_status per source (forces a full re-parse: the gen-based probe "
        "only re-runs phases whose settings/deps changed)",
        "reembed_sources phase dispatch (parsing_chunking -> embedding -> "
        "graph_building)",
    ],
    "graph": [
        "mark graph_building stale per source (strip the recorded gen from "
        "completed_phases)",
        "reembed_sources phase dispatch (graph_building phase: similarity + "
        "concept relations)",
    ],
    "wipe_graph": [
        "A0 wipe Cypher (5 tenant-filtered statements: RELATED_TO, MENTIONS, "
        "PROVENANCE, SIMILAR_TO_<gen>, orphan entities)",
        "mark graph_building stale per source (strip the recorded gen from "
        "completed_phases)",
        "reembed_sources phase dispatch (graph_building phase)",
    ],
    "reembed": [
        "reembed_sources phase dispatch (gen-based: stale phases recomputed "
        "from settings fingerprints + dependency gens)",
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
            "re-embed or drive the graph_building phase for one agent or all "
            "agents. Must run inside the Cat container."
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
        help="Re-ingest from scratch: delete points per source, then re-parse "
             "and re-embed (raw deletion forces a full re-parse the gen-based "
             "probe would not trigger). DESTRUCTIVE (requires --yes).",
    )
    parser.add_argument(
        "--reembed",
        action="store_true",
        help="Re-embed via the gen-based phase machine: stale phases (embedder "
             "change -> embedding; chunker change -> parsing + followers) are "
             "recomputed from the settings fingerprints. Requires --yes.",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Drive the graph_building phase: mark it stale per source (strip "
             "the recorded generation from completed_phases) and invoke the "
             "phase path via reembed_sources (similarity + concept relations). "
             "Requires --yes.",
    )
    parser.add_argument(
        "--wipe-graph",
        action="store_true",
        help="Explicit destructive step: wipe the tenant's graph edges + "
             "orphan entities (A0 Cypher), then drive the graph_building "
             "phase. DESTRUCTIVE (requires --yes).",
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
        help="Confirm destructive steps (required for every op: status/point "
             "deletes, the graph wipe, re-ingest, phase re-runs).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run acceptance assertions after the ops (read-only; never "
             "against the system agent).",
    )

    args = parser.parse_args()

    # At least one op required; combinable. --wipe-graph is a superset of
    # --graph (the wipe is followed by the same phase drive), so it replaces
    # it when both are given.
    args.ops = [op for op in ("reingest", "reembed", "graph") if getattr(args, op)]
    if args.wipe_graph:
        if "graph" in args.ops:
            args.ops.remove("graph")
        args.ops.append("wipe_graph")
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
    ``system``. When ``--graph`` / ``--wipe-graph`` is requested, the agent
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
    if any(op in ("graph", "wipe_graph") for op in args.ops):
        plugin_id = _resolve_plugin_id()
        if plugin_id is None:
            print(
                "ABORTED: GraphRAG plugin not found in the plugins dir "
                "(--graph/--wipe-graph requires it)"
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
        if any(op in ("graph", "wipe_graph") for op in args.ops) and agent_id not in active:
            entry["skip_reason"] = "plugin-not-active"
        plan.append(entry)
    return plan


def _print_plan(plan: list[_PlanEntry]) -> None:
    """Print the dry-run plan: per agent+op the concrete steps.

    Pure plan: NO bootstrap, NO writes, NO ``CheshireCat.create``. The A0 wipe
    statements (``--wipe-graph``) are printed verbatim (with a ``<gen>``
    placeholder for the run-time generation); the per-source wipe/status-delete
    steps are printed as templates — the actual source names are only known
    after bootstrap, which dry-run must NOT do.
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
                    "      1. for each source <name>: delete_tenant_points("
                    f"'{collection}', metadata={{'source': <name>}})"
                )
                print(
                    "      2. for each source <name>: delete_status("
                    "agent, 'agent', <name>)"
                )
                print(
                    "      3. reembed_sources(ccat, collection, sources)  "
                    "# probe-driven phase dispatch (parsing_chunking -> "
                    "embedding -> graph_building)"
                )
            elif op == "reembed":
                print("    steps:")
                print(
                    "      1. reembed_sources(ccat, collection, sources)  "
                    "# probe-driven phase dispatch (gen-based: stale phases "
                    "recomputed from settings fingerprints + dependency gens)"
                )
            elif op == "graph":
                print("    steps:")
                print(
                    "      1. for each source <name>: strip the graph_building "
                    "entry from completed_phases (mark the phase stale)"
                )
                print(
                    "      2. reembed_sources(ccat, collection, sources)  "
                    "# graph_building phase (similarity + concept relations)"
                )
            elif op == "wipe_graph":
                print("    steps:")
                print("      1. A0 wipe Cypher (5 tenant-filtered statements):")
                for stmt in _graph_wipe_statements("<gen>"):
                    print(f"         {stmt}")
                print(
                    "      2. for each source <name>: strip the graph_building "
                    "entry from completed_phases (mark the phase stale)"
                )
                print(
                    "      3. reembed_sources(ccat, collection, sources)  "
                    "# graph_building phase"
                )
    print("(dry-run: exiting 2, nothing was written)")


async def _bootstrap_agent(agent_id: str) -> tuple[object, object | None, str | None]:
    """Bootstrap one agent for the maintenance ops.

    Creates the agent's ``CheshireCat`` (NOT ``BillTheLizard`` — the lizard
    fires the resume-sweep hooks we want to skip), checks that the vector
    memory handler is the plugin's ``GraphRAGHandler``, connects it, and runs
    ``initialize`` (required for the vector index; on an embedder change it
    launches the seamless shadow-swap ``reembed_tenant``, which flips the
    Epoch token — the flip is logged). Snapshots the current generation on
    the handler as ``_walk_gen`` (consumed by the ``--wipe-graph`` A0 wipe)
    and records the GraphRAG detection (``_graphrag_detected`` /
    ``_concept_relations_enabled`` / ``_enable_derived_graph``) from the
    ``Neo4jGraphRAGConfig`` vector-db setting entry.

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
    # is the observable side effect we log.
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
    # class (same semantics as main.py). The --graph/--wipe-graph op requires
    # the handler's ``_enable_derived_graph`` (the phase gate in main.py's
    # ``ingestion_phase_pending``/``ingestion_phase_run`` hooks); the per-op
    # skip is decided in _run_agent.
    from cat.db.cruds import settings as crud_settings

    entry = await crud_settings.get_setting_by_name(agent_id, "Neo4jGraphRAGConfig")
    handler._graphrag_detected = entry is not None
    kg = bool((entry or {}).get("value", {}).get("enable_knowledge_graph", False))
    cr = bool((entry or {}).get("value", {}).get("enable_concept_relations", False))
    dg = bool((entry or {}).get("value", {}).get("enable_derived_graph", False))
    handler._concept_relations_enabled = bool(kg and cr)
    handler._enable_derived_graph = dg

    return (ccat, handler, None)


async def _resolve_ingestion_engine(ccat, op: str = "reembed") -> str | None:
    """Resolve the active ingestion engine; refuse anything but the efficient one.

    Reads the ``ingestion`` settings category (the saved config name) and
    resolves it against the allowed classes via ``cat.services.factory.ingestion``
    (``resolved_config_name``: saved entry, else first plugin class, else the
    base). Only the efficient engine (``EfficientIngestionConfiguration`` /
    ``EfficientIngestionEngine``) is supported by ``--reembed``/``--reingest``/
    ``--graph``/``--wipe-graph``: it is the ONE probe-driven phase machine that
    dispatches the ``graph_building`` phase; the base engine's re-embed path
    (``embed_all_in_cheshire_cats``) is destructive and out of scope, so it is
    refused with a message naming the active engine.

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


async def _op_reembed(ccat, handler, collection: str) -> bool:
    """Re-embed via the gen-based phase machine (no status-doc deletion).

    ``reembed_sources`` is the ONE probe-driven phase machine: per source it
    probes ``ingestion_phase_pending`` and runs the stale phases serially. A
    phase is stale when its recorded generation (in ``completed_phases``)
    differs from the recomputed one (settings fingerprint + dependency gens):
    an embedder change re-runs ``embedding`` (chunk reuse), a chunker change
    re-runs ``parsing_chunking`` first and ``embedding`` follows on the next
    probe. With unchanged settings nothing is stale and the pass is a no-op.
    Status docs are NOT deleted: raw deletion would make the probe see no
    completed_phases and re-parse everything, which is ``--reingest``'s job.
    Status docs whose source is absent from the enumeration are warned
    (``detail=file-missing``). ``handler`` is accepted for signature symmetry
    with the other ops.
    """
    from cat.core_plugins.efficient_ingestion.reembed import reembed_sources
    from cat.core_plugins.ingestion_status.registry import list_statuses
    from cat.services.memory.models import VectorMemoryType

    agent_id = getattr(ccat, "agent_key", None) or getattr(ccat, "_id", None)

    all_sources = await ccat.get_stored_sources_with_metadata()
    sources = all_sources.get(VectorMemoryType(collection), [])

    # Status docs whose source is absent from the enumeration.
    statuses = await list_statuses(agent_id)
    known = {s.name for s in sources}
    for doc in statuses:
        src = doc.get("source")
        if src and src not in known:
            print(
                f"[maintenance] agent={agent_id} op=reembed result=warn "
                f"detail=file-missing source={src}"
            )

    await reembed_sources(ccat, VectorMemoryType(collection), sources)
    return True


async def _op_reingest(ccat, handler, collection: str) -> bool:
    """Re-ingest from scratch: delete points per source, then re-parse + re-embed.

    Raw deletion is STILL needed here: the gen-based probe only re-runs a
    phase when its recorded generation differs from the recomputed one, and a
    FORCED full re-ingest must re-parse regardless of fingerprints. Deleting
    the source's points FIRST (``handler.delete_tenant_points(str(collection),
    metadata={"source": name})`` — which also triggers the provenance-cascade
    graph cleanup) and THEN the status doc makes the probe see no
    completed_phases → ``parsing_chunking`` pending (clean re-parse) →
    ``embedding`` follows on the next probe → ``graph_building`` follows. URL
    sources are passed through as-is: the engine re-downloads them (the
    ``content=None`` path in ``_source_from_entry``), so their points are NOT
    deleted by source name — each pass-through is logged
    (``detail=url-pass-through``).
    """
    from cat.core_plugins.efficient_ingestion.reembed import reembed_sources
    from cat.core_plugins.ingestion_status.registry import delete_status
    from cat.services.memory.models import VectorMemoryType
    from cat.utils import is_url

    agent_id = getattr(ccat, "agent_key", None) or getattr(ccat, "_id", None)

    all_sources = await ccat.get_stored_sources_with_metadata()
    sources = all_sources.get(VectorMemoryType(collection), [])

    # Points-first: delete the source's points, THEN the status doc, so the
    # probe sees no doc + no points → PHASE_PARSING_CHUNKING.
    for source in sources:
        if is_url(source.name):
            print(
                f"[maintenance] agent={agent_id} op=reingest result=warn "
                f"detail=url-pass-through source={source.name}"
            )
            continue
        await handler.delete_tenant_points(
            str(collection), metadata={"source": source.name}
        )
        await delete_status(agent_id, "agent", source.name)

    await reembed_sources(ccat, VectorMemoryType(collection), sources)
    return True


def _graph_wipe_statements(gen) -> list[str]:
    """The 5 tenant-filtered A0 wipe statements (single source of truth).

    Shared by ``_graph_wipe`` (execution) and the dry-run plan printer
    (``_print_plan``, which passes the literal ``"<gen>"`` placeholder — the
    real generation is only known after bootstrap, which dry-run must NOT
    do). Every statement is tenant-filtered; no ``MATCH (e:Entity)`` without
    ``{tenant_id}``.
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
        #    deleted in step 3, so the cascade-prune condition matches).
        "MATCH (e:Entity {tenant_id: $tenant_id}) WHERE NOT (e)--() DELETE e",
    ]


async def _graph_wipe(handler, tenant_id, gen) -> None:
    """A0 wipe: delete the tenant's fixed-graph edges + orphan entities.

    The fixed part is additive-only (``_extract_and_link_entities`` MERGEs,
    never deletes stale edges), so a plain re-walk would accumulate ghost
    entities and stale relations — a deliberate wipe (``--wipe-graph``) is the
    only way to remove them. No handler helper exists for this — the Cypher is
    defined HERE (``_graph_wipe_statements``), tenant-filtered on every
    statement. The Document/SourceFile/Collection structure is kept: only
    edges and orphan entities are deleted. The SIMILAR_TO relation name is the
    versioned one of the current generation (``SIMILAR_TO_<gen>``).
    """
    async with handler._get_session() as session:
        for stmt in _graph_wipe_statements(gen):
            await session.run(stmt, tenant_id=tenant_id)


async def _mark_graph_building_stale(agent_id: str, source: str, doc: dict[str, Any]) -> bool:
    """Strip the ``graph_building`` entry from a status doc's diary.

    The probe (``ingestion_phase_pending``) declares ``graph_building``
    pending exactly when its recorded generation is absent from
    ``completed_phases`` — so removing the entry forces the phase machine to
    re-run the graph work for the source. ``set_status`` merge semantics
    cannot REMOVE an entry (it merges by phase-id), so the stripped diary is
    written via ``extra`` (wholesale replace). Only settled (``completed``)
    rows are touched: error rows are absorbing (re-entered only by a fresh
    claim) and mid-flight rows are left to their worker. Returns True when
    the diary was stripped, False when the row was skipped.
    """
    from cat.core_plugins.ingestion_status.registry import IngestionStatus, set_status
    from cat.utils import is_url

    status = doc.get("status")
    if status == IngestionStatus.ERROR.value:
        print(
            f"[maintenance] agent={agent_id} op=graph result=warn "
            f"detail=error-row-absorbing source={source}"
        )
        return False
    if status != IngestionStatus.COMPLETED.value:
        print(
            f"[maintenance] agent={agent_id} op=graph result=warn "
            f"detail=row-not-settled status={status} source={source}"
        )
        return False
    completed = [
        e
        for e in (doc.get("completed_phases") or [])
        if not (isinstance(e, dict) and e.get("phase") == "graph_building")
    ]
    await set_status(
        agent_id,
        "agent",
        source,
        type_="url" if is_url(source) else "file",
        status=IngestionStatus.COMPLETED,
        extra={"completed_phases": completed},
    )
    return True


async def _op_graph(ccat, handler, collection: str, wipe: bool = False) -> bool:
    """Drive the ``graph_building`` phase through the phase machine.

    The graph work is now a first-class ingestion phase (deps:
    ``parsing_chunking`` + ``embedding`` — SIMILAR_TO needs valid vectors).
    This op marks the phase stale per source (strips the recorded generation
    from the status doc's ``completed_phases`` diary) and invokes the phase
    path via ``reembed_sources``: the probe declares ``graph_building``
    pending and the dispatcher runs it through the ``ingestion_phase_run``
    hook (``run_graph_building_phase``: joins the source's background NER
    tasks, re-runs similarity relationships for the stored points, and runs
    the LLM concept-relations step when ``enable_knowledge_graph`` AND
    ``enable_concept_relations``). All writes are idempotent MERGEs. Sources
    without a status doc cannot be driven (no diary → the probe cannot declare
    the phase) and are warned (``detail=no-status-doc``).

    ``wipe=True`` (the explicit ``--wipe-graph`` destructive step) runs the
    tenant-filtered A0 wipe Cypher FIRST — the fixed part is additive-only
    (MERGEs never delete stale edges), so a deliberate wipe is the only way to
    remove ghost entities and stale relations.
    """
    from cat.core_plugins.efficient_ingestion.reembed import reembed_sources
    from cat.core_plugins.ingestion_status.registry import list_statuses
    from cat.services.memory.models import VectorMemoryType

    agent_id = getattr(ccat, "agent_key", None) or getattr(ccat, "_id", None)
    assert agent_id is not None  # a bootstrapped CheshireCat always carries its key

    all_sources = await ccat.get_stored_sources_with_metadata()
    sources = all_sources.get(VectorMemoryType(collection), [])

    if wipe:
        gen = getattr(handler, "_walk_gen", None)
        if gen is None:
            gen = await handler._read_generation(agent_id)
            handler._walk_gen = gen
        await _graph_wipe(handler, agent_id, gen)

    # Mark graph_building stale per source: strip the recorded gen from the
    # completed_phases diary so the probe declares the phase pending.
    statuses = await list_statuses(agent_id)
    by_source = {d.get("source"): d for d in statuses if d.get("source")}
    known = {s.name for s in sources}
    for src, doc in by_source.items():
        if src not in known:
            print(
                f"[maintenance] agent={agent_id} op=graph result=warn "
                f"detail=file-missing source={src}"
            )
            continue
        await _mark_graph_building_stale(agent_id, src, doc)
    for s in sources:
        if s.name not in by_source:
            print(
                f"[maintenance] agent={agent_id} op=graph result=warn "
                f"detail=no-status-doc source={s.name}"
            )

    await reembed_sources(ccat, VectorMemoryType(collection), sources)
    return True


async def _run_agent(
    agent_id: str, ops: list[str], collection: str = "declarative"
) -> bool:
    """Run the requested ops for one agent.

    Bootstraps the agent (``CheshireCat.create`` + handler checks +
    ``initialize``) and dispatches each op. ``--reembed`` and ``--reingest``
    resolve the active ingestion engine first (refusing the base engine) and
    then run the phase machine via ``_op_reembed`` (gen-based recompute) /
    ``_op_reingest`` (points-first wipe + clean re-parse); ``--graph`` /
    ``--wipe-graph`` resolve the engine too (the phase machine dispatches the
    ``graph_building`` phase) and run ``_op_graph`` (mark the phase stale +
    invoke the phase path; ``--wipe-graph`` runs the A0 wipe first). Skip
    reasons: ``handler-not-graphrag`` (no GraphRAG handler) and
    ``graphrag-not-enabled`` (the ``--graph``/``--wipe-graph`` op requires the
    ``Neo4jGraphRAGConfig`` setting AND the handler's ``_enable_derived_graph``
    — the phase gate in main.py's hooks).
    Returns True when every op succeeded or was skipped, False otherwise.
    """
    ccat, handler, reason = await _bootstrap_agent(agent_id)
    if reason is not None:
        for op in ops:
            print(f"[maintenance] agent={agent_id} op={op} result=skip reason={reason}")
        return True

    ok = True
    for op in ops:
        if op in ("graph", "wipe_graph") and not (
            getattr(handler, "_graphrag_detected", False)
            and getattr(handler, "_enable_derived_graph", False)
        ):
            print(
                f"[maintenance] agent={agent_id} op={op} result=skip "
                f"reason=graphrag-not-enabled"
            )
            continue
        if op in ("reembed", "reingest", "graph", "wipe_graph"):
            engine = await _resolve_ingestion_engine(ccat, op=op)
            if engine is None:
                ok = False
                continue
        if op == "reembed":
            try:
                await _op_reembed(ccat, handler, collection)
            except Exception as exc:  # noqa: BLE001 - per-op isolation
                print(f"[maintenance] agent={agent_id} op={op} result=fail detail={exc}")
                ok = False
                continue
            print(f"[maintenance] agent={agent_id} op={op} result=ok detail=reembedded")
            continue
        if op == "reingest":
            try:
                await _op_reingest(ccat, handler, collection)
            except Exception as exc:  # noqa: BLE001 - per-op isolation
                print(f"[maintenance] agent={agent_id} op={op} result=fail detail={exc}")
                ok = False
                continue
            print(f"[maintenance] agent={agent_id} op={op} result=ok detail=reingested")
            continue
        if op in ("graph", "wipe_graph"):
            try:
                await _op_graph(ccat, handler, collection, wipe=(op == "wipe_graph"))
            except Exception as exc:  # noqa: BLE001 - per-op isolation
                print(f"[maintenance] agent={agent_id} op={op} result=fail detail={exc}")
                ok = False
                continue
            print(f"[maintenance] agent={agent_id} op={op} result=ok detail=graph")
            continue
    return ok


async def _verify_agent(
    ccat, handler, collection: str, gen: str | None = None
) -> list[tuple[str, bool]]:
    """Run the acceptance checks (plan Success criteria).

    Read-only assertions against the agent's status docs and the Neo4j
    graph (via ``handler._get_session()``):

      1. status docs completed with the active fingerprint dicts and the
         ``completed_phases`` diary (``parsing_chunking`` + ``embedding``,
         plus ``graph_building`` when the graphrag phase is enabled);
      2. zero ``Document.embedding_<gen> IS NULL``;
      3. ``MENTIONS`` edge count > 0 and ``SIMILAR_TO_<gen>`` edge count > 0;
      4. ``Collection.concept_gen_active == handler._concept_fingerprint()``
         (same read as ``_read_concept_gen``);
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

    # 1. status docs completed with the active fingerprint dicts and the
    #    completed_phases diary (parsing_chunking + embedding [+ graph_building
    #    when the graphrag phase is enabled]).
    statuses = await list_statuses(tenant_id)
    graphrag_expected = bool(
        getattr(handler, "_graphrag_detected", False)
        and getattr(handler, "_enable_derived_graph", False)
    )

    def _has_phase(doc, phase: str) -> bool:
        return any(
            isinstance(e, dict) and e.get("phase") == phase and e.get("gen")
            for e in (doc.get("completed_phases") or [])
        )

    checks.append(
        (
            "status-completed-phases",
            bool(statuses)
            and all(
                doc.get("status") == "completed"
                and doc.get("chunker") is not None
                and doc.get("embedder") is not None
                and _has_phase(doc, "parsing_chunking")
                and _has_phase(doc, "embedding")
                and (not graphrag_expected or _has_phase(doc, "graph_building"))
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

    # 4. Collection.concept_gen_active == fingerprint.
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
        # Sequential per-agent execution: agents are processed one at a time,
        # so no two ops ever run concurrently on the same agent.
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