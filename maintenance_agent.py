#!/usr/bin/env python3
"""
Maintenance agent CLI for the Neo4j GraphRAG plugin.

Standalone, import-safe maintenance script that drives Grinning Cat's
two-phase ingestion engine and the GraphRAG graph rebuild from INSIDE the
Cat container.

Usage (inside the Cat container):
    # Re-embed only (chunk reuse) for one agent
    docker exec cheshire_cat_core python /app/maintenance_agent.py --agent <id> --reembed --yes

    # Re-ingest from scratch (points-first wipe, then re-parse + re-embed) for all agents
    docker exec cheshire_cat_core python /app/maintenance_agent.py --all --reingest --yes

    # Rebuild the GraphRAG graph (wipe-then-rebuild) for one agent
    docker exec cheshire_cat_core python /app/maintenance_agent.py --agent <id> --graph --yes

    # Combine ops and target the episodic collection
    docker exec cheshire_cat_core python /app/maintenance_agent.py --agent <id> --reingest --graph --collection episodic --yes

    # Show the plan without touching anything (exits 2)
    docker exec cheshire_cat_core python /app/maintenance_agent.py --all --graph --dry-run

Exit codes:
    0 = all agents/ops ok
    1 = partial failure (some agent/op failed, logged)
    2 = aborted (dry-run plan printed, or --yes missing for a destructive op)

IMPORT-SAFETY: this file lives in the plugin folder, which the Cat plugin
loader imports recursively at activation. It must have ZERO top-level side
effects: only stdlib imports at module level; every ``cat`` / ``neo4j``
import happens inside functions.
"""

import argparse
import asyncio
import importlib.util
import sys
import types  # noqa: F401  (used by the --graph LLM part in a later todo)
from typing import NotRequired, TypedDict

# Ops that delete or rewrite stored data. `--reingest` wipes points per
# source; `--graph` wipes the tenant's graph edges + orphan entities.
# `--reembed` only recomputes vectors (chunk reuse) — not destructive.
DESTRUCTIVE_OPS = frozenset({"reingest", "graph"})

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
            "re-embed or rebuild the graph for one agent or all agents. "
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
        help="Re-ingest from scratch: delete points per source, then re-parse "
             "and re-embed. DESTRUCTIVE (requires --yes).",
    )
    parser.add_argument(
        "--reembed",
        action="store_true",
        help="Re-embed only: reuse stored chunks, recompute vectors. "
             "Not destructive.",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Rebuild the GraphRAG graph (wipe-then-rebuild: fixed NER/"
             "similarity/derived part + optional LLM concept relations). "
             "DESTRUCTIVE (requires --yes).",
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
        help="Confirm destructive steps (required for --reingest / --graph).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run acceptance assertions after the ops (todo 8).",
    )

    args = parser.parse_args()

    # At least one op required; combinable.
    args.ops = [op for op in ("reingest", "reembed", "graph") if getattr(args, op)]
    if not args.ops:
        parser.error("at least one operation is required: --reingest, --reembed, --graph")

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
    ``system``. When ``--graph`` is requested, the agent set is intersected
    with the agents that have the GraphRAG plugin active; the others get a
    ``skip_reason="plugin-not-active"`` entry.
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
    if "graph" in args.ops:
        plugin_id = _resolve_plugin_id()
        if plugin_id is None:
            print(
                "ABORTED: GraphRAG plugin not found in the plugins dir "
                "(--graph requires it)"
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
        if "graph" in args.ops and agent_id not in active:
            entry["skip_reason"] = "plugin-not-active"
        plan.append(entry)
    return plan


def _print_plan(plan: list[_PlanEntry]) -> None:
    """Print the dry-run plan: per agent+op the concrete steps."""
    print("DRY-RUN PLAN (no writes performed)")
    for entry in plan:
        agent_id = entry["agent_id"]
        if entry.get("skip_reason"):
            print(f"  agent={agent_id} SKIPPED (reason={entry.get('skip_reason')})")
            continue
        collection = entry["collection"]
        for op in entry["ops"]:
            destructive = "yes" if op in DESTRUCTIVE_OPS else "no"
            print(
                f"  agent={agent_id} op={op} collection={collection} "
                f"destructive={destructive}"
            )
            if op == "reingest":
                print(
                    "    steps: delete points per source -> delete status -> "
                    "reembed_sources (clean re-parse + re-embed)"
                )
            elif op == "reembed":
                print(
                    "    steps: delete status per source -> reembed_sources "
                    "(chunk reuse, embedding phase only)"
                )
            elif op == "graph":
                print(
                    "    steps: wipe tenant graph edges + orphan entities -> "
                    "re-walk documents (NER + similarity + derived) -> LLM "
                    "concept relations (if enabled)"
                )
    print("(dry-run: exiting 2, nothing was written)")


async def _run_agent(agent_id: str, ops: list[str]) -> bool:
    """Run the requested ops for one agent.

    STUB — todo 3+ implements the real work (bootstrap, engine, graph walk).
    Logs one grep-able summary line per op. Returns True when every op
    succeeded, False otherwise.
    """
    ok = True
    for op in ops:
        # TODO(todo 3+): real implementation.
        print(f"[maintenance] agent={agent_id} op={op} result=ok detail=stub")
    return ok


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
            if op in DESTRUCTIVE_OPS
        ]
        if destructive:
            print("ABORTED: destructive operations require --yes:")
            for agent_id, op in destructive:
                print(f"  agent={agent_id} op={op}")
            print("Re-run with --yes to confirm (or --dry-run to preview).")
            sys.exit(2)

    results = []
    for entry in runnable:
        try:
            ok = await _run_agent(entry["agent_id"], entry["ops"])
        except Exception as exc:  # noqa: BLE001 - per-agent isolation
            print(f"[maintenance] agent={entry['agent_id']} result=fail detail={exc}")
            ok = False
        results.append(ok)

    if all(results):
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())