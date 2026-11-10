#!/usr/bin/env python3
"""
Regression test: the maintenance CLI decouples ENGINE-level ops from GraphRAG.

``--reingest`` / ``--reembed`` must run for ANY agent (their phases are
engine-level; the graph phases no-op when the active handler is not
GraphRAG). Only ``--graph`` requires the ``GraphRAGHandler`` and must report
``skip reason=handler-not-graphrag`` otherwise. The re-embed preflight must
still block ``--reingest`` before the engine touches any storage.

Run from the host repository (the plugin is importable by file path):

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-graphrag.my/tests/test_maintenance_decoupling.py

IMPORT-SAFETY: this file lives in the plugin folder, which the Cat plugin
loader imports recursively at activation. The ENTIRE body lives under
``if __name__ == "__main__":`` so importing the file executes zero code.
"""

if __name__ == "__main__":
    import asyncio
    import importlib.util
    import io
    import sys
    from contextlib import redirect_stdout
    from pathlib import Path
    from types import SimpleNamespace

    def _load_maintenance_agent():
        path = Path(__file__).resolve().parent.parent / "maintenance_agent.py"
        spec = importlib.util.spec_from_file_location(
            "maintenance_agent_decoupling", path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _fail(msg: str) -> None:
        print(f"FAIL: {msg}")
        sys.exit(1)

    class FakeCcat:
        def __init__(self) -> None:
            self.agent_key = "test-agent"
            self._id = "test-agent"
            self.vector_memory_handler = None
            self.chunker = object()
            self.rabbit_hole = SimpleNamespace(
                reingest_calls=[],
                rerun_calls=[],
            )

        async def get_stored_sources_with_metadata(self):
            from cat.services.memory.models import VectorMemoryType

            return {VectorMemoryType("declarative"): []}

    async def _case_reingest_non_graphrag(m) -> None:
        calls = []

        async def fake_bootstrap(agent_id):
            return (FakeCcat(), None, "handler-not-graphrag")

        async def fake_engine(ccat, op="reembed"):
            return "CoreRabbitHole"

        async def fake_reingest(ccat, handler, collection):
            calls.append((handler, collection))
            return True

        orig = (
            m._bootstrap_agent,
            m._resolve_ingestion_engine,
            m._op_reingest,
        )
        m._bootstrap_agent = fake_bootstrap
        m._resolve_ingestion_engine = fake_engine
        m._op_reingest = fake_reingest
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                ok = await m._run_agent("test-agent", ["reingest"])
            out = buf.getvalue()
        finally:
            (m._bootstrap_agent, m._resolve_ingestion_engine, m._op_reingest) = orig

        if not ok:
            _fail(f"--reingest on a non-GraphRAG agent failed: {out!r}")
        if len(calls) != 1:
            _fail(f"--reingest helper calls={len(calls)}, expected 1: {out!r}")
        if "result=fail" in out:
            _fail(f"--reingest reported a failure: {out!r}")
        print("PASS (a): --reingest planned+executed for non-GraphRAG agent")

    async def _case_graph_non_graphrag_skipped(m) -> None:
        graph_calls = []

        async def fake_bootstrap(agent_id):
            return (FakeCcat(), None, "handler-not-graphrag")

        async def fake_graph_a(ccat, handler, collection):
            graph_calls.append(handler)
            return True

        orig = (m._bootstrap_agent, m._op_graph_part_a)
        m._bootstrap_agent = fake_bootstrap
        m._op_graph_part_a = fake_graph_a
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                ok = await m._run_agent("test-agent", ["graph"])
            out = buf.getvalue()
        finally:
            m._bootstrap_agent, m._op_graph_part_a = orig

        if not ok:
            _fail(f"--graph skip was not a success: {out!r}")
        if graph_calls:
            _fail("--graph ran graph work on a non-GraphRAG agent")
        if "op=graph result=skip reason=handler-not-graphrag" not in out:
            _fail(f"missing handler-not-graphrag skip: {out!r}")
        print("PASS (b): --graph skipped with reason=handler-not-graphrag")

    async def _case_preflight_blocks(m) -> None:
        # 1. The real preflight refuses an unusable re-embed path (no engine).
        ccat = FakeCcat()
        ccat.rabbit_hole = None
        reason = await m._preflight_reembed_path(ccat)
        if reason != "rabbit-hole-unbound":
            _fail(f"preflight reason={reason!r}, expected rabbit-hole-unbound")

        # 2. _op_reingest aborts before calling the engine when the preflight
        #    reports the path unusable.
        ccat = FakeCcat()
        orig_preflight = m._preflight_reembed_path

        async def fake_preflight(_ccat):
            return "rabbit-hole-unbound"

        m._preflight_reembed_path = fake_preflight
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                ok = await m._op_reingest(ccat, None, "declarative")
            out = buf.getvalue()
        finally:
            m._preflight_reembed_path = orig_preflight

        if ok:
            _fail(f"_op_reingest succeeded despite preflight failure: {out!r}")
        if ccat.rabbit_hole.reingest_calls:
            _fail(
                "preflight failure still called the engine: "
                f"{ccat.rabbit_hole.reingest_calls!r}"
            )
        if "detail=preflight-reembed-path-unusable" not in out:
            _fail(f"missing preflight failure message: {out!r}")
        print("PASS (c): preflight still blocks --reingest before any write")

    async def _case_reingest_graphrag_unchanged(m) -> None:
        calls = []
        graph_handler = SimpleNamespace(
            _graphrag_detected=True,
            _concept_relations_enabled=True,
        )

        async def fake_bootstrap(agent_id):
            return (FakeCcat(), graph_handler, None)

        async def fake_engine(ccat, op="reembed"):
            return "CoreRabbitHole"

        async def fake_reingest(ccat, handler, collection):
            calls.append((handler, collection))
            return True

        orig = (m._bootstrap_agent, m._resolve_ingestion_engine, m._op_reingest)
        m._bootstrap_agent = fake_bootstrap
        m._resolve_ingestion_engine = fake_engine
        m._op_reingest = fake_reingest
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                ok = await m._run_agent("test-agent", ["reingest"])
            out = buf.getvalue()
        finally:
            (m._bootstrap_agent, m._resolve_ingestion_engine, m._op_reingest) = orig

        if not ok or len(calls) != 1:
            _fail(f"GraphRAG --reingest changed: {out!r}")
        if calls[0][0] is not graph_handler:
            _fail("GraphRAG --reingest did not receive the GraphRAG handler")
        print("PASS (d): --reingest for a GraphRAG agent is unchanged")

    async def _main() -> int:
        m = _load_maintenance_agent()
        try:
            await _case_reingest_non_graphrag(m)
            await _case_graph_non_graphrag_skipped(m)
            await _case_preflight_blocks(m)
            await _case_reingest_graphrag_unchanged(m)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 - report and fail
            _fail(f"{type(exc).__name__}: {exc}")
        print("PASS: maintenance engine ops decoupled from the GraphRAG handler")
        return 0

    raise SystemExit(asyncio.run(_main()))
