#!/usr/bin/env python3
"""Status contract of MyGRAPH's core ingestion phases.

The core records whatever ``IngestionPhase.process`` returns:

======================  ============================================
situation               result
======================  ============================================
handler not GraphRAG    ``skipped`` (nothing to do for this agent)
points not stored yet   ``not_ready`` with ``retry_after``
phase done              ``done`` (with the real product count)
LLM failure             ``error`` with the verbatim reason
======================  ============================================

The real plugin ``phases.py`` is loaded (its handler module stubbed) and driven
through ``process`` exactly like the host does.

Standalone script (plugin rules): the host imports every ``.py`` under the
plugin folder, so the ENTIRE body lives inside ``if __name__ == "__main__":``
and importing this file executes zero code. Run it as a plain script from the
host checkout:

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-graphrag.my/tests/test_phase_run.py
"""

if __name__ == "__main__":
    import asyncio
    import os
    import sys
    import traceback
    import types
    from pathlib import Path

    PLUGIN_DIR = Path("/root/git/plugins/cat-graphrag.my")

    def _cleanup(package_name: str) -> None:
        for name in list(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)

    def _load_phases(package_name: str):
        assert PLUGIN_DIR.is_dir(), f"MyGRAPH plugin not found at {PLUGIN_DIR}"

        package = types.ModuleType(package_name)
        package.__path__ = [str(PLUGIN_DIR)]
        package.__package__ = package_name
        sys.modules[package_name] = package

        class _GraphRAGHandler:
            def __init__(self):
                self.spacy_calls = 0
                self.llm_calls = 0

            async def run_spacy_extraction_phase(self, source, cat=None):
                self.spacy_calls += 1
                return {"status": "done"}

            async def run_llm_extraction_phase(self, source, cat=None):
                self.llm_calls += 1
                return {"status": "done"}

        handler = types.ModuleType(package_name + ".graphrag_handler")
        setattr(handler, "GraphRAGHandler", _GraphRAGHandler)
        sys.modules[package_name + ".graphrag_handler"] = handler

        from importlib import import_module

        phases = import_module(package_name + ".phases")
        # These tests target the status mapping, not embedder/extractor setup.
        async def _no_prepare(_handler, _cat):
            return None

        phases._prepare_handler = _no_prepare
        return phases, _GraphRAGHandler

    class _Source:
        def __init__(self, name="file.pdf"):
            self.name = name
            self.agent_id = "agent_test"
            self.chat_id = None
            self.type = "file"

    def _cat(handler):
        async def _embedder():
            return None

        return types.SimpleNamespace(
            vector_memory_handler=handler, embedder=_embedder
        )

    def _test_done_when_handler_is_not_graphrag():
        phases, _ = _load_phases("_mygraph_run_handler_absent")
        try:
            result = asyncio.run(
                phases.GraphSpacyPhase().process(_Source(), _cat(object()))
            )
            assert result.status == "skipped", result
            assert "inactive" in (result.reason or ""), result
        finally:
            _cleanup("_mygraph_run_handler_absent")

    def _test_phase_runs_unconditionally_not_gated_by_retired_flag():
        phases, handler_cls = _load_phases("_mygraph_run_unconditional")
        try:
            handler = handler_cls()
            result = asyncio.run(
                phases.GraphSpacyPhase().process(_Source(), _cat(handler))
            )
            assert result.status == "done", result
            assert handler.spacy_calls == 1, "graph phase must not be gated by a flag"
        finally:
            _cleanup("_mygraph_run_unconditional")

    def _test_passes_through_not_ready_when_points_absent():
        phases, handler_cls = _load_phases("_mygraph_run_not_ready")
        try:
            class _NotReady(handler_cls):
                async def run_spacy_extraction_phase(self, source, cat=None):
                    return {"status": "not_ready", "retry_after": 5}

            result = asyncio.run(
                phases.GraphSpacyPhase().process(_Source(), _cat(_NotReady()))
            )
            assert result.status == "not_ready", result
            assert result.retry_after == 5, result
        finally:
            _cleanup("_mygraph_run_not_ready")

    def _test_error_reason_is_passed_through():
        phases, handler_cls = _load_phases("_mygraph_run_unknown")
        try:
            class _Failed(handler_cls):
                async def run_llm_extraction_phase(self, source, cat=None):
                    return {"status": "error", "reason": "no LLM configured"}

            result = asyncio.run(
                phases.GraphConceptPhase().process(_Source(), _cat(_Failed()))
            )
            assert result.status == "error", result
            assert result.reason == "no LLM configured", result
        finally:
            _cleanup("_mygraph_run_unknown")

    _tests = [
        ("test_done_when_handler_is_not_graphrag", _test_done_when_handler_is_not_graphrag),
        ("test_phase_runs_unconditionally_not_gated_by_retired_flag",
         _test_phase_runs_unconditionally_not_gated_by_retired_flag),
        ("test_passes_through_not_ready_when_points_absent",
         _test_passes_through_not_ready_when_points_absent),
        ("test_error_reason_is_passed_through", _test_error_reason_is_passed_through),
    ]
    _failed = 0
    for _name, _fn in _tests:
        try:
            _fn()
            print(f"PASS {_name}")
        except Exception as _exc:  # noqa: BLE001 - report and continue
            _failed += 1
            traceback.print_exc()
            print(f"FAIL {_name}: {_exc}")

    print(f"\ntest_phase_run: {len(_tests) - _failed}/{len(_tests)} passed")
    raise SystemExit(1 if _failed else 0)

