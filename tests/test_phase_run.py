#!/usr/bin/env python3
"""Tri-state contract of MyGRAPH's ``ingestion_phase_run`` (B1).

EffING's phase machine treats a ``None`` return as a hard ERROR (fail-hard for
an unimplemented phase). MyGRAPH must therefore return the documented tri-state:

======================  ============================================
situation               return
======================  ============================================
handler not GraphRAG    ``{"status": "done"}`` (nothing to do)
points not stored yet   ``{"status": "not_ready", "retry_after": 5}``
phase done              ``{"status": "done"}``
unknown phase           ``None`` (genuinely unimplemented -> error)
======================  ============================================

The real plugin ``main.py`` is loaded (its heavy relative modules stubbed) and
driven through the resolved hook function, exactly like the host does.

Standalone script (plugin rules): the host imports every ``.py`` under the
plugin folder, so the ENTIRE body lives inside ``if __name__ == "__main__":``
and importing this file executes zero code. Run it as a plain script from the
host checkout:

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-graphrag.my/tests/test_phase_run.py
"""

if __name__ == "__main__":
    import asyncio
    import importlib.util
    import sys
    import traceback
    import types
    from pathlib import Path

    PLUGIN_DIR = Path("/root/git/plugins/cat-graphrag.my")
    MAIN_PATH = PLUGIN_DIR / "main.py"

    def _cleanup(package_name: str) -> None:
        for name in list(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)

    def _load_plugin_main(package_name: str):
        assert MAIN_PATH.is_file(), f"MyGRAPH plugin not found at {PLUGIN_DIR}"

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
        setattr(handler, "Neo4jGraphRAGConfig", type("Neo4jGraphRAGConfig", (), {}))
        sys.modules[package_name + ".graphrag_handler"] = handler

        extractor = types.ModuleType(package_name + ".entity_extractor")
        setattr(extractor, "EntityExtractor", type("EntityExtractor", (), {}))
        sys.modules[package_name + ".entity_extractor"] = extractor

        spec = importlib.util.spec_from_file_location(package_name + ".main", MAIN_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name + ".main"] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            _cleanup(package_name)
            raise
        return module, _GraphRAGHandler

    def _cat(handler):
        return types.SimpleNamespace(vector_memory_handler=handler)

    def _test_done_when_handler_is_not_graphrag():
        module, _ = _load_plugin_main("_mygraph_run_handler_absent")
        try:
            result = asyncio.run(module.ingestion_phase_run.function(
                "graph_spacy_extraction", "file.pdf", [], _cat(object())
            ))
            assert result == {"status": "done"}, result
        finally:
            _cleanup("_mygraph_run_handler_absent")

    def _test_phase_runs_unconditionally_not_gated_by_retired_flag():
        module, handler_cls = _load_plugin_main("_mygraph_run_unconditional")
        try:
            handler = handler_cls()
            result = asyncio.run(module.ingestion_phase_run.function(
                "graph_spacy_extraction", "file.pdf", [], _cat(handler)
            ))
            assert result == {"status": "done"}, result
            assert handler.spacy_calls == 1, "graph phase must not be gated by a flag"
        finally:
            _cleanup("_mygraph_run_unconditional")

    def _test_passes_through_not_ready_when_points_absent():
        module, handler_cls = _load_plugin_main("_mygraph_run_not_ready")
        try:
            class _NotReady(handler_cls):
                async def run_spacy_extraction_phase(self, source, cat=None):
                    return {"status": "not_ready", "retry_after": 5}

            result = asyncio.run(module.ingestion_phase_run.function(
                "graph_spacy_extraction", "file.pdf", [], _cat(_NotReady())
            ))
            assert result == {"status": "not_ready", "retry_after": 5}, result
        finally:
            _cleanup("_mygraph_run_not_ready")

    def _test_unknown_phase_is_fail_hard_none():
        module, handler_cls = _load_plugin_main("_mygraph_run_unknown")
        try:
            result = asyncio.run(module.ingestion_phase_run.function(
                "embedding", "file.pdf", [], _cat(handler_cls())
            ))
            assert result is None, result
        finally:
            _cleanup("_mygraph_run_unknown")

    _tests = [
        ("test_done_when_handler_is_not_graphrag", _test_done_when_handler_is_not_graphrag),
        ("test_phase_runs_unconditionally_not_gated_by_retired_flag",
         _test_phase_runs_unconditionally_not_gated_by_retired_flag),
        ("test_passes_through_not_ready_when_points_absent",
         _test_passes_through_not_ready_when_points_absent),
        ("test_unknown_phase_is_fail_hard_none", _test_unknown_phase_is_fail_hard_none),
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
