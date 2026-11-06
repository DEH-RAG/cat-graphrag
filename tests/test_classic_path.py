#!/usr/bin/env python3
"""The classic ingestion path builds the graph only when it is the active engine.

Single rule under test (``main._phase_path_active``): the phase path is active
iff EffING is installed/loadable AND the SYSTEM-level ``ingestion`` setting
resolves to the efficient engine; in every other case the classic callback
(``after_rabbithole_stored_documents`` -> ``create_derived_graph_for_source``)
is the one and only graph writer.

These tests load the real MyGRAPH ``main.py``; its heavy relative modules
(``.graphrag_handler`` / ``.entity_extractor``) are replaced by local fakes, so
no Neo4j/spaCy is required. The ingestion-engine resolver is stubbed because the
host core at the test base does not ship ``resolved_config_name``; the callback
gating is what is asserted here, and the real node/relationship writes are
represented by the handler's counters (stated in the task evidence).

Standalone script (plugin rules): the host imports every ``.py`` under the
plugin folder, so the ENTIRE body lives inside ``if __name__ == "__main__":``
and importing this file executes zero code. Run it as a plain script from the
host checkout:

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-graphrag.my/tests/test_classic_path.py
"""

if __name__ == "__main__":
    import asyncio
    import importlib
    import importlib.util
    import sys
    import traceback
    import types
    from pathlib import Path
    from unittest import mock

    PLUGIN_DIR = Path("/root/git/plugins/cat-graphrag.my")
    MAIN_PATH = PLUGIN_DIR / "main.py"
    EFFING = "cat.core_plugins.cat_efficient_ingestion"

    EFFICIENT = "EfficientIngestionConfiguration"
    CORE = "BaseIngestionConfiguration"

    class FakeGraphRAGHandler:
        """Counts graph writes so both mechanisms can be told apart."""

        def __init__(self):
            self.classic_calls = 0
            self.spacy_calls = 0
            self.llm_calls = 0
            self.nodes = 0
            self.relationships = 0

        async def drain_source_tasks(self, source):
            # The plugin joins the per-source background extraction before the
            # classic write; the fake has no background tasks to join.
            return False

        async def create_derived_graph_for_source(self, source, stored_points, cat):
            self.classic_calls += 1
            self.nodes += 2
            self.relationships += 1

        async def run_spacy_extraction_phase(self, source, cat=None):
            self.spacy_calls += 1
            self.nodes += 1
            self.relationships += 1
            return "done"

        async def run_llm_extraction_phase(self, source, cat=None):
            self.llm_calls += 1
            self.nodes += 1
            self.relationships += 1
            return "done"

    def _cleanup(package_name):
        for name in list(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)

    def _load_plugin_main(package_name):
        package = types.ModuleType(package_name)
        package.__path__ = [str(PLUGIN_DIR)]
        package.__package__ = package_name
        sys.modules[package_name] = package

        handler = types.ModuleType(package_name + ".graphrag_handler")
        handler.GraphRAGHandler = FakeGraphRAGHandler
        handler.Neo4jGraphRAGConfig = type("Neo4jGraphRAGConfig", (), {})
        sys.modules[package_name + ".graphrag_handler"] = handler

        extractor = types.ModuleType(package_name + ".entity_extractor")
        extractor.EntityExtractor = type("EntityExtractor", (), {})
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
        return module

    def _install_resolver(name):
        ingestion = importlib.import_module("cat.services.factory.ingestion")

        async def resolved_config_name(lizard):
            return name

        return mock.patch.object(
            ingestion, "resolved_config_name", resolved_config_name, create=True
        )

    def _hide_effing():
        return mock.patch.dict(sys.modules, {EFFING: None})

    def _cat(handler):
        return types.SimpleNamespace(vector_memory_handler=handler, agent_key="agent_test")

    async def _test_inactive_phase_path_runs_classic_once_and_writes_graph():
        """EffING installed but the system engine is the classic one: one classic
        build, zero phase work, real graph output counters > 0."""
        module = _load_plugin_main("_mygraph_classic_inactive")
        try:
            with _install_resolver(CORE):
                handler = FakeGraphRAGHandler()
                cat = _cat(handler)

                await module.after_rabbithole_stored_documents.function("file.pdf", [], cat)

                assert handler.classic_calls == 1
                assert handler.nodes > 0
                assert handler.relationships > 0
                assert handler.spacy_calls == 0
                assert handler.llm_calls == 0
        finally:
            _cleanup("_mygraph_classic_inactive")

    async def _test_active_phase_path_never_runs_classic_and_phases_do_the_work():
        """EffING installed + system engine efficient: the classic callback fires
        ZERO times and the two phases are the ones writing the graph (no dead path)."""
        module = _load_plugin_main("_mygraph_classic_active")
        try:
            with _install_resolver(EFFICIENT):
                handler = FakeGraphRAGHandler()
                cat = _cat(handler)

                await module.after_rabbithole_stored_documents.function("file.pdf", [], cat)
                assert handler.classic_calls == 0

                await module.ingestion_phase_run.function(
                    "graph_spacy_extraction", "file.pdf", [], cat
                )
                await module.ingestion_phase_run.function(
                    "graph_llm_extraction", "file.pdf", [], cat
                )

                assert handler.classic_calls == 0
                assert handler.spacy_calls == 1
                assert handler.llm_calls == 1
                assert handler.nodes > 0
                assert handler.relationships > 0
        finally:
            _cleanup("_mygraph_classic_active")

    async def _test_phase_path_active_truth_table():
        """uninstalled -> classic; installed+core engine -> classic;
        installed+efficient engine -> phase path."""
        module = _load_plugin_main("_mygraph_classic_truth_table")
        try:
            handler = FakeGraphRAGHandler()
            cat = _cat(handler)

            with _install_resolver(EFFICIENT), _hide_effing():
                assert await module._phase_path_active(cat) is False

            with _install_resolver(CORE):
                assert await module._phase_path_active(cat) is False

            with _install_resolver(EFFICIENT):
                assert await module._phase_path_active(cat) is True
        finally:
            _cleanup("_mygraph_classic_truth_table")

    _tests = [
        ("test_inactive_phase_path_runs_classic_once_and_writes_graph",
         _test_inactive_phase_path_runs_classic_once_and_writes_graph),
        ("test_active_phase_path_never_runs_classic_and_phases_do_the_work",
         _test_active_phase_path_never_runs_classic_and_phases_do_the_work),
        ("test_phase_path_active_truth_table", _test_phase_path_active_truth_table),
    ]
    _failed = 0
    for _name, _fn in _tests:
        try:
            asyncio.run(_fn())
            print(f"PASS {_name}")
        except Exception as _exc:  # noqa: BLE001 - report and continue
            _failed += 1
            traceback.print_exc()
            print(f"FAIL {_name}: {_exc}")

    print(f"\ntest_classic_path: {len(_tests) - _failed}/{len(_tests)} passed")
    raise SystemExit(1 if _failed else 0)
