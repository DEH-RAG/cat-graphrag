#!/usr/bin/env python3
"""There is ONE graph writer: the core ingestion phases.

The old classic path (``after_rabbithole_stored_documents`` ->
``create_derived_graph_for_source``) and its engine-selection switch are gone.
Every source is ingested by the core RabbitHole, which runs
``graph_spacy_extraction`` and ``graph_llm_extraction`` after parsing/embedding.

These tests load the real MyGRAPH ``main.py``; its heavy relative modules
(``.graphrag_handler`` / ``.entity_extractor``) are replaced by local fakes, so
no Neo4j/spaCy is required.

Standalone script (plugin rules): the host imports every ``.py`` under the
plugin folder, so the ENTIRE body lives inside ``if __name__ == "__main__":``
and importing this file executes zero code. Run it as a plain script from the
host checkout:

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-graphrag.my/tests/test_classic_path.py
"""

if __name__ == "__main__":
    import asyncio
    import importlib.util
    import inspect
    import os
    import sys
    import traceback
    import types
    from pathlib import Path

    os.environ.setdefault("CAT_CRYPTO_KEY", "graphrag-test-crypto-key")
    os.environ.setdefault("CAT_CRYPTO_SALT", "graphrag-test-crypto-salt")

    PLUGIN_DIR = Path("/root/git/plugins/cat-graphrag.my")
    MAIN_PATH = PLUGIN_DIR / "main.py"

    class FakeGraphRAGHandler:
        """Counts graph writes so a double-write would be visible."""

        def __init__(self):
            self.classic_calls = 0
            self.spacy_calls = 0
            self.llm_calls = 0
            self.nodes = 0
            self.relationships = 0

        async def drain_source_tasks(self, source):
            return False

        async def create_derived_graph_for_source(self, source, stored_points, cat):
            self.classic_calls += 1
            self.nodes += 2
            self.relationships += 1

        async def run_spacy_extraction_phase(self, source, cat=None):
            self.spacy_calls += 1
            self.nodes += 1
            self.relationships += 1
            return {"status": "done"}

        async def run_llm_extraction_phase(self, source, cat=None):
            self.llm_calls += 1
            self.nodes += 1
            self.relationships += 1
            return {"status": "done"}

        def _spacy_fingerprint(self):
            return "spacy"

        async def _concept_fingerprint(self):
            return "concept"

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
            vector_memory_handler=handler, agent_key="agent_test", embedder=_embedder
        )

    async def _test_single_registration_hook():
        """The plugin contributes its two phases through ONE hook.

        The removed hooks/callbacks must not exist on the module.
        """
        module = _load_plugin_main("_mygraph_classic_single")
        try:
            classes = module.rabbithole_ingestion_phases.function([], None)
            assert [cls.id for cls in classes] == [
                "graph_spacy_extraction",
                "graph_llm_extraction",
            ], classes
            for retired in (
                "after_rabbithole_stored_documents",
                "rabbithole_ingestion_processing",
                "ingestion_phase_specs",
                "ingestion_phase_objects",
                "ingestion_phase_settings_marker",
                "ingestion_phase_run",
                "ingestion_phase_purge",
                "before_ingestion_status_completed",
                "_phase_path_active",
            ):
                assert not hasattr(module, retired), f"retired hook still present: {retired}"
        finally:
            _cleanup("_mygraph_classic_single")

    async def _test_phases_are_the_only_graph_writers():
        """Running both phases writes the graph once; the classic writer never runs."""
        module = _load_plugin_main("_mygraph_classic_writers")
        try:
            phases = __import__(
                "_mygraph_classic_writers.phases", fromlist=["phases"]
            )
            # Skip embedder/extractor preparation: this test counts writers.
            async def _no_prepare(_handler, _cat):
                return None

            phases._prepare_handler = _no_prepare

            handler = FakeGraphRAGHandler()
            cat = _cat(handler)
            source = _Source()

            await phases.GraphSpacyPhase().process(source, cat)
            await phases.GraphConceptPhase().process(source, cat)

            assert handler.classic_calls == 0
            assert handler.spacy_calls == 1
            assert handler.llm_calls == 1
            assert handler.nodes > 0
            assert handler.relationships > 0
        finally:
            _cleanup("_mygraph_classic_writers")

    def _test_no_engine_switch_remains():
        """No EffING import and no classic/efficient mode selection in main.py."""
        module = _load_plugin_main("_mygraph_classic_no_switch")
        try:
            src = inspect.getsource(module)
            for ref in (
                "import cat.core_plugins",
                "from cat.core_plugins",
                "cat_efficient_ingestion",
                "EfficientIngestionConfiguration",
                "_resolved_system_ingestion_config_name",
                "ingestion_phase_",
            ):
                assert ref not in src, f"retired ingestion path still present: {ref}"
        finally:
            _cleanup("_mygraph_classic_no_switch")

    _tests = [
        ("test_single_registration_hook", _test_single_registration_hook),
        ("test_phases_are_the_only_graph_writers",
         _test_phases_are_the_only_graph_writers),
        ("test_no_engine_switch_remains", _test_no_engine_switch_remains),
    ]
    _failed = 0
    for _name, _fn in _tests:
        try:
            if inspect.iscoroutinefunction(_fn):
                asyncio.run(_fn())
            else:
                _fn()
            print(f"PASS {_name}")
        except Exception as _exc:  # noqa: BLE001 - report and continue
            _failed += 1
            traceback.print_exc()
            print(f"FAIL {_name}: {_exc}")

    print(f"\ntest_classic_path: {len(_tests) - _failed}/{len(_tests)} passed")
    raise SystemExit(1 if _failed else 0)

