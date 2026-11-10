#!/usr/bin/env python3
"""MyGRAPH registers its graph phases on the CORE ingestion protocol.

The two graph phases are ``cat.rabbit_hole.phases.IngestionPhase`` subclasses
returned as CLASSES by the single ``rabbithole_ingestion_phases`` hook (the
core instantiates them without arguments for every run). There is no EffING
module identity to pin any more and no legacy phase-spec hook set.

These tests load the real MyGRAPH ``main.py`` (its heavy relative modules are
stubbed) and pin the exactly-two-phase contract, the accumulator behaviour and
import safety.

Standalone script (plugin rules): the host imports every ``.py`` under the
plugin folder, so the ENTIRE body lives inside ``if __name__ == "__main__":``
and importing this file executes zero code. Run it as a plain script from the
host checkout:

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-graphrag.my/tests/test_phase_registration.py
"""

if __name__ == "__main__":
    import importlib.util
    import os
    import sys
    import traceback
    import types
    from pathlib import Path

    PLUGIN_DIR = Path("/root/git/plugins/cat-graphrag.my")
    MAIN_PATH = PLUGIN_DIR / "main.py"

    EXPECTED_IDS = ["graph_spacy_extraction", "graph_llm_extraction"]
    EXPECTED_DEPS = ("parsing_chunking", "embedding")

    os.environ.setdefault("CAT_CRYPTO_KEY", "graphrag-test-crypto-key")
    os.environ.setdefault("CAT_CRYPTO_SALT", "graphrag-test-crypto-salt")

    from cat.rabbit_hole.phases import IngestionPhase

    def _cleanup(package_name: str) -> None:
        for name in list(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)

    def _load_plugin_main(package_name: str):
        """Load the real plugin ``main.py`` as ``package_name.main``.

        The plugin's heavy relative modules (``.graphrag_handler`` /
        ``.entity_extractor``) pull the neo4j/spaCy stack, which the host test
        environment does not install; both are stubbed with the names ``main.py``
        binds at import time. The package is removed by the caller in
        ``finally``.
        """
        assert MAIN_PATH.is_file(), f"MyGRAPH plugin not found at {PLUGIN_DIR}"

        package = types.ModuleType(package_name)
        package.__path__ = [str(PLUGIN_DIR)]
        package.__package__ = package_name
        sys.modules[package_name] = package

        handler = types.ModuleType(package_name + ".graphrag_handler")
        setattr(handler, "GraphRAGHandler", type("GraphRAGHandler", (), {}))
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
        return module

    def _test_hook_registers_exactly_two_core_phases():
        """The hook returns two classes in order, all core IngestionPhases."""
        module = _load_plugin_main("_mygraph_phase_present")
        try:
            classes = module.rabbithole_ingestion_phases.function([], None)

            assert len(classes) == 2
            assert [cls.id for cls in classes] == EXPECTED_IDS
            for cls in classes:
                assert issubclass(cls, IngestionPhase), cls
                assert cls.depends_on == EXPECTED_DEPS, cls.depends_on
                cls()  # no-argument construction, as the host does it
        finally:
            _cleanup("_mygraph_phase_present")

    def _test_hook_preserves_prior_accumulator_entries():
        """The hook APPENDS: phases of other plugins stay in front."""
        module = _load_plugin_main("_mygraph_phase_merged")
        try:
            incoming = ["sentinel"]
            classes = module.rabbithole_ingestion_phases.function(incoming, None)

            assert classes[0] == "sentinel"
            assert len(classes) == 3
            assert [cls.id for cls in classes[1:]] == EXPECTED_IDS
        finally:
            _cleanup("_mygraph_phase_merged")

    def _test_module_imports_and_returns_phases_unchanged_when_missing():
        """Without the plugin phase module the hook is an import-safe no-op."""
        module = _load_plugin_main("_mygraph_phase_absent")
        try:
            saved = sys.modules.get("_mygraph_phase_absent.phases")
            sys.modules["_mygraph_phase_absent.phases"] = None
            try:
                incoming = ["sentinel"]
                assert (
                    module.rabbithole_ingestion_phases.function(incoming, None)
                    is incoming
                )
            finally:
                if saved is not None:
                    sys.modules["_mygraph_phase_absent.phases"] = saved
                else:
                    sys.modules.pop("_mygraph_phase_absent.phases", None)
        finally:
            _cleanup("_mygraph_phase_absent")

    _tests = [
        ("test_hook_registers_exactly_two_core_phases",
         _test_hook_registers_exactly_two_core_phases),
        ("test_hook_preserves_prior_accumulator_entries",
         _test_hook_preserves_prior_accumulator_entries),
        ("test_module_imports_and_returns_phases_unchanged_when_missing",
         _test_module_imports_and_returns_phases_unchanged_when_missing),
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

    print(f"\ntest_phase_registration: {len(_tests) - _failed}/{len(_tests)} passed")
    raise SystemExit(1 if _failed else 0)

