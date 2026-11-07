#!/usr/bin/env python3
"""MyGRAPH phase registration must use the canonical EffING module identity.

The host loads ``cat/core_plugins/`` in preference to ``cat/plugins/``, so the
canonical ``PhaseSpec`` lives at
``cat.core_plugins.cat_efficient_ingestion.phases``. A consumer that imports the
same file through ``cat.plugins.cat_efficient_ingestion.phases`` creates a
SECOND module identity: ``isinstance(spec, PhaseSpec)`` is then False and
EffING's ``merged_phases`` silently drops the specs. These tests load the real
MyGRAPH ``main.py`` (its heavy relative modules are stubbed) and pin the
canonical identity plus the exactly-two-phase contract.

The EffING phase contract is provided by the shared ``effing_testkit`` (the
real installation when the host provides it, a synthetic one in a bare
checkout) instead of a private stub.

Standalone script (plugin rules): the host imports every ``.py`` under the
plugin folder, so the ENTIRE body lives inside ``if __name__ == "__main__":``
and importing this file executes zero code. Run it as a plain script from the
host checkout:

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-graphrag.my/tests/test_phase_registration.py
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

    PLUGIN_DIR = Path("/root/git/plugins/cat-graphrag.my")
    MAIN_PATH = PLUGIN_DIR / "main.py"
    CANONICAL = "cat.core_plugins.cat_efficient_ingestion.phases"
    LEGACY = "cat.plugins.cat_efficient_ingestion"
    MISSING = object()

    EXPECTED_IDS = ["graph_spacy_extraction", "graph_llm_extraction"]
    EXPECTED_DEPS = ("parsing_chunking", "embedding")

    def _load_effing_testkit():
        """Import the shared ``effing_testkit`` from its sibling plugin checkout."""
        candidates = (
            os.path.join(os.path.dirname(str(PLUGIN_DIR)), "cat-efficient-ingestion"),
            os.path.join(os.path.dirname(str(PLUGIN_DIR)), "cat-efficient-ingestion.my"),
        )
        for path in candidates:
            if os.path.isfile(os.path.join(path, "effing_testkit.py")):
                if path not in sys.path:
                    sys.path.insert(0, path)
                break
        import effing_testkit  # noqa: E402

        return effing_testkit

    def _cleanup(package_name: str) -> None:
        for name in list(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)

    def _load_plugin_main(package_name: str):
        """Load the real plugin ``main.py`` as ``package_name.main``.

        The plugin's relative imports (``.graphrag_handler`` / ``.entity_extractor``)
        pull the neo4j/spaCy stack, which the host test environment does not
        install; both are stubbed with the two names ``main.py`` binds at import
        time. The package is removed by the caller in ``finally``.
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

    def _test_hook_registers_exactly_two_canonical_specs():
        """The hook returns two specs in order, all of the canonical class."""
        from cat.core_plugins.cat_efficient_ingestion import phases

        module = _load_plugin_main("_mygraph_phase_present")
        try:
            specs = module.ingestion_phase_specs.function([], None)

            assert len(specs) == 2
            assert [spec.id for spec in specs] == EXPECTED_IDS
            for spec in specs:
                assert isinstance(spec, phases.PhaseSpec)
                assert type(spec).__module__ == CANONICAL
                assert type(spec).__module__.startswith("cat.core_plugins")
                assert spec.settings_category is None
                assert spec.depends_on == EXPECTED_DEPS

            assert LEGACY + ".phases" not in sys.modules
        finally:
            _cleanup("_mygraph_phase_present")

    async def _test_merged_phases_keeps_both_graph_phases():
        """EffING's merged_phases accepts both specs (no silent isinstance drop)."""
        from cat.core_plugins.cat_efficient_ingestion import phases

        module = _load_plugin_main("_mygraph_phase_merged")
        try:
            class _PluginManager:
                # The host registry maps a hook name to the LIST of registered
                # CatHooks; a bare CatHook is not iterable.
                hooks = {"ingestion_phase_specs": [module.ingestion_phase_specs]}

                async def execute_hook(self, name, *args, caller=None):
                    assert name == "ingestion_phase_specs"
                    return module.ingestion_phase_specs.function(*args, cat=caller)

            cat = types.SimpleNamespace(plugin_manager=_PluginManager())
            merged = await phases.merged_phases(cat, cat)

            assert len(merged) == len(phases.PHASES) + 2
            assert [phase for phase in merged if phase in EXPECTED_IDS] == EXPECTED_IDS
            for phase_id in EXPECTED_IDS:
                assert isinstance(merged[phase_id], phases.PhaseSpec)
        finally:
            _cleanup("_mygraph_phase_merged")

    def _test_module_imports_and_returns_specs_unchanged_when_effing_blocked():
        """Without EffING the plugin imports and the hook is an accumulator no-op."""
        saved = sys.modules.get(CANONICAL, MISSING)
        sys.modules[CANONICAL] = None  # type: ignore[assignment]
        try:
            module = _load_plugin_main("_mygraph_phase_absent")
            try:
                incoming = ["sentinel"]
                assert module.ingestion_phase_specs.function(incoming, None) is incoming
            finally:
                _cleanup("_mygraph_phase_absent")
        finally:
            if saved is MISSING:
                sys.modules.pop(CANONICAL, None)
            else:
                sys.modules[CANONICAL] = saved  # type: ignore[assignment]

    _tests = [
        ("test_hook_registers_exactly_two_canonical_specs",
         _test_hook_registers_exactly_two_canonical_specs),
        ("test_merged_phases_keeps_both_graph_phases",
         _test_merged_phases_keeps_both_graph_phases),
        ("test_module_imports_and_returns_specs_unchanged_when_effing_blocked",
         _test_module_imports_and_returns_specs_unchanged_when_effing_blocked),
    ]

    _testkit = _load_effing_testkit()
    _testkit.install_stub_package()
    _failed = 0
    try:
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
    finally:
        _testkit.uninstall_stub_package()

    print(f"\ntest_phase_registration: {len(_tests) - _failed}/{len(_tests)} passed")
    raise SystemExit(1 if _failed else 0)
