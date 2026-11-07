#!/usr/bin/env python3
"""MyGRAPH tenant cleanup must match the host's hook contract.

The host calls ``after_vector_memory_transfer_on_agent(success, cat)``
(``cat/core_plugins/base_plugin/hooks/flow.py``, dispatched from
``CheshireCat.transfer_vector_points_from``). The plugin used to declare a
third ``old_handler_name`` parameter, so ``_run_hooks`` raised ``TypeError``,
logged it and continued: the Neo4j tenant wipe NEVER ran.

These tests load the real MyGRAPH ``main.py`` with the heavy relative modules
(``.graphrag_handler`` / ``.entity_extractor``) stubbed, invoke the hook with
the host's exact argument shape, and assert the wipe policy:

* success + new handler NOT GraphRAG -> wipe the tenant (graphrag -> other);
* failure -> no wipe;
* new handler IS GraphRAG -> no wipe (the graph phases own the work).

An adjacent test pins the classic ingestion callback to a no-op when the
EffING phase path is active.

Standalone script (plugin rules): the host imports every ``.py`` under the
plugin folder, so the ENTIRE body lives inside ``if __name__ == "__main__":``
and importing this file executes zero code. It is a pure unit test: it stubs
the plugin's heavy modules and needs only the host venv for the one test that
resolves the ingestion factory. Run it as a plain script from the host
checkout:

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-graphrag.my/tests/test_transfer_hook.py
"""

if __name__ == "__main__":
    import asyncio
    import importlib
    import importlib.util
    import inspect
    import sys
    import traceback
    import types
    from pathlib import Path
    from unittest import mock

    PLUGIN_DIR = Path("/root/git/plugins/cat-graphrag.my")
    MAIN_PATH = PLUGIN_DIR / "main.py"
    EFFICIENT = "EfficientIngestionConfiguration"

    class FakeGraphRAGHandler:
        """Current handler stand-in; counts classic graph writes."""

        def __init__(self):
            self.classic_calls = 0

        async def create_derived_graph_for_source(self, source, stored_points, cat):
            self.classic_calls += 1

    def _wipe_spy():
        calls = []

        async def wipe_tenant_graph_data(agent_key):
            calls.append(agent_key)

        return calls, wipe_tenant_graph_data

    def _cleanup(package_name):
        for name in list(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)

    def _load_plugin_main(package_name, wipe_spy):
        package = types.ModuleType(package_name)
        package.__path__ = [str(PLUGIN_DIR)]
        package.__package__ = package_name
        sys.modules[package_name] = package

        handler = types.ModuleType(package_name + ".graphrag_handler")
        setattr(handler, "GraphRAGHandler", FakeGraphRAGHandler)
        setattr(handler, "Neo4jGraphRAGConfig", type("Neo4jGraphRAGConfig", (), {}))
        setattr(handler, "wipe_tenant_graph_data", wipe_spy)
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

    def _cat(handler):
        return types.SimpleNamespace(vector_memory_handler=handler, agent_key="agent_test")

    def _install_resolver(name):
        ingestion = importlib.import_module("cat.services.factory.ingestion")

        async def resolved_config_name(lizard):
            return name

        return mock.patch.object(
            ingestion, "resolved_config_name", resolved_config_name, create=True
        )

    def _test_hook_signature_matches_host_contract():
        """Exactly (success, cat) — a third parameter makes _run_hooks raise TypeError."""
        calls, wipe = _wipe_spy()
        module = _load_plugin_main("_mygraph_transfer_signature", wipe)
        try:
            params = list(
                inspect.signature(module.after_vector_memory_transfer_on_agent.function).parameters
            )
            assert params == ["success", "cat"]
        finally:
            _cleanup("_mygraph_transfer_signature")

    async def _test_host_shaped_success_non_graphrag_wipes_tenant():
        """Host calls (success, cat) with a non-GraphRAG new handler: the wipe runs
        with no TypeError (the exact bug: the old 3-arg signature never wiped)."""
        calls, wipe = _wipe_spy()
        module = _load_plugin_main("_mygraph_transfer_wipes", wipe)
        try:
            cat = _cat(None)

            await module.after_vector_memory_transfer_on_agent.function(True, cat)

            assert calls == ["agent_test"]
        finally:
            _cleanup("_mygraph_transfer_wipes")

    async def _test_no_wipe_on_failed_transfer():
        """A failed transfer must never wipe the old store, even if the new handler
        is not GraphRAG."""
        calls, wipe = _wipe_spy()
        module = _load_plugin_main("_mygraph_transfer_failure", wipe)
        try:
            cat = _cat(None)

            await module.after_vector_memory_transfer_on_agent.function(False, cat)

            assert calls == []
        finally:
            _cleanup("_mygraph_transfer_failure")

    async def _test_no_wipe_when_new_handler_is_graphrag():
        """A non-graphrag -> GraphRAG switch keeps the old graph: the new handler IS
        GraphRAG, so the revalidation sweep + graph phases own the work."""
        calls, wipe = _wipe_spy()
        module = _load_plugin_main("_mygraph_transfer_new_graphrag", wipe)
        try:
            cat = _cat(FakeGraphRAGHandler())

            await module.after_vector_memory_transfer_on_agent.function(True, cat)

            assert calls == []
        finally:
            _cleanup("_mygraph_transfer_new_graphrag")

    async def _test_classic_callback_noop_when_phase_path_active():
        """With EffING active (same gate as the transfer hook's sibling), the
        classic ``after_rabbithole_stored_documents`` does zero graph work."""
        calls, wipe = _wipe_spy()
        module = _load_plugin_main("_mygraph_transfer_classic_gate", wipe)
        try:
            with _install_resolver(EFFICIENT):
                handler = FakeGraphRAGHandler()
                cat = _cat(handler)

                await module.after_rabbithole_stored_documents.function("file.pdf", [], cat)

                assert handler.classic_calls == 0
        finally:
            _cleanup("_mygraph_transfer_classic_gate")

    _tests = [
        ("test_hook_signature_matches_host_contract", _test_hook_signature_matches_host_contract),
        ("test_host_shaped_success_non_graphrag_wipes_tenant",
         _test_host_shaped_success_non_graphrag_wipes_tenant),
        ("test_no_wipe_on_failed_transfer", _test_no_wipe_on_failed_transfer),
        ("test_no_wipe_when_new_handler_is_graphrag",
         _test_no_wipe_when_new_handler_is_graphrag),
        ("test_classic_callback_noop_when_phase_path_active",
         _test_classic_callback_noop_when_phase_path_active),
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

    print(f"\ntest_transfer_hook: {len(_tests) - _failed}/{len(_tests)} passed")
    raise SystemExit(1 if _failed else 0)
