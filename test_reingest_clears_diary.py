#!/usr/bin/env python3
"""Standalone, import-safe test: ``--reingest`` must clear the lying diary.

The completed-phases diary is bookkeeping, not proof the points exist. When
``_op_reingest`` wipes a source's points it MUST also clear that source's
``completed_phases`` diary (through ``cat.db.crud``) BEFORE re-running, so the
phases re-run deterministically even if the status row survives. This test
asserts the op sequencing with a fake crud/handler (no real store, no Redis).

Run from the host repository (the plugin is importable by file path):

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-graphrag.my/test_reingest_clears_diary.py

IMPORT-SAFETY: this file lives in the plugin folder, which the Cat plugin
loader imports recursively at activation. The ENTIRE body lives under
``if __name__ == "__main__":`` so importing the file executes zero code.
"""

if __name__ == "__main__":
    import asyncio
    import importlib.util
    import sys
    import types
    from pathlib import Path
    from unittest.mock import patch

    def _load_maintenance_agent():
        path = Path(__file__).with_name("maintenance_agent.py")
        spec = importlib.util.spec_from_file_location("maintenance_agent_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _source(name):
        return types.SimpleNamespace(name=name, metadata={}, path=name)

    async def test_reingest_clears_diary_for_wiped_source():
        import cat.core_plugins.cat_efficient_ingestion.reembed as effing_reembed
        import cat.core_plugins.cat_efficient_ingestion.registry as effing_registry
        from cat.services.memory.models import VectorMemoryType

        maintenance_agent = _load_maintenance_agent()
        calls = []

        class FakeHandler:
            async def delete_tenant_points(self, collection, metadata=None):
                calls.append(("delete_points", (metadata or {}).get("source")))

        class FakeCcat:
            agent_key = "agent1"
            _id = "agent1"
            vector_memory_handler = FakeHandler()

            async def get_stored_sources_with_metadata(self):
                return {
                    VectorMemoryType("declarative"): [
                        _source("file.txt"),
                        _source("https://example.test/paper.pdf"),
                    ]
                }

        async def fake_preflight(ccat):
            return None

        async def fake_marker(agent_id):
            return None

        async def fake_clear(agent_id, scope, source):
            calls.append(("clear_diary", source))
            return True

        async def fake_delete_status(agent_id, scope, source):
            calls.append(("delete_status", source))

        async def fake_reembed(ccat, collection, sources):
            calls.append(("reembed", [s.name for s in sources]))

        with patch.object(maintenance_agent, "_preflight_reembed_path", fake_preflight), \
             patch.object(maintenance_agent, "_set_maintenance_marker", fake_marker), \
             patch.object(effing_registry, "clear_completed_phases", fake_clear), \
             patch.object(effing_registry, "delete_status", fake_delete_status), \
             patch.object(effing_reembed, "reembed_sources", fake_reembed):
            ok = await maintenance_agent._op_reingest(
                FakeCcat(), None, "declarative"
            )

        assert ok is True, "reingest should succeed with a usable path"
        assert calls == [
            ("delete_points", "file.txt"),
            ("clear_diary", "file.txt"),
            ("delete_status", "file.txt"),
            ("reembed", ["file.txt", "https://example.test/paper.pdf"]),
        ], calls
        print("PASS --reingest wipes points, then clears the diary, then re-runs")

    async def _main():
        await test_reingest_clears_diary_for_wiped_source()

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
    print("OK reingest clears the diary")
