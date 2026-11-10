#!/usr/bin/env python3
"""Standalone, import-safe test: ``--reingest`` re-runs every phase per source.

Re-ingesting must NOT reuse the chunks/diary of a previous run: the core
``RabbitHole.reingest_source`` resets the source diary and runs every phase
again from the bytes kept for it (each phase replaces what it produced
before). The maintenance CLI therefore delegates one ``reingest_source`` call
per enumerated source and never deletes points or state itself.

This test asserts the delegation with a fake engine (no real store, no Redis).

Run from the host repository (the plugin is importable by file path):

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-graphrag.my/tests/test_reingest_clears_diary.py

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
        path = Path(__file__).resolve().parent.parent / "maintenance_agent.py"
        spec = importlib.util.spec_from_file_location("maintenance_agent_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _source(name):
        return types.SimpleNamespace(name=name, metadata={}, path=name)

    class FakeRabbitHole:
        def __init__(self):
            self.reingest_calls = []

        async def reingest_source(self, ccat, name, chat_id=None):
            self.reingest_calls.append(name)
            return {"source": name}

    class FakeCcat:
        agent_key = "agent1"
        _id = "agent1"
        chunker = object()

        def __init__(self):
            self.rabbit_hole = FakeRabbitHole()

        async def get_stored_sources_with_metadata(self):
            from cat.services.memory.models import VectorMemoryType

            return {
                VectorMemoryType("declarative"): [
                    _source("file.txt"),
                    _source("https://example.test/paper.pdf"),
                ]
            }

    async def test_reingest_delegates_one_call_per_source():
        from cat.services.memory.models import VectorMemoryType  # noqa: F401

        maintenance_agent = _load_maintenance_agent()
        ccat = FakeCcat()

        async def fake_preflight(_ccat):
            return None

        async def fake_marker(_agent_id):
            return None

        async def fake_outcome(_agent_id, _op, _sources):
            return True

        with patch.object(maintenance_agent, "_preflight_reembed_path", fake_preflight), \
             patch.object(maintenance_agent, "_set_maintenance_marker", fake_marker), \
             patch.object(maintenance_agent, "_report_ingestion_outcome", fake_outcome):
            ok = await maintenance_agent._op_reingest(ccat, None, "declarative")

        assert ok is True, "reingest should succeed with a usable path"
        assert ccat.rabbit_hole.reingest_calls == [
            "file.txt",
            "https://example.test/paper.pdf",
        ], ccat.rabbit_hole.reingest_calls
        print("PASS --reingest re-runs every phase per source (core engine)")

    async def _main():
        await test_reingest_delegates_one_call_per_source()

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
    print("OK reingest clears the diary (delegated to the core engine)")
