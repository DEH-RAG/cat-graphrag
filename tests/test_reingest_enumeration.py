#!/usr/bin/env python3
"""Standalone, import-safe test: ``--reingest`` enumerates stored FILES too.

``_op_reingest`` must not enumerate sources from the VECTOR STORE only: a
source whose points were already wiped by a failed re-ingest is invisible
there, so the recovery path could never recover it — exactly the live bug
where 1 of 9 stored files was re-ingested. The enum is the UNION of the
vector-store sources and ``ccat.file_manager.list_files(agent_id)``, deduped
by name, with derived image artifacts
(``<stem>_img_<index>_<digest>.<ext>``) skipped.

The core ``reingest_source`` reads the kept bytes itself, so the CLI never
pre-reads a file; this test asserts that too.

Run from the host repository (the plugin is importable by file path):

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-graphrag.my/tests/test_reingest_enumeration.py

IMPORT-SAFETY: this file lives in the plugin folder, which the Cat plugin
loader imports recursively at activation. The ENTIRE body lives under
``if __name__ == "__main__":`` so importing the file executes zero code.
"""

if __name__ == "__main__":
    import asyncio
    import contextlib
    import importlib.util
    import io
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

    def _file(name):
        return types.SimpleNamespace(
            name=name, path=name, hash="h", size=1, last_modified=""
        )

    class FakeFileManager:
        def __init__(self, names):
            self.names = names
            self.listed_dirs = []
            self.read = []

        def list_files(self, remote_root_dir):
            self.listed_dirs.append(remote_root_dir)
            return [_file(name) for name in self.names]

        def read_file(self, remote_filename, remote_root_dir=None):
            self.read.append((remote_filename, remote_root_dir))
            return b"content of " + remote_filename.encode()

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

        def __init__(self, vector_sources, file_manager):
            self.rabbit_hole = FakeRabbitHole()
            self.file_manager = file_manager
            self._vector_sources = vector_sources

        async def get_stored_sources_with_metadata(self):
            from cat.services.memory.models import VectorMemoryType

            return {
                VectorMemoryType("declarative"): [
                    _source(name) for name in self._vector_sources
                ]
            }

    async def _run_reingest(ccat):
        maintenance_agent = _load_maintenance_agent()

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
        return ok, ccat.rabbit_hole.reingest_calls

    async def test_union_dedupes_and_skips_images():
        # Vector store holds one source (a ghost with no recoverable file) and
        # the file manager holds the three real files of the live bug, one of
        # which is a derived image artifact of a parent PDF.
        file_manager = FakeFileManager([
            "consigli.pdf",
            "criteri_correzioni.pdf",
            "consigli_img_0_deadbeef.jpg",
        ])
        ccat = FakeCcat(["ghost_source.pdf"], file_manager)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            ok, reingested = await _run_reingest(ccat)

        assert ok is True, "reingest should succeed"
        assert file_manager.listed_dirs == ["agent1"], file_manager.listed_dirs
        assert sorted(reingested) == [
            "consigli.pdf",
            "criteri_correzioni.pdf",
            "ghost_source.pdf",
        ], reingested
        assert (
            "detail=derived-image-skip source=consigli_img_0_deadbeef.jpg"
            in stdout.getvalue()
        ), stdout.getvalue()
        # The core engine reads the bytes; the CLI must not pre-read them.
        assert file_manager.read == [], file_manager.read
        print("PASS reingest unions stored files, skips derived images")

    async def test_union_dedupes_by_name():
        # The vector store already knows a.pdf: it must not be listed twice.
        file_manager = FakeFileManager(["a.pdf", "b.pdf", "c.pdf"])
        ccat = FakeCcat(["a.pdf"], file_manager)

        with contextlib.redirect_stdout(io.StringIO()):
            ok, reingested = await _run_reingest(ccat)

        assert ok is True
        assert reingested == ["a.pdf", "b.pdf", "c.pdf"], reingested
        assert file_manager.read == [], file_manager.read
        print("PASS reingest dedupes the union by name")

    async def _main():
        await test_union_dedupes_and_skips_images()
        await test_union_dedupes_by_name()

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
    print("OK reingest enumeration")
