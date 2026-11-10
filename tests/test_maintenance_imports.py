#!/usr/bin/env python3
"""MyGRAPH maintenance CLI on the core ingestion API.

``maintenance_agent.py`` lives in the GraphRAG plugin folder and is imported by
the Cat plugin loader, so it must stay import-safe: only stdlib imports at
module level, every ``cat`` import inside a function. The engine-level ops use
the core ``RabbitHole`` (``ccat.rabbit_hole``) and
``cat.db.cruds.ingestions``; there is no external ingestion plugin.

These tests load the real ``maintenance_agent.py`` from the GraphRAG plugin
checkout (the plugin root by default, overridable with
``MYGRAPH_MAINTENANCE_AGENT``) and assert:

1. the module imports and ``_op_reembed`` delegates to
   ``ccat.rabbit_hole.rerun_source(..., invalidate=["embedding"])``;
2. the source has no ``cat_efficient_ingestion`` reference;
3. the CLI ``--help`` works (import-safety);
4. the graph phases are core ``IngestionPhase`` subclasses.

Standalone script (plugin rules): the host imports every ``.py`` under the
plugin folder, so the ENTIRE body lives inside ``if __name__ == "__main__":``
and importing this file executes zero code. Run it as a plain script from the
host checkout:

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-graphrag.my/tests/test_maintenance_imports.py
"""

if __name__ == "__main__":
    import asyncio
    import importlib.util
    import os
    import subprocess
    import sys
    import tempfile
    import traceback
    from pathlib import Path
    from types import ModuleType, SimpleNamespace
    from unittest import mock

    def _maintenance_path() -> Path:
        override = os.environ.get("MYGRAPH_MAINTENANCE_AGENT")
        if override:
            return Path(override)
        return Path(__file__).resolve().parents[1] / "maintenance_agent.py"

    def _load_maintenance() -> ModuleType:
        path = _maintenance_path()
        assert path.is_file(), f"maintenance_agent.py not found at {path}"
        spec = importlib.util.spec_from_file_location("mygraph_maintenance_agent", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    class FakeRabbitHole:
        def __init__(self) -> None:
            self.rerun_calls = []
            self.reingest_calls = []

        async def rerun_source(self, ccat, name, chat_id=None, blocking=True, invalidate=None):
            self.rerun_calls.append((name, invalidate))
            return {"source": name}

        async def reingest_source(self, ccat, name, chat_id=None):
            self.reingest_calls.append(name)
            return {"source": name}

    class _FakeCCat:
        agent_key = "agent-test"

        def __init__(self) -> None:
            self.rabbit_hole = FakeRabbitHole()

        async def get_stored_sources_with_metadata(self):
            from cat.services.memory.models import VectorMemoryType

            return {
                VectorMemoryType("declarative"): [
                    SimpleNamespace(name="doc.pdf", metadata={}, path="doc.pdf")
                ]
            }

    def _test_op_reembed_uses_core_rabbit_hole():
        from cat.db.cruds import ingestions as crud_ingestions

        module = _load_maintenance()
        ccat = _FakeCCat()

        async def no_statuses(*args, **kwargs):
            return []

        async def no_outcome(*args, **kwargs):
            return True

        async def no_marker(*args, **kwargs):
            return None

        with mock.patch.object(crud_ingestions, "list_statuses", no_statuses), \
             mock.patch.object(module, "_report_ingestion_outcome", no_outcome), \
             mock.patch.object(module, "_set_maintenance_marker", no_marker):
            assert asyncio.run(module._op_reembed(ccat, None, "declarative")) is True

        # The op delegates to the core engine, forcing the embedding phase.
        assert ccat.rabbit_hole.rerun_calls == [("doc.pdf", ["embedding"])], (
            ccat.rabbit_hole.rerun_calls
        )
        source = _maintenance_path().read_text()
        assert "cat_efficient_ingestion" not in source
        assert "cat.core_plugins.efficient_ingestion" not in source
        assert "rerun_source" in source and "reingest_source" in source

    def _test_graph_phases_use_the_core_contract():
        """The graph phases are core IngestionPhases, registered by the hook."""
        from cat.rabbit_hole.phases import IngestionPhase

        plugin_dir = _maintenance_path().parent
        spec = importlib.util.spec_from_file_location(
            "mygraph_phases_core", plugin_dir / "phases.py"
        )
        assert spec is not None and spec.loader is not None
        phases = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(phases)

        for cls, phase_id in (
            (phases.GraphSpacyPhase, "graph_spacy_extraction"),
            (phases.GraphConceptPhase, "graph_llm_extraction"),
        ):
            assert issubclass(cls, IngestionPhase), cls
            instance = cls()
            assert instance.id == phase_id
            assert instance.depends_on == ("parsing_chunking", "embedding")

        main_src = (plugin_dir / "main.py").read_text()
        assert "rabbithole_ingestion_phases" in main_src
        assert "ingestion_phase_specs" not in main_src

    def _test_help_works():
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            env = dict(os.environ)
            result = subprocess.run(
                [sys.executable, str(_maintenance_path()), "--help"],
                capture_output=True,
                text=True,
                cwd=tmp_path,
                env=env,
                timeout=120,
            )
            assert result.returncode == 0, result.stderr
            assert "usage:" in result.stdout.lower()
            assert "Traceback" not in result.stderr

    _tests = [
        ("test_op_reembed_uses_core_rabbit_hole", _test_op_reembed_uses_core_rabbit_hole),
        ("test_graph_phases_use_the_core_contract", _test_graph_phases_use_the_core_contract),
        ("test_help_works", _test_help_works),
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

    print(f"\ntest_maintenance_imports: {len(_tests) - _failed}/{len(_tests)} passed")
    raise SystemExit(1 if _failed else 0)
