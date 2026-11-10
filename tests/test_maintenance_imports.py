#!/usr/bin/env python3
"""MyGRAPH maintenance CLI against the extracted EffING plugin.

EffING moved to ``cat.core_plugins.cat_efficient_ingestion`` (``reembed`` and
``registry`` modules). ``maintenance_agent.py`` lives in the GraphRAG plugin
folder and is imported by the Cat plugin loader, so it must stay import-safe:
its EffING imports are lazy and guarded.

These tests load the real ``maintenance_agent.py`` from the GraphRAG plugin
checkout (the plugin root by default, overridable with
``MYGRAPH_MAINTENANCE_AGENT``) and assert:

1. with EffING importable the module imports and the symbols it needs resolve
   through the new module paths (and are the ones actually used by an op);
2. with EffING blocked the module still imports, the CLI ``--help`` works, and
   an op needing EffING aborts with a clear message and exit code 2 instead of
   raising a traceback.

Standalone script (plugin rules): the host imports every ``.py`` under the
plugin folder, so the ENTIRE body lives inside ``if __name__ == "__main__":``
and importing this file executes zero code. Run it as a plain script from the
host checkout:

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-graphrag.my/tests/test_maintenance_imports.py
"""

if __name__ == "__main__":
    import asyncio
    import contextlib
    import importlib
    import importlib.abc
    import importlib.util
    import io
    import os
    import subprocess
    import sys
    import tempfile
    import textwrap
    import traceback
    from pathlib import Path
    from types import ModuleType
    from unittest import mock

    EFFING = "cat.core_plugins.cat_efficient_ingestion"

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

    class _BlockEffing(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == EFFING or fullname.startswith(EFFING + "."):
                raise ImportError(f"{fullname} blocked for test")
            return None

    class _FakeCCat:
        agent_key = "agent-test"

        async def get_stored_sources_with_metadata(self):
            from cat.services.memory.models import VectorMemoryType

            return {VectorMemoryType("declarative"): []}

    def _test_module_imports_and_uses_new_effing_paths():
        import cat.services.memory.models  # noqa: F401  (used by _op_reembed)

        calls: list[tuple[object, ...]] = []

        fake_reembed = ModuleType(f"{EFFING}.reembed")
        fake_registry = ModuleType(f"{EFFING}.registry")

        async def reembed_sources(ccat, memory_type, sources):
            calls.append((ccat, memory_type, sources))

        async def delete_status(*args, **kwargs):
            calls.append(("delete_status", args, kwargs))

        async def list_statuses(agent_id):
            calls.append(("list_statuses", agent_id))
            return []

        async def get_status(agent_id, scope, source):
            calls.append(("get_status", agent_id, scope, source))
            return {"status": "completed", "completed_phases": {}}

        def phase_state(entry):
            return (entry or {}).get("state")

        setattr(fake_reembed, "reembed_sources", reembed_sources)
        setattr(fake_registry, "delete_status", delete_status)
        setattr(fake_registry, "list_statuses", list_statuses)
        setattr(fake_registry, "get_status", get_status)
        setattr(fake_registry, "phase_state", phase_state)
        setattr(fake_registry, "PHASE_STATE_ERROR", "error")
        fake_modules = {
            f"{EFFING}.reembed": fake_reembed,
            f"{EFFING}.registry": fake_registry,
        }
        with mock.patch.dict(sys.modules, fake_modules):
            module = _load_maintenance()

            assert asyncio.run(module._op_reembed(_FakeCCat(), None, "declarative")) is True
            assert calls[0] == ("list_statuses", "agent-test")
            assert calls[-1][2] == []  # reembed_sources(ccat, VectorMemoryType, sources)

            source = _maintenance_path().read_text()
            assert "cat.core_plugins.efficient_ingestion" not in source
            assert "cat.core_plugins.ingestion_status" not in source
            assert EFFING in source

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

    def _test_missing_effing_aborts_without_traceback(op_name):
        module = _load_maintenance()  # import first: module itself stays import-safe

        saved = {
            name: sys.modules.pop(name)
            for name in list(sys.modules)
            if name == EFFING or name.startswith(EFFING + ".")
        }
        blocker = _BlockEffing()
        sys.meta_path.insert(0, blocker)
        stderr = io.StringIO()
        try:
            with contextlib.redirect_stderr(stderr):
                try:
                    if op_name == "_verify_agent":
                        asyncio.run(module._verify_agent(_FakeCCat(), None, "declarative"))
                    else:
                        asyncio.run(getattr(module, op_name)(_FakeCCat(), None, "declarative"))
                except SystemExit as exc:
                    assert exc.code == 2
                else:
                    raise AssertionError(f"{op_name} did not abort with SystemExit")
        finally:
            sys.meta_path.remove(blocker)
            sys.modules.update(saved)

        err = stderr.getvalue()
        assert "ABORTED" in err
        assert EFFING in err
        assert "Traceback" not in err

    def _test_help_works_with_effing_blocked():
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            sitecustomize = tmp_path / "sitecustomize.py"
            sitecustomize.write_text(
                textwrap.dedent(
                    f"""
                    import sys

                    class _BlockEffing:
                        def find_spec(self, fullname, path=None, target=None):
                            if fullname == {EFFING!r} or fullname.startswith({EFFING!r} + "."):
                                raise ImportError(f"{{fullname}} blocked for test")
                            return None

                    sys.meta_path.insert(0, _BlockEffing())
                    """
                )
            )

            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(
                [str(tmp_path), env.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)

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
        ("test_module_imports_and_uses_new_effing_paths",
         _test_module_imports_and_uses_new_effing_paths),
        ("test_graph_phases_use_the_core_contract",
         _test_graph_phases_use_the_core_contract),
        ("test_help_works_with_effing_blocked", _test_help_works_with_effing_blocked),
    ]
    _tests.extend(
        (f"test_missing_effing_aborts_without_traceback[{op}]",
         lambda op=op: _test_missing_effing_aborts_without_traceback(op))
        for op in ("_op_reembed", "_op_reingest", "_verify_agent")
    )

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
