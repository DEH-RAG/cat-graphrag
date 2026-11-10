#!/usr/bin/env python3
"""Standalone, import-safe harness for ``--resume-failed``.

The operation resumes the sources whose state status is ``error`` by calling
``ccat.rabbit_hole.rerun_source(ccat, name)``: the core engine keeps the
phases whose diary entry is still fresh (``complete``) and re-runs only the
failed ones. No purge, no point wipe, no re-parse and no re-embed of a healthy
phase. No manual diary rewrite happens in the CLI.

Covered:
  (a) dry-run resolves the plan and exits 2 with ZERO writes;
  (b) --yes calls rerun_source once per error source and reports ok;
  (c) a source with no failed phase is ``nothing-to-resume`` and untouched;
  (d) per-source failure isolation: one failing source never aborts the
      others and the process exits 1 naming the failing source;
  (e) a source whose LLM phase is disabled ends ``skipped``, never ``error``;
  (f) ``--all`` vs ``--agent`` scoping (protect system/``__*__`` unless
      explicitly named) and repeatable ``--agent``.

Run it as a plain script from the host repository:

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-graphrag.my/tests/test_maintenance_resume_cli.py

IMPORT-SAFETY: this file lives in the plugin folder, which the Cat plugin
loader imports recursively at activation. The ENTIRE body lives under
``if __name__ == "__main__":`` so importing the file executes zero code.
"""

if __name__ == "__main__":
    import asyncio
    import contextlib
    import copy
    import importlib.util
    import io
    import sys
    import types
    from pathlib import Path
    from unittest.mock import patch

    def _load_maintenance_agent():
        path = Path(__file__).resolve().parent.parent / "maintenance_agent.py"
        spec = importlib.util.spec_from_file_location(
            "maintenance_agent_resume_under_test", path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _complete(tag, started_at):
        return {
            "state": "complete",
            "tag": tag,
            "started_at": started_at,
            "settings_version": None,
            "deps": {},
        }

    def _error(reason, tag="err-tag", started_at=900):
        return {"state": "error", "reason": reason, "tag": tag, "started_at": started_at}

    def _doc(name, status, diary, error=None):
        return {
            "source": name,
            "chat_id": None,
            "type": "file",
            "status": status,
            "error": error,
            "completed_phases": dict(diary),
        }

    class Store:
        """In-memory ingestion states + the engine calls the op performs."""

        def __init__(self, docs):
            self.docs = {name: copy.deepcopy(doc) for name, doc in docs.items()}
            self.rerun_calls = []

        def plan(self, agent_id):
            resolved = []
            for name, doc in self.docs.items():
                if doc.get("status") != "error":
                    continue
                diary = doc.get("completed_phases") or {}
                failed = [
                    phase
                    for phase, entry in diary.items()
                    if (entry or {}).get("state") == "error"
                ]
                kept = [
                    phase
                    for phase, entry in diary.items()
                    if (entry or {}).get("state") != "error"
                ]
                resolved.append(
                    {
                        "source": name,
                        "scope": "agent",
                        "type": doc.get("type") or "file",
                        "failed_phases": failed,
                        "kept_phases": kept,
                    }
                )
            return resolved

        async def get_status(self, agent_id, source):
            doc = self.docs.get(source)
            return copy.deepcopy(doc) if doc is not None else None

    def finish_completed(store, name):
        """Engine semantics: drop the failed diary, complete the phase."""
        doc = store.docs[name]
        diary = doc["completed_phases"]
        for phase in [p for p, e in diary.items() if (e or {}).get("state") == "error"]:
            diary.pop(phase, None)
            diary[phase] = _complete("rerun", 1000)
        doc["status"] = "completed"
        doc["error"] = None

    def finish_skipped(store, name):
        doc = store.docs[name]
        doc["completed_phases"]["graph_llm_extraction"] = {
            "state": "skipped",
            "reason": "concept-relations extraction disabled",
        }
        doc["status"] = "completed"
        doc["error"] = None

    def explode(store, name):
        raise RuntimeError("reembed backend unavailable")

    class FakeRabbitHole:
        def __init__(self, store, behaviors):
            self.store = store
            self.behaviors = behaviors

        async def rerun_source(self, ccat, name, chat_id=None, blocking=True, invalidate=None):
            self.store.rerun_calls.append((name, None if invalidate is None else list(invalidate)))
            behavior = self.behaviors.get(name, finish_completed)
            behavior(self.store, name)
            return {"source": name}

    def _ccat(store, behaviors):
        return types.SimpleNamespace(
            agent_key="a1",
            _id="a1",
            rabbit_hole=FakeRabbitHole(store, behaviors),
        )

    def _patch_state(maintenance_agent, store):
        async def fake_resolve(agent_id):
            return store.plan(agent_id)

        async def fake_read(agent_id, source_name):
            return await store.get_status(agent_id, source_name)

        return [
            patch.object(maintenance_agent, "_resolve_resume_failed_plan", fake_resolve),
            patch.object(maintenance_agent, "_read_ingestion_status", fake_read),
        ]

    def _enter(patches):
        for p in patches:
            p.start()
        return patches

    def _exit(patches):
        for p in reversed(patches):
            p.stop()

    async def test_dry_run_exits_two_with_zero_writes(maintenance_agent):
        store = Store(
            {
                "broken.pdf": _doc(
                    "broken.pdf",
                    "error",
                    {
                        "parsing_chunking": _complete("p", 1),
                        "embedding": _complete("e", 2),
                        "graph_llm_extraction": _error("boom"),
                    },
                    error="boom",
                )
            }
        )

        async def fake_list_agents(include_protected=False):
            return ["a1"]

        async def forbidden_bootstrap(agent_id):
            raise AssertionError("dry-run must not bootstrap the agent")

        patches = _patch_state(maintenance_agent, store)
        patches += [
            patch.object(maintenance_agent, "_runtime_guard", lambda: True),
            patch.object(maintenance_agent, "_list_agents", fake_list_agents),
            patch.object(maintenance_agent, "_bootstrap_agent", forbidden_bootstrap),
        ]
        _enter(patches)
        stdout = io.StringIO()
        argv = ["maintenance_agent", "--agent", "a1", "--resume-failed", "--dry-run"]
        try:
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout):
                try:
                    await maintenance_agent.main()
                except SystemExit as exc:
                    code = exc.code
                else:
                    code = None
        finally:
            _exit(patches)

        out = stdout.getvalue()
        assert code == 2, f"dry-run must exit 2, got {code}: {out}"
        assert store.rerun_calls == [], store.rerun_calls
        assert "failed_phases=graph_llm_extraction" in out, out
        assert "kept_phases=2" in out, out
        print("PASS (a) dry-run resolves the plan, exits 2, zero writes")

    async def test_yes_resumes_only_error_sources(maintenance_agent):
        diary = {
            "parsing_chunking": _complete("p-tag", 111),
            "embedding": _complete("e-tag", 222),
            "graph_llm_extraction": _error("llm exploded", "l-tag", 333),
        }
        store = Store(
            {
                "doc.pdf": _doc("doc.pdf", "error", diary, error="llm exploded"),
            }
        )
        kept_before = {
            phase: dict(store.docs["doc.pdf"]["completed_phases"][phase])
            for phase in ("parsing_chunking", "embedding")
        }

        patches = _patch_state(maintenance_agent, store)
        _enter(patches)
        try:
            ok = await maintenance_agent._op_resume_failed(
                _ccat(store, {"doc.pdf": finish_completed}), None, "declarative"
            )
        finally:
            _exit(patches)

        assert ok is True, ok
        assert store.rerun_calls == [("doc.pdf", None)], store.rerun_calls
        after = store.docs["doc.pdf"]["completed_phases"]
        for phase, expected in kept_before.items():
            assert after[phase] == expected, (phase, after[phase], expected)
        assert store.docs["doc.pdf"]["status"] == "completed"
        print("PASS (b) --yes calls rerun_source once per error source")

    async def test_nothing_to_resume_is_untouched(maintenance_agent):
        store = Store(
            {
                "ok.pdf": _doc(
                    "ok.pdf",
                    "error",
                    {
                        "parsing_chunking": _complete("p", 1),
                        "embedding": _complete("e", 2),
                    },
                    error="listed as error but no phase failed",
                )
            }
        )
        patches = _patch_state(maintenance_agent, store)
        _enter(patches)
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                ok = await maintenance_agent._op_resume_failed(
                    _ccat(store, {}), None, "declarative"
                )
        finally:
            _exit(patches)

        out = stdout.getvalue()
        assert ok is True, out
        assert store.rerun_calls == [], store.rerun_calls
        assert "detail=nothing-to-resume" in out, out
        assert store.docs["ok.pdf"]["status"] == "error", store.docs["ok.pdf"]
        print("PASS (c) no failed phase -> nothing-to-resume, untouched")

    async def test_per_source_failure_isolation(maintenance_agent):
        store = Store(
            {
                "good.pdf": _doc(
                    "good.pdf",
                    "error",
                    {
                        "parsing_chunking": _complete("p", 1),
                        "graph_llm_extraction": _error("boom"),
                    },
                ),
                "bad.pdf": _doc(
                    "bad.pdf",
                    "error",
                    {
                        "parsing_chunking": _complete("p", 1),
                        "embedding": _error("embed down"),
                    },
                ),
            }
        )

        patches = _patch_state(maintenance_agent, store)
        _enter(patches)
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                ok = await maintenance_agent._op_resume_failed(
                    _ccat(store, {"good.pdf": finish_completed, "bad.pdf": explode}),
                    None,
                    "declarative",
                )
        finally:
            _exit(patches)

        out = stdout.getvalue()
        assert ok is False, out
        assert "source=good.pdf result=ok" in out, out
        assert "source=bad.pdf result=fail" in out, out
        assert [call[0] for call in store.rerun_calls] == ["good.pdf", "bad.pdf"], (
            store.rerun_calls
        )
        print("PASS (d) one failing source is isolated and named")

    async def test_main_exits_one_on_partial_failure(maintenance_agent):
        store = Store(
            {
                "good.pdf": _doc(
                    "good.pdf", "error", {"graph_llm_extraction": _error("boom")}
                ),
                "bad.pdf": _doc(
                    "bad.pdf", "error", {"graph_llm_extraction": _error("boom")}
                ),
            }
        )

        async def fake_list_agents(include_protected=False):
            return ["a1"]

        async def fake_bootstrap(agent_id):
            return (_ccat(store, {"good.pdf": finish_completed, "bad.pdf": explode}),
                    None, None)

        async def fake_engine(ccat, op="reembed"):
            return "CoreRabbitHole"

        async def fake_marker(agent_id):
            return None

        patches = _patch_state(maintenance_agent, store)
        patches += [
            patch.object(maintenance_agent, "_runtime_guard", lambda: True),
            patch.object(maintenance_agent, "_list_agents", fake_list_agents),
            patch.object(maintenance_agent, "_bootstrap_agent", fake_bootstrap),
            patch.object(maintenance_agent, "_resolve_ingestion_engine", fake_engine),
            patch.object(maintenance_agent, "_set_maintenance_marker", fake_marker),
            patch.object(maintenance_agent, "_clear_maintenance_marker", fake_marker),
        ]
        _enter(patches)
        argv = ["maintenance_agent", "--agent", "a1", "--resume-failed", "--yes"]
        try:
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                try:
                    await maintenance_agent.main()
                except SystemExit as exc:
                    code = exc.code
                else:
                    code = None
        finally:
            _exit(patches)

        assert code == 1, f"partial failure must exit 1, got {code}"
        print("PASS (d) main exits 1 on a partial per-source failure")

    async def test_disabled_llm_phase_ends_skipped(maintenance_agent):
        store = Store(
            {
                "doc.pdf": _doc(
                    "doc.pdf",
                    "error",
                    {
                        "parsing_chunking": _complete("p", 1),
                        "embedding": _complete("e", 2),
                        "graph_llm_extraction": _error(
                            "concept-relations extraction disabled"
                        ),
                    },
                )
            }
        )

        patches = _patch_state(maintenance_agent, store)
        _enter(patches)
        try:
            ok = await maintenance_agent._op_resume_failed(
                _ccat(store, {"doc.pdf": finish_skipped}), None, "declarative"
            )
        finally:
            _exit(patches)

        assert ok is True, store.docs["doc.pdf"]
        entry = store.docs["doc.pdf"]["completed_phases"]["graph_llm_extraction"]
        assert entry["state"] == "skipped", entry
        assert store.docs["doc.pdf"]["status"] == "completed", store.docs["doc.pdf"]
        print("PASS (e) disabled LLM phase ends skipped, never error")

    async def test_all_vs_agent_scoping(maintenance_agent):
        async def fake_list_agents(include_protected=False):
            if include_protected:
                return ["a1", "a2", "system", "__hidden__"]
            return ["a1", "a2"]

        async def fake_resolve(agent_id):
            return []

        patches = [
            patch.object(maintenance_agent, "_list_agents", fake_list_agents),
            patch.object(maintenance_agent, "_resolve_resume_failed_plan", fake_resolve),
        ]
        saved_argv = sys.argv
        _enter(patches)
        try:
            all_args = types.SimpleNamespace(
                agent=None,
                all=True,
                ops=["resume-failed"],
                collection="declarative",
            )
            all_plan = await maintenance_agent._build_ops_plan(all_args)
            one = types.SimpleNamespace(
                agent=["a1"],
                all=False,
                ops=["resume-failed"],
                collection="declarative",
            )
            one_plan = await maintenance_agent._build_ops_plan(one)

            sys.argv = [
                "maintenance_agent", "--agent", "system",
                "--resume-failed", "--dry-run",
            ]
            explicit = maintenance_agent._parse_args()
            assert explicit.agent == ["system"], explicit.agent

            sys.argv = [
                "maintenance_agent",
                "--agent", "a1",
                "--agent", "a2",
                "--resume-failed", "--dry-run",
            ]
            multi = maintenance_agent._parse_args()
            assert multi.agent == ["a1", "a2"], multi.agent
        finally:
            sys.argv = saved_argv
            _exit(patches)

        assert [entry["agent_id"] for entry in all_plan] == ["a1", "a2"], all_plan
        assert [entry["agent_id"] for entry in one_plan] == ["a1"], one_plan
        print("PASS (f) --all skips protected agents, --agent is repeatable/explicit")

    async def _main():
        maintenance_agent = _load_maintenance_agent()
        await test_dry_run_exits_two_with_zero_writes(maintenance_agent)
        await test_yes_resumes_only_error_sources(maintenance_agent)
        await test_nothing_to_resume_is_untouched(maintenance_agent)
        await test_per_source_failure_isolation(maintenance_agent)
        await test_main_exits_one_on_partial_failure(maintenance_agent)
        await test_disabled_llm_phase_ends_skipped(maintenance_agent)
        await test_all_vs_agent_scoping(maintenance_agent)

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
    print("OK maintenance resume-failed CLI")
