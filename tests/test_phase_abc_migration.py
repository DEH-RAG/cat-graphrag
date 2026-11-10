"""Standalone, import-safe tests for the GraphRAG phases on the CORE contract.

Covers the migration of the two graph phases to
``cat.rabbit_hole.phases.IngestionPhase``:

  (a) a pure-vector agent (no ``GraphRAGHandler``) makes BOTH phases
      inapplicable, so the core records them as ``skipped``;
  (b) ``delete()`` removes only the given source's artefacts and is
      idempotent (called twice, second call is a no-op);
  (c) ``settings()`` is the existing configuration fingerprint: stable while
      the config is unchanged, different when the spaCy / concept config
      changes, and it never calls the LLM;
  (d) ``process()`` DELEGATES to the existing
      ``run_spacy_extraction_phase`` / ``run_llm_extraction_phase`` path
      instead of re-implementing extraction;
  (e) the module owns no diary/state/lock/retry protocol logic (the tenant
      write lock lives in the handler's write methods);
  (f) both phases are real core ``IngestionPhase`` subclasses with the
      expected ids/dependencies and no-argument construction;
  (g) the concept phase declares its own 300 s LLM not-ready budget;
  (h) a ``done`` result carrying warnings is complete, never retried.

The Cat plugin loader imports every ``.py`` under a plugin folder at
activation, so the whole file is inert on import: every import, class and test
lives inside ``if __name__ == "__main__":``. Run it as a plain script.

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-graphrag.my/tests/test_phase_abc_migration.py
"""

if __name__ == "__main__":
    import asyncio
    import inspect
    import os
    import sys
    import traceback
    import types
    from unittest.mock import AsyncMock

    PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Importing the real core phase contract builds Cat's crypto singleton.
    os.environ.setdefault("CAT_CRYPTO_KEY", "graphrag-test-crypto-key")
    os.environ.setdefault("CAT_CRYPTO_SALT", "graphrag-test-crypto-salt")

    def _load_phases():
        """Load the plugin's ``phases`` module behind a synthetic package.

        The real ``graphrag_handler`` is replaced by a stub exposing only
        ``GraphRAGHandler`` (the class ``isinstance`` is checked against).
        """
        package_name = "catgraphrag_phase_contract"
        package = types.ModuleType(package_name)
        package.__path__ = [PLUGIN_DIR]
        package.__package__ = package_name
        sys.modules[package_name] = package

        class GraphRAGHandler:
            pass

        handler_mod = types.ModuleType(package_name + ".graphrag_handler")
        handler_mod.GraphRAGHandler = GraphRAGHandler
        sys.modules[package_name + ".graphrag_handler"] = handler_mod

        from catgraphrag_phase_contract import phases

        return phases, GraphRAGHandler

    phases, _GraphRAGHandler = _load_phases()

    class FakeGraphHandler(_GraphRAGHandler):
        """A GraphRAG handler stand-in: only the phase-facing surface."""

        def __init__(self, spacy_marker="spacy-v1", concept_marker="concept-v1"):
            self.spacy_marker = spacy_marker
            self.concept_marker = concept_marker
            self.artifacts = {"file.pdf": {"entity:A"}, "other.pdf": {"entity:B"}}
            self.llm_calls = 0
            self.run_spacy_extraction_phase = AsyncMock(
                return_value={"status": "done"}
            )
            self.run_llm_extraction_phase = AsyncMock(return_value={"status": "done"})
            self.purge_source_graph_artifacts = AsyncMock(side_effect=self._purge)
            self._extract_and_link_entities = AsyncMock()
            self._extract_concept_relations = AsyncMock()

        def _purge(self, source):
            self.artifacts.pop(source, None)
            return 0

        def _spacy_fingerprint(self):
            return self.spacy_marker

        async def _concept_fingerprint(self):
            return self.concept_marker

        async def _resolve_agent_llm(self):  # must never run inside settings()
            self.llm_calls += 1
            raise AssertionError("settings() must never resolve or call the LLM")

    class FakeVectorHandler:
        """A pure-vector handler: not GraphRAG, so the graph is not active."""

    class FakeCat:
        def __init__(self, handler):
            self.vector_memory_handler = handler
            self.agent_key = "agent_test"

        async def embedder(self):
            return None

    class Source:
        """The core ``IngestionSource`` surface the phases read."""

        def __init__(self, name="file.pdf"):
            self.name = name
            self.agent_id = "agent_test"
            self.chat_id = None
            self.type = "file"

    def _source(name="file.pdf"):
        return Source(name)

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    async def test_a_vector_only_agent_is_skipped():
        cat = FakeCat(FakeVectorHandler())
        for phase in (phases.GraphSpacyPhase(), phases.GraphConceptPhase()):
            applies, reason = await phase.applies_to(_source(), cat)
            assert applies is False, (phase.id, applies)
            assert "not active" in (reason or ""), (phase.id, reason)
        print("PASS (a) vector-only agent: both graph phases are skipped")

    async def test_b_delete_source_scoped_and_idempotent():
        handler = FakeGraphHandler()
        cat = FakeCat(handler)
        phase = phases.GraphSpacyPhase()

        await phase.delete(_source("file.pdf"), cat)
        assert handler.artifacts == {"other.pdf": {"entity:B"}}, handler.artifacts
        handler.purge_source_graph_artifacts.assert_awaited_once_with("file.pdf")

        # second call is a no-op (idempotent) and still only targets file.pdf
        await phase.delete(_source("file.pdf"), cat)
        assert handler.artifacts == {"other.pdf": {"entity:B"}}, handler.artifacts
        assert handler.purge_source_graph_artifacts.await_count == 2
        assert handler.purge_source_graph_artifacts.await_args_list[1].args == (
            "file.pdf",
        )

        # the other phase shares the same graph cleanup contract
        concept = phases.GraphConceptPhase()
        await concept.delete(_source("file.pdf"), cat)
        assert handler.artifacts == {"other.pdf": {"entity:B"}}, handler.artifacts
        print("PASS (b) delete() is source-scoped and idempotent")

    async def test_c_tag_follows_config_only():
        handler = FakeGraphHandler()
        cat = FakeCat(handler)
        spacy = phases.GraphSpacyPhase()
        concept = phases.GraphConceptPhase()

        spacy_first = await spacy.settings(cat)
        spacy_second = await spacy.settings(cat)
        assert spacy_first == spacy_second, (spacy_first, spacy_second)
        assert spacy_first == handler.spacy_marker, spacy_first

        concept_first = await concept.settings(cat)
        concept_second = await concept.settings(cat)
        assert concept_first == concept_second, (concept_first, concept_second)
        assert concept_first == handler.concept_marker

        handler.spacy_marker = "spacy-v2"
        assert await spacy.settings(cat) != spacy_first
        handler.concept_marker = "concept-v2"
        assert await concept.settings(cat) != concept_first
        assert handler.llm_calls == 0, "settings() must not call the LLM"
        print("PASS (c) settings() is stable per config and changes with the config")

    async def test_d_process_delegates_to_existing_path():
        handler = FakeGraphHandler()
        cat = FakeCat(handler)

        spacy = phases.GraphSpacyPhase()
        spacy_result = await spacy.process(_source("file.pdf"), cat)
        assert spacy_result.status == "done", spacy_result
        handler.run_spacy_extraction_phase.assert_awaited_once_with(
            "file.pdf", cat=cat
        )
        handler._extract_and_link_entities.assert_not_awaited()

        concept = phases.GraphConceptPhase()
        concept_result = await concept.process(_source("file.pdf"), cat)
        assert concept_result.status == "done", concept_result
        handler.run_llm_extraction_phase.assert_awaited_once_with(
            "file.pdf", cat=cat
        )
        handler._extract_concept_relations.assert_not_awaited()
        print("PASS (d) process() delegates to the existing ingestion path")

    async def test_e_phase_module_owns_no_protocol_logic():
        """The plugin must NOT re-implement the phase-ingestion protocol.

        Diary writes, staleness, retries and status derivation belong to the
        core; the plugin only declares + executes + cleans up. The tenant write
        lock is graph-specific and lives in the handler's write methods, never
        in the phase module.
        """
        source = inspect.getsource(phases)
        forbidden = (
            "record_phase_",
            "acquire_tenant_write_lock",
            "derive_status",
            "cat.core_plugins",
        )
        found = [name for name in forbidden if name in source]
        assert not found, f"phase module re-implements protocol logic: {found}"
        assert "cat.core_plugins" not in source, "phases must not import EffING"
        print("PASS (e) phase module contains no diary/state/lock/retry protocol")

    async def test_f_phase_for_accepts_spec_or_id():
        """Both phases ARE core ``IngestionPhase`` subclasses, constructible
        without arguments (the host instantiates them for every run)."""
        from cat.rabbit_hole.phases import IngestionPhase

        for cls, phase_id in (
            (phases.GraphSpacyPhase, "graph_spacy_extraction"),
            (phases.GraphConceptPhase, "graph_llm_extraction"),
        ):
            assert issubclass(cls, IngestionPhase), cls
            instance = cls()  # no arguments: the core contract
            assert instance.id == phase_id, instance.id
            assert instance.depends_on == ("parsing_chunking", "embedding"), (
                instance.depends_on
            )
        print("PASS (f) both phases are no-argument core IngestionPhase subclasses")

    async def test_g_concept_phase_declares_llm_budget():
        """The concept phase declares its own 300s LLM not-ready budget."""
        concept = phases.GraphConceptPhase()
        spacy = phases.GraphSpacyPhase()

        assert concept.not_ready_budget_seconds == 300.0, (
            concept.not_ready_budget_seconds
        )
        assert spacy.not_ready_budget_seconds is None, (
            spacy.not_ready_budget_seconds
        )
        print("PASS (g) concept phase declares the 300s LLM not-ready budget")

    async def test_h_warning_phase_is_complete_not_retried():
        """A ``done`` result carrying warnings is COMPLETE.

        Warnings are telemetry attached to ``done``; they never change the
        status. The core retries only ``not_ready`` (and blocks dependents on
        ``error``/``skipped``), so a warning phase is never re-run and never
        blocks a dependent.
        """
        handler = FakeGraphHandler()
        warning = {
            "code": "empty-extraction",
            "message": "no entities found in 42 chars of 1 chunks for 'file.pdf'",
        }
        handler.run_spacy_extraction_phase = AsyncMock(
            return_value={
                "status": "done",
                "warnings": [warning],
                "processed": 0,
            }
        )
        cat = FakeCat(handler)
        phase = phases.GraphSpacyPhase()

        result = await phase.process(_source("file.pdf"), cat)

        assert result.status == "done", result
        assert result.processed == 0, result
        assert result.warnings == [warning], result
        # The machine's retry rule: only not_ready is retried; error blocks
        # dependents; done (with or without warnings) is terminal-complete.
        assert result.status not in ("not_ready", "error"), (
            "a phase with warnings must never be retried or treated as failed"
        )
        print("PASS (h) a phase with warnings is complete and not retried")

    async def _main():
        await test_a_vector_only_agent_is_skipped()
        await test_b_delete_source_scoped_and_idempotent()
        await test_c_tag_follows_config_only()
        await test_d_process_delegates_to_existing_path()
        await test_e_phase_module_owns_no_protocol_logic()
        await test_f_phase_for_accepts_spec_or_id()
        await test_g_concept_phase_declares_llm_budget()
        await test_h_warning_phase_is_complete_not_retried()

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        traceback.print_exc()
        sys.exit(1)
    print("OK phase ABC migration (eight tests)")

