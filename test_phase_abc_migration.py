"""Standalone, import-safe tests for the class-based GraphRAG phases.

Covers the phase-contract v2 migration of the two graph phases:

  (a) a pure-vector agent (no ``GraphRAGHandler``) makes BOTH phases
      inapplicable, so the dispatcher records them as ``skipped``;
  (b) ``delete()`` removes only the given source's artefacts and is
      idempotent (called twice, second call is a no-op);
  (c) ``tag()`` is the existing configuration fingerprint: stable while the
      config is unchanged, different when the spaCy / concept config changes,
      and it never calls the LLM;
  (d) ``process()`` DELEGATES to the existing
      ``run_spacy_extraction_phase`` / ``run_llm_extraction_phase`` path
      instead of re-implementing extraction.

The Cat plugin loader imports every ``.py`` under a plugin folder at
activation, so the whole file is inert on import: every import, class and test
lives inside ``if __name__ == "__main__":``. Run it as a plain script:

    python test_phase_abc_migration.py
"""

if __name__ == "__main__":
    import asyncio
    import sys
    import types
    from abc import ABC, abstractmethod
    from dataclasses import dataclass
    from unittest.mock import AsyncMock

    # ------------------------------------------------------------------
    # Import-safe stubs
    # ------------------------------------------------------------------

    def _install_cat_stub():
        cat_mod = types.ModuleType("cat")
        cat_mod.__path__ = []

        def _hook(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return lambda f: f

        cat_mod.hook = _hook
        sys.modules["cat"] = cat_mod

    def _install_effing_stub():
        """A faithful minimal EffING phase contract (ABC + resolver)."""

        for name in (
            "cat.core_plugins",
            "cat.core_plugins.cat_efficient_ingestion",
        ):
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod

        phases = types.ModuleType("cat.core_plugins.cat_efficient_ingestion.phases")

        @dataclass(frozen=True)
        class PhaseResult:
            status: str
            reason: str | None = None
            processed: int | None = None
            expected: int | None = None

        class IngestionPhase(ABC):
            id = ""
            depends_on = ()
            settings_category = None
            settings_check = None

            @property
            def name(self):
                return self.id

            @abstractmethod
            async def applies_to(self, cat, deps_state):
                ...

            @abstractmethod
            async def process(self, source, cat):
                ...

            @abstractmethod
            async def delete(self, source, cat):
                ...

            async def tag(self, settings=None):
                from cat.core_plugins.cat_efficient_ingestion.phase_tag_resolver import (
                    own_tag,
                )

                return own_tag(self.id, {} if settings is None else settings)

        phases.IngestionPhase = IngestionPhase
        phases.PhaseResult = PhaseResult
        sys.modules[phases.__name__] = phases

        import hashlib
        import json

        resolver = types.ModuleType(
            "cat.core_plugins.cat_efficient_ingestion.phase_tag_resolver"
        )

        def own_tag(kind, value):
            payload = json.dumps(
                value, sort_keys=True, separators=(",", ":"), default=str
            )
            return hashlib.sha256(f"{kind}:{payload}".encode("utf-8")).hexdigest()

        resolver.own_tag = own_tag
        sys.modules[resolver.__name__] = resolver

        return IngestionPhase, PhaseResult, own_tag

    class FakeGraphHandler:
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

        async def _resolve_agent_llm(self):  # must never run inside tag()
            self.llm_calls += 1
            raise AssertionError("tag() must never resolve or call the LLM")

    class FakeVectorHandler:
        """A pure-vector handler: not GraphRAG, so the graph is not active."""

    class FakeCat:
        def __init__(self, handler):
            self.vector_memory_handler = handler

    _install_cat_stub()
    _IngestionPhase, _PhaseResult, _own_tag = _install_effing_stub()

    _pkg = types.ModuleType("catgraphrag_phase")
    _pkg.__path__ = [__import__("os").path.dirname(__import__("os").path.abspath(__file__))]
    sys.modules["catgraphrag_phase"] = _pkg

    from catgraphrag_phase import phases

    # The tests exercise the contract with a fake handler; production resolves
    # the live ``GraphRAGHandler`` through ``isinstance`` (same guard).
    _ORIGINAL_RESOLVER = phases.graphrag_handler
    phases.graphrag_handler = lambda cat: (
        cat.vector_memory_handler
        if isinstance(cat.vector_memory_handler, FakeGraphHandler)
        else None
    )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    async def test_a_vector_only_agent_is_skipped():
        cat = FakeCat(FakeVectorHandler())
        for phase in (phases.GraphSpacyPhase(cat), phases.GraphConceptPhase(cat)):
            applies, reason = await phase.applies_to(cat, {})
            assert applies is False, (phase.id, applies)
            assert "not active" in (reason or ""), (phase.id, reason)
        print("PASS (a) vector-only agent: both graph phases are skipped")

    async def test_b_delete_source_scoped_and_idempotent():
        handler = FakeGraphHandler()
        cat = FakeCat(handler)
        phase = phases.GraphSpacyPhase(cat)

        await phase.delete("file.pdf", cat)
        assert handler.artifacts == {"other.pdf": {"entity:B"}}, handler.artifacts
        handler.purge_source_graph_artifacts.assert_awaited_once_with("file.pdf")

        # second call is a no-op (idempotent) and still only targets file.pdf
        await phase.delete("file.pdf", cat)
        assert handler.artifacts == {"other.pdf": {"entity:B"}}, handler.artifacts
        assert handler.purge_source_graph_artifacts.await_count == 2
        assert handler.purge_source_graph_artifacts.await_args_list[1].args == (
            "file.pdf",
        )

        # the other phase shares the same graph cleanup contract
        concept = phases.GraphConceptPhase(cat)
        await concept.delete("file.pdf", cat)
        assert handler.artifacts == {"other.pdf": {"entity:B"}}, handler.artifacts
        print("PASS (b) delete() is source-scoped and idempotent")

    async def test_c_tag_follows_config_only():
        handler = FakeGraphHandler()
        cat = FakeCat(handler)
        spacy = phases.GraphSpacyPhase(cat)
        concept = phases.GraphConceptPhase(cat)

        spacy_first = await spacy.tag({})
        spacy_second = await spacy.tag({})
        assert spacy_first == spacy_second, (spacy_first, spacy_second)
        assert spacy_first == _own_tag(spacy.id, handler.spacy_marker), spacy_first

        concept_first = await concept.tag({})
        concept_second = await concept.tag({})
        assert concept_first == concept_second, (concept_first, concept_second)
        assert concept_first == _own_tag(concept.id, handler.concept_marker)

        handler.spacy_marker = "spacy-v2"
        assert await spacy.tag({}) != spacy_first
        handler.concept_marker = "concept-v2"
        assert await concept.tag({}) != concept_first
        assert handler.llm_calls == 0, "tag() must not call the LLM"
        print("PASS (c) tag() is stable per config and changes with the config")

    async def test_d_process_delegates_to_existing_path():
        handler = FakeGraphHandler()
        cat = FakeCat(handler)

        spacy = phases.GraphSpacyPhase(cat)
        spacy_result = await spacy.process("file.pdf", cat)
        assert spacy_result.status == "done", spacy_result
        handler.run_spacy_extraction_phase.assert_awaited_once_with(
            "file.pdf", cat=cat
        )
        handler._extract_and_link_entities.assert_not_awaited()

        concept = phases.GraphConceptPhase(cat)
        concept_result = await concept.process("file.pdf", cat)
        assert concept_result.status == "done", concept_result
        handler.run_llm_extraction_phase.assert_awaited_once_with(
            "file.pdf", cat=cat
        )
        handler._extract_concept_relations.assert_not_awaited()
        print("PASS (d) process() delegates to the existing ingestion path")

    async def test_e_phase_module_owns_no_protocol_logic():
        """The plugin must NOT re-implement the phase-ingestion protocol.

        Diary writes, staleness, locks, retries and status derivation all
        belong to EffING; the plugin only declares + executes + cleans up.
        Guard the module source so a future edit cannot reintroduce them.
        """
        import inspect

        source = inspect.getsource(phases)
        forbidden = (
            "record_phase_",
            "should_abandon",
            "distributed_lock",
            "derive_source_status",
        )
        found = [name for name in forbidden if name in source]
        assert not found, f"phase module re-implements protocol logic: {found}"
        print("PASS (e) phase module contains no diary/state/lock/retry protocol")

    async def _main():
        await test_a_vector_only_agent_is_skipped()
        await test_b_delete_source_scoped_and_idempotent()
        await test_c_tag_follows_config_only()
        await test_d_process_delegates_to_existing_path()
        await test_e_phase_module_owns_no_protocol_logic()

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
    print("OK phase ABC migration (four tests)")
