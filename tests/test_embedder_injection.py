#!/usr/bin/env python3
"""MyGRAPH injects the agent embedder before graph extraction.

The embedder is injected on two independent paths:

* ``before_rabbithole_stores_documents`` — fired by the core during the
  ``embedding`` phase (before the graph phases run);
* ``IngestionPhase.process`` — the phase makes its handler self-sufficient, so
  a resumed/retried graph phase that skips embedding still holds the embedder
  (and an initialized entity extractor).

These tests load the real MyGRAPH ``main.py`` / ``phases.py``; its heavy
relative modules are replaced by local fakes, so no Neo4j/spaCy/Redis is
needed.

Standalone script (plugin rules): the host imports every ``.py`` under the
plugin folder, so the ENTIRE body lives inside ``if __name__ == "__main__":``
and importing this file executes zero code. Run it as a plain script from the
host checkout:

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-graphrag.my/tests/test_embedder_injection.py
"""

if __name__ == "__main__":
    import asyncio
    import importlib.util
    import os
    import sys
    import traceback
    import types
    from pathlib import Path

    PLUGIN_DIR = Path("/root/git/plugins/cat-graphrag.my")
    MAIN_PATH = PLUGIN_DIR / "main.py"

    os.environ.setdefault("CAT_CRYPTO_KEY", "graphrag-test-crypto-key")
    os.environ.setdefault("CAT_CRYPTO_SALT", "graphrag-test-crypto-salt")

    class FakeEmbedder:
        """Stand-in for the agent-resolved embedder instance (identity matters)."""

    class FakeEntityExtractor:
        def __init__(self):
            self.ensure_initialized_calls = 0

        async def ensure_initialized(self):
            self.ensure_initialized_calls += 1

    class FakeGraphRAGHandler:
        """Mirrors the real handler's embedder property + once-only alignment."""

        def __init__(self):
            self._embedder = None
            self._alignment_done = False
            self.align_calls = 0
            self.spacy_calls = 0
            self.llm_calls = 0
            self.seen_embedder = "UNSET"
            self._entity_extractor = FakeEntityExtractor()

        @property
        def embedder(self):
            return self._embedder

        @embedder.setter
        def embedder(self, value):
            self._embedder = value

        @property
        def entity_extractor(self):
            return self._entity_extractor

        async def _align_embedder_lazy(self):
            if self._alignment_done or self._embedder is None:
                return
            self._alignment_done = True
            self.align_calls += 1

        async def run_spacy_extraction_phase(self, source, cat=None):
            self.spacy_calls += 1
            self.seen_embedder = self._embedder
            return {"status": "done"}

        async def run_llm_extraction_phase(self, source, cat=None):
            self.llm_calls += 1
            self.seen_embedder = self._embedder
            return {"status": "done"}

        def _spacy_fingerprint(self):
            return "spacy"

        async def _concept_fingerprint(self):
            return "concept"

    def _cleanup(package_name):
        for name in list(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)

    def _load_plugin_main(package_name):
        package = types.ModuleType(package_name)
        package.__path__ = [str(PLUGIN_DIR)]
        package.__package__ = package_name
        sys.modules[package_name] = package

        handler = types.ModuleType(package_name + ".graphrag_handler")
        setattr(handler, "GraphRAGHandler", FakeGraphRAGHandler)
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

    class Source:
        def __init__(self, name="file.pdf"):
            self.name = name
            self.agent_id = "agent_test"
            self.chat_id = None
            self.type = "file"

    def _cat(handler, embedder):
        async def resolve_embedder():
            return embedder

        return types.SimpleNamespace(
            vector_memory_handler=handler,
            agent_key="agent_test",
            embedder=resolve_embedder,
        )

    async def _test_phase_injects_embedder_before_extraction():
        """A resumed graph phase gets the embedder without the embedding phase."""
        module = _load_plugin_main("_mygraph_embedder_phase")
        try:
            phases = __import__("_mygraph_embedder_phase.phases", fromlist=["phases"])
            handler = FakeGraphRAGHandler()
            embedder = FakeEmbedder()
            cat = _cat(handler, embedder)

            result = await phases.GraphSpacyPhase().process(Source(), cat)

            assert result.status == "done", result
            assert handler.embedder is embedder
            assert handler.align_calls == 1
            assert handler.entity_extractor.ensure_initialized_calls == 1
            assert handler.seen_embedder is embedder
        finally:
            _cleanup("_mygraph_embedder_phase")

    async def _test_stores_documents_hook_keeps_identical_behaviour():
        """The core-fired stores hook injects embedder, extractor and chunk_index."""
        module = _load_plugin_main("_mygraph_embedder_classic")
        try:
            handler = FakeGraphRAGHandler()
            embedder = FakeEmbedder()
            cat = _cat(handler, embedder)

            docs = [types.SimpleNamespace(metadata={})]
            returned = await module.before_rabbithole_stores_documents.function(docs, cat)

            assert returned is docs
            assert handler.embedder is embedder
            assert handler.entity_extractor.ensure_initialized_calls == 1
            assert docs[0].metadata["chunk_index"] == 0
        finally:
            _cleanup("_mygraph_embedder_classic")

    async def _test_injection_twice_is_harmless():
        """Re-injecting (retries, the stores hook plus a phase) changes no state."""
        module = _load_plugin_main("_mygraph_embedder_idempotent")
        try:
            phases = __import__(
                "_mygraph_embedder_idempotent.phases", fromlist=["phases"]
            )
            handler = FakeGraphRAGHandler()
            embedder = FakeEmbedder()
            cat = _cat(handler, embedder)

            await phases.GraphSpacyPhase().process(Source(), cat)
            first = handler.embedder
            await phases.GraphConceptPhase().process(Source(), cat)

            assert handler.embedder is first is embedder
            assert handler.align_calls == 1

            docs = [types.SimpleNamespace(metadata={})]
            await module.before_rabbithole_stores_documents.function(docs, cat)
            await module.before_rabbithole_stores_documents.function(docs, cat)

            assert handler.embedder is embedder
            assert handler.align_calls == 1
            assert docs[0].metadata["chunk_index"] == 0
        finally:
            _cleanup("_mygraph_embedder_idempotent")

    _tests = [
        ("test_phase_injects_embedder_before_extraction",
         _test_phase_injects_embedder_before_extraction),
        ("test_stores_documents_hook_keeps_identical_behaviour",
         _test_stores_documents_hook_keeps_identical_behaviour),
        ("test_injection_twice_is_harmless", _test_injection_twice_is_harmless),
    ]
    _failed = 0
    for _name, _fn in _tests:
        try:
            asyncio.run(_fn())
            print(f"PASS {_name}")
        except Exception as _exc:  # noqa: BLE001 - report and continue
            _failed += 1
            traceback.print_exc()
            print(f"FAIL {_name}: {_exc}")

    print(f"\ntest_embedder_injection: {len(_tests) - _failed}/{len(_tests)} passed")
    raise SystemExit(1 if _failed else 0)

