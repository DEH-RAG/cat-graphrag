"""Standalone, import-safe tests for the LLM-failure -> phase-state mapping.

Covers the graph concept phase contract (``_extract_concept_relations`` /
``_llm_extract_relations``): a failed LLM call must never become a silent
``done``. The five cases:

  (a) ``StructuredLLMError`` with unparsable raw -> ``not_ready`` on the first
      attempt, ``error`` afterwards (never ``done``);
  (b) no usable LLM -> ``error`` with the configuration reason;
  (c) empty-but-valid output on non-empty text -> ``not_ready`` then ``error``
      carrying the raw model text;
  (d) a clean extraction still yields ``done`` with the product;
  (e) a partial lenient parse (items dropped) -> ``error`` with ``dropped=<n>``.

The Cat plugin loader imports every ``.py`` under a plugin folder at
activation, so this whole file is inert on import: every import, class and
test lives inside ``if __name__ == "__main__":``. External dependencies are
stubbed in ``sys.modules`` before the plugin module is loaded. Run it as a
plain script:

    python test_llm_error_mapping.py
"""

if __name__ == "__main__":
    import asyncio
    import importlib
    import json
    import os
    import sys
    import types
    from unittest.mock import AsyncMock, patch

    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
    PKG = "catgraphrag_llmtest"

    # ------------------------------------------------------------------
    # Minimal external stubs (installed BEFORE importing the plugin)
    # ------------------------------------------------------------------

    class _StubLog:
        def info(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    def _install_cat_stub():
        cat_mod = types.ModuleType("cat")
        cat_mod.__path__ = []
        cat_mod.hook = lambda *a, **kwargs: (lambda f: f)
        cat_mod.RecallSettings = type("RecallSettings", (), {})
        cat_mod.VectorDatabaseSettings = type("VectorDatabaseSettings", (), {})
        cat_mod.Embeddings = type("Embeddings", (), {})
        cat_mod.AgenticWorkflowTask = type("AgenticWorkflowTask", (), {})

        class _BaseHandler:
            def __init__(self, **kwargs):
                pass

        cat_mod.BaseVectorDatabaseHandler = _BaseHandler
        cat_mod.log = _StubLog()
        sys.modules["cat"] = cat_mod

        log_mod = types.ModuleType("cat.log")
        log_mod.log = _StubLog()
        sys.modules["cat.log"] = log_mod

        looking = types.ModuleType("cat.looking_glass")
        looking.__path__ = []
        sys.modules["cat.looking_glass"] = looking

        stray = types.ModuleType("cat.looking_glass.stray_cat")
        stray.StrayCat = type("StrayCat", (), {})
        sys.modules["cat.looking_glass.stray_cat"] = stray

        services = types.ModuleType("cat.services")
        services.__path__ = []
        sys.modules["cat.services"] = services

        memory = types.ModuleType("cat.services.memory")
        memory.__path__ = []
        sys.modules["cat.services.memory"] = memory

        models_mod = types.ModuleType("cat.services.memory.models")

        class _PointStruct:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        models_mod.PointStruct = _PointStruct
        models_mod.DocumentRecall = type("DocumentRecall", (), {})
        models_mod.Record = type("Record", (), {})
        models_mod.ScoredPoint = type("ScoredPoint", (), {})
        models_mod.UpdateResult = type("UpdateResult", (), {})
        sys.modules["cat.services.memory.models"] = models_mod

    def _install_neo4j_stub():
        neo4j_mod = types.ModuleType("neo4j")
        neo4j_mod.AsyncGraphDatabase = type("AsyncGraphDatabase", (), {})
        neo4j_mod.AsyncDriver = type("AsyncDriver", (), {})
        neo4j_mod.AsyncSession = type("AsyncSession", (), {})
        sys.modules["neo4j"] = neo4j_mod

        exceptions_mod = types.ModuleType("neo4j.exceptions")
        exceptions_mod.Neo4jError = type("Neo4jError", (), {})
        sys.modules["neo4j.exceptions"] = exceptions_mod

    def _install_langchain_stub():
        lc = types.ModuleType("langchain_core")
        lc.__path__ = []
        sys.modules["langchain_core"] = lc

        docs = types.ModuleType("langchain_core.documents")
        docs.Document = type("Document", (), {})
        sys.modules["langchain_core.documents"] = docs

    def _install_langdetect_stub():
        langdetect_mod = types.ModuleType("langdetect")

        class DetectorFactory:
            seed = 0

        setattr(langdetect_mod, "DetectorFactory", DetectorFactory)
        setattr(langdetect_mod, "detect_langs", lambda text: [])
        sys.modules["langdetect"] = langdetect_mod

    def _install_spacy_stubs():
        def _forbidden(*args, **kwargs):
            raise AssertionError("real spaCy must not be loaded inside the test")

        spacy_mod = types.ModuleType("spacy")
        spacy_mod.__path__ = []
        setattr(spacy_mod, "load", _forbidden)
        sys.modules["spacy"] = spacy_mod

        util_mod = types.ModuleType("spacy.util")
        setattr(util_mod, "is_package", lambda name: True)
        sys.modules["spacy.util"] = util_mod

        cli_mod = types.ModuleType("spacy.cli")
        cli_mod.__path__ = []
        sys.modules["spacy.cli"] = cli_mod

        download_mod = types.ModuleType("spacy.cli.download")
        setattr(download_mod, "download", _forbidden)
        sys.modules["spacy.cli.download"] = download_mod

        language_mod = types.ModuleType("spacy.language")
        setattr(language_mod, "Language", type("Language", (), {}))
        sys.modules["spacy.language"] = language_mod

        tokens_mod = types.ModuleType("spacy.tokens")
        setattr(tokens_mod, "Doc", type("Doc", (), {}))
        sys.modules["spacy.tokens"] = tokens_mod

    def _install_pydantic_stub():
        pydantic_mod = types.ModuleType("pydantic")

        class Field:
            def __init__(self, default=None, *, default_factory=None, **kwargs):
                self.default = default
                self.default_factory = default_factory

        class ConfigDict(dict):
            pass

        class BaseModel:
            def __init__(self, **kwargs):
                for name, value in kwargs.items():
                    setattr(self, name, value)

        def BeforeValidator(fn):
            return fn

        def create_model(name, **kwargs):
            return type(name, (BaseModel,), {})

        setattr(pydantic_mod, "Field", Field)
        setattr(pydantic_mod, "ConfigDict", ConfigDict)
        setattr(pydantic_mod, "BaseModel", BaseModel)
        setattr(pydantic_mod, "BeforeValidator", BeforeValidator)
        setattr(pydantic_mod, "create_model", create_model)
        sys.modules["pydantic"] = pydantic_mod

    _install_cat_stub()
    _install_neo4j_stub()
    _install_langchain_stub()
    _install_langdetect_stub()
    _install_spacy_stubs()
    _install_pydantic_stub()

    _pkg = types.ModuleType(PKG)
    _pkg.__path__ = [REPO_ROOT]
    sys.modules[PKG] = _pkg

    graphrag_handler = importlib.import_module(f"{PKG}.graphrag_handler")
    structured_llm = importlib.import_module(f"{PKG}.structured_llm")

    StructuredLLMError = structured_llm.StructuredLLMError

    # ------------------------------------------------------------------
    # Fakes / helpers
    # ------------------------------------------------------------------

    class _Point:
        def __init__(self, pid, text):
            self.id = pid
            self.payload = {"page_content": text}

    def _points(text="Some document text with concepts."):
        return [_Point("doc1", text)]

    class _Result:
        """Stand-in for ``StructuredLLMResult`` (only ``.model``/``.text``)."""

        def __init__(self, model, text):
            self.model = model
            self.text = text

    class _FakeStructuredLLM:
        behavior = None
        calls = 0

        async def run(self, prompt, llm, output_model):
            type(self).calls += 1
            return type(self).behavior()

    def _make_handler():
        handler = graphrag_handler.GraphRAGHandler(
            neo4j_uri="bolt://fake",
            neo4j_user="u",
            neo4j_password="p",
            enable_concept_relations=True,
        )
        handler.agent_id = "agent_test"
        handler._store_concept_relations = AsyncMock()
        handler._concept_fingerprint = AsyncMock(return_value="gen-test")
        return handler

    def _reset_fake():
        _FakeStructuredLLM.calls = 0
        _FakeStructuredLLM.behavior = None

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    async def test_a_structured_error_is_never_done():
        _reset_fake()
        _FakeStructuredLLM.behavior = lambda: (_ for _ in ()).throw(
            StructuredLLMError("unparsable", "not json at all")
        )
        handler = _make_handler()

        with patch.object(graphrag_handler, "StructuredLLM", _FakeStructuredLLM):
            first = await handler._extract_concept_relations(
                "src-a", _points(), llm=object()
            )
            second = await handler._extract_concept_relations(
                "src-a", _points(), llm=object()
            )

        assert first["status"] == "not_ready", first
        assert first["status"] != "done", first
        assert "not json at all" in (first.get("reason") or ""), first
        assert second["status"] == "error", second
        assert "not json at all" in (second.get("reason") or ""), second
        assert _FakeStructuredLLM.calls == 2, _FakeStructuredLLM.calls
        handler._store_concept_relations.assert_not_awaited()
        print("PASS (a) StructuredLLMError -> not_ready then error, never done")

    async def test_b_no_llm_is_configuration_error():
        handler = _make_handler()
        handler._resolve_agent_llm = AsyncMock(return_value=None)

        outcome = await handler._extract_concept_relations(
            "src-b", _points(), llm=None, stray_cat=None
        )

        assert outcome["status"] == "error", outcome
        assert outcome["reason"] == (
            "no LLM configured for graph concept extraction"
        ), outcome
        handler._store_concept_relations.assert_not_awaited()
        print("PASS (b) no LLM -> error with the configuration reason")

    async def test_c_empty_valid_is_not_ready_then_error():
        _reset_fake()
        raw = '{"concepts": [], "relations": []}'
        _FakeStructuredLLM.behavior = lambda: _Result(
            {"concepts": [], "relations": []}, raw
        )
        handler = _make_handler()

        with patch.object(graphrag_handler, "StructuredLLM", _FakeStructuredLLM):
            first = await handler._extract_concept_relations(
                "src-c", _points(), llm=object()
            )
            second = await handler._extract_concept_relations(
                "src-c", _points(), llm=object()
            )

        assert first["status"] == "not_ready", first
        assert raw in (first.get("reason") or ""), first
        assert second["status"] == "error", second
        assert raw in (second.get("reason") or ""), second
        handler._store_concept_relations.assert_not_awaited()
        print("PASS (c) empty-but-valid -> not_ready then error with raw text")

    async def test_d_clean_extraction_is_done_with_product():
        _reset_fake()
        product = {
            "concepts": [
                {"type": "CONCEPT", "text": "Algebra"},
                {"type": "CONCEPT", "text": "Calculus"},
            ],
            "relations": [
                {
                    "type": "PREREQUISITE_FOR",
                    "origin": "Algebra",
                    "destination": "Calculus",
                    "text": "Algebra is a prerequisite for Calculus",
                }
            ],
        }
        _FakeStructuredLLM.behavior = lambda: _Result(product, json.dumps(product))
        handler = _make_handler()

        with patch.object(graphrag_handler, "StructuredLLM", _FakeStructuredLLM):
            outcome = await handler._extract_concept_relations(
                "src-d", _points(), llm=object()
            )

        assert outcome["status"] == "done", outcome
        assert outcome["processed"] == 1, outcome
        handler._store_concept_relations.assert_awaited_once()
        stored = handler._store_concept_relations.await_args.args[1]
        assert len(stored["concepts"]) == 2, stored
        assert len(stored["relations"]) == 1, stored
        print("PASS (d) clean extraction -> done with the product")

    async def test_e_partial_extraction_reports_dropped_count():
        _reset_fake()
        raw = json.dumps(
            {
                "concepts": [
                    {"type": "CONCEPT", "text": "A"},
                    {"type": "CONCEPT", "text": "B"},
                ],
                "relations": [
                    {"type": "IS_A", "origin": "A", "destination": "B", "text": ""},
                    {"type": "NOT_A_TYPE", "origin": "A", "destination": "B", "text": ""},
                    {"type": "IS_A", "origin": "", "destination": "B", "text": ""},
                ],
            }
        )
        _FakeStructuredLLM.behavior = lambda: (_ for _ in ()).throw(
            StructuredLLMError("schema validation failed", raw)
        )
        handler = _make_handler()

        with patch.object(graphrag_handler, "StructuredLLM", _FakeStructuredLLM):
            outcome = await handler._extract_concept_relations(
                "src-e", _points(), llm=object()
            )

        assert outcome["status"] == "error", outcome
        assert "dropped=2" in (outcome.get("reason") or ""), outcome
        handler._store_concept_relations.assert_not_awaited()
        print("PASS (e) partial extraction -> error with dropped=2")

    async def _main():
        await test_a_structured_error_is_never_done()
        await test_b_no_llm_is_configuration_error()
        await test_c_empty_valid_is_not_ready_then_error()
        await test_d_clean_extraction_is_done_with_product()
        await test_e_partial_extraction_reports_dropped_count()

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
    print("OK LLM error mapping (five tests)")
