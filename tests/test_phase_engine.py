"""Standalone verification for the core-ingestion-phase registration.

Covers the single ``rabbithole_ingestion_phases`` hook and the handler helpers
the phases drive (``run_spacy_extraction_phase``, ``run_llm_extraction_phase``,
``drain_source_tasks``, ``has_pending_source_tasks``,
``get_source_points_by_collection``, ``wipe_tenant_graph_data``):

- ``rabbithole_ingestion_phases`` appends the two phase CLASSES
  (``graph_spacy_extraction`` / ``graph_llm_extraction``, both depending on
  ``parsing_chunking`` + ``embedding``) and returns the accumulator unchanged
  when the plugin phase module is missing or its import raises;
- each phase's ``settings(cat)`` returns ``_spacy_fingerprint`` for
  ``graph_spacy_extraction``, ``_concept_fingerprint`` for
  ``graph_llm_extraction``, and ``None`` for an inactive handler; it never
  calls the LLM;
- ``process(source, cat)`` maps the handler result to a ``PhaseResult``:
  ``skipped`` for an inactive handler, ``done`` (with the NER + similarity
  work / the LLM concept-relations step awaited), ``not_ready`` with
  ``retry_after`` when the source's points are absent, and ``error`` with the
  reason on failure; dispatch to ``run_spacy_extraction_phase`` /
  ``run_llm_extraction_phase`` is asserted directly with AsyncMocks;
- the retired completion gate and the classic fire-and-forget callback are
  GONE (the core owns the source lifecycle and the phases are the only graph
  writer); no engine-selection switch remains;
- the spaCy entity batch write keeps ``REMOVE e.concept_gen,
  e.concept_gen_active`` — the shared Entity-node contract stays internal
  to the plugin (T7 shared-schema guard);
- ``after_vector_memory_transfer_on_agent`` wipes the tenant's Neo4j data
  ONLY on success when the NEW handler is not GraphRAG (host no longer passes
  the old handler name), and never on failure;
- ``wipe_tenant_graph_data`` skips without a recorded config and wipes the
  tenant (tenant-scoped) when one exists.

Runnable:  python test_phase_engine.py
Pure-stdlib + unittest.mock, plain asserts, no pytest, no model loading. All
external packages (cat, langchain_core, spacy, langdetect, pydantic, neo4j)
are stubbed in sys.modules *before* the plugin modules are imported (same
pattern as test_kg_master_switch.py). No EffING package is required: the plugin
phases are plain core ``IngestionPhase`` subclasses.
"""

import asyncio
import os
import sys
import types
from unittest.mock import AsyncMock, Mock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESOLVED_INGESTION = {"name": "Baseline"}



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
    cat_mod.__path__ = []  # look like a package root for submodule imports

    # @hook decorator: supports both bare ``@hook`` and ``@hook(priority=...)``;
    # returns the function unchanged either way.
    def _hook(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

    cat_mod.hook = _hook
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
    models_mod.PointStruct = type("PointStruct", (), {})
    models_mod.DocumentRecall = type("DocumentRecall", (), {})
    models_mod.Record = type("Record", (), {})
    models_mod.ScoredPoint = type("ScoredPoint", (), {})
    models_mod.UpdateResult = type("UpdateResult", (), {})
    sys.modules["cat.services.memory.models"] = models_mod

    factory = types.ModuleType("cat.services.factory")
    factory.__path__ = []
    sys.modules["cat.services.factory"] = factory

    ingestion_factory = types.ModuleType("cat.services.factory.ingestion")

    async def resolved_config_name(lizard):
        return _RESOLVED_INGESTION["name"]

    ingestion_factory.resolved_config_name = resolved_config_name
    sys.modules["cat.services.factory.ingestion"] = ingestion_factory


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
        raise AssertionError("real spaCy must not be loaded inside the standalone test")

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
        # Identity is enough for the stub: the real one wraps a validator fn.
        return fn

    def create_model(name, **kwargs):
        return type(name, (BaseModel,), {})

    setattr(pydantic_mod, "Field", Field)
    setattr(pydantic_mod, "ConfigDict", ConfigDict)
    setattr(pydantic_mod, "BaseModel", BaseModel)
    setattr(pydantic_mod, "BeforeValidator", BeforeValidator)
    setattr(pydantic_mod, "create_model", create_model)
    sys.modules["pydantic"] = pydantic_mod


def _install_stubs():
    _install_cat_stub()
    _install_neo4j_stub()
    _install_langchain_stub()
    _install_langdetect_stub()
    _install_spacy_stubs()
    _install_pydantic_stub()


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeCat:
    """Minimal stand-in for CheshireCat: no ``id`` (agent-scoped), handler set."""

    def __init__(self, handler):
        self.vector_memory_handler = handler
        self.agent_key = "agent_test"

    async def embedder(self):
        return getattr(self.vector_memory_handler, "embedder", None)


class _FakePoint:
    def __init__(self, pid, vector=None, payload=None):
        self.id = pid
        self.vector = vector
        self.payload = payload or {}


def _make_handler(kg=True, cr=True):
    """Build a GraphRAGHandler with the given flags (no driver needed)."""
    from catgraphrag_phase import graphrag_handler

    handler = graphrag_handler.GraphRAGHandler(
        neo4j_uri="bolt://fake",
        neo4j_user="u",
        neo4j_password="p",
        enable_knowledge_graph=kg,
        enable_concept_relations=cr,
    )
    handler.agent_id = "agent_test"
    return handler


def _completed(deps):
    return [{"phase": pid, "gen": gen} for pid, gen in deps]


# ---------------------------------------------------------------------------
# T5 — phase registration (core ``rabbithole_ingestion_phases``)
# ---------------------------------------------------------------------------


def _phase(package, phase_id):
    from catgraphrag_phase import phases

    return {
        "graph_spacy_extraction": phases.GraphSpacyPhase,
        "graph_llm_extraction": phases.GraphConceptPhase,
    }[phase_id]()


def _src(name="file.pdf"):
    return types.SimpleNamespace(name=name, agent_id="agent", chat_id=None)


def test_specs_registers_two_graph_phases():
    from catgraphrag_phase import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    classes = main.rabbithole_ingestion_phases([], cat)
    assert len(classes) == 2, f"expected two phase classes, got {classes}"
    by_id = {cls.id: cls for cls in classes}
    assert set(by_id) == {"graph_spacy_extraction", "graph_llm_extraction"}, (
        f"got {set(by_id)}"
    )
    for cls in classes:
        assert cls.depends_on == ("parsing_chunking", "embedding"), (
            f"{cls.id}: depends_on must be parsing_chunking+embedding, got {cls.depends_on}"
        )
        # the core instantiates the class without arguments
        cls()


def test_specs_preserves_prior_accumulator_entries():
    from catgraphrag_phase import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    accumulator = ["sentinel"]  # opaque prior entry from another plugin
    classes = main.rabbithole_ingestion_phases(accumulator, cat)
    assert classes[0] == "sentinel", "prior accumulator entries must be preserved"
    assert len(classes) == 3, f"got {len(classes)}"


def test_specs_returns_accumulator_when_effing_absent():
    """A missing plugin phase module leaves the accumulator unchanged."""
    from catgraphrag_phase import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    saved = sys.modules.get("catgraphrag_phase.phases")
    sys.modules["catgraphrag_phase.phases"] = None
    try:
        accumulator = ["sentinel"]
        classes = main.rabbithole_ingestion_phases(accumulator, cat)
        assert classes is accumulator, "must return the accumulator unchanged"
    finally:
        if saved is not None:
            sys.modules["catgraphrag_phase.phases"] = saved
        else:
            sys.modules.pop("catgraphrag_phase.phases", None)


def test_specs_returns_accumulator_when_phases_import_raises():
    """A broken phase module must not crash activation."""
    from catgraphrag_phase import main

    handler = _make_handler()
    cat = _FakeCat(handler)

    class _RaisingPhases(types.ModuleType):
        def __getattr__(self, name):
            raise ImportError(f"cannot import {name} from phases")

    saved = sys.modules.get("catgraphrag_phase.phases")
    sys.modules["catgraphrag_phase.phases"] = _RaisingPhases(
        "catgraphrag_phase.phases"
    )
    try:
        accumulator = ["sentinel"]
        classes = main.rabbithole_ingestion_phases(accumulator, cat)
        assert classes is accumulator, (
            "must return the accumulator unchanged when the phases import raises"
        )
    finally:
        if saved is not None:
            sys.modules["catgraphrag_phase.phases"] = saved
        else:
            sys.modules.pop("catgraphrag_phase.phases", None)


def test_marker_returns_spacy_fingerprint():
    handler = _make_handler()
    cat = _FakeCat(handler)
    marker = asyncio.run(_phase("pkg", "graph_spacy_extraction").settings(cat))
    assert marker == handler._spacy_fingerprint(), f"got {marker}"


def test_marker_returns_concept_fingerprint():
    handler = _make_handler()
    cat = _FakeCat(handler)
    marker = asyncio.run(_phase("pkg", "graph_llm_extraction").settings(cat))
    assert marker == asyncio.run(handler._concept_fingerprint()), f"got {marker}"


def test_marker_returns_none_for_unknown_phase():
    class _OtherHandler:
        pass

    cat = _FakeCat(_OtherHandler())
    for phase_id in ("graph_spacy_extraction", "graph_llm_extraction"):
        assert asyncio.run(_phase("pkg", phase_id).settings(cat)) is None


def test_marker_returns_none_when_not_graphrag_handler():
    handler = _make_handler()
    cat = _FakeCat(handler)
    with patch.object(type(handler), "_resolve_agent_llm",
                      new=AsyncMock(side_effect=AssertionError("no LLM in settings"))):
        assert asyncio.run(_phase("pkg", "graph_spacy_extraction").settings(cat)) is not None
        assert asyncio.run(_phase("pkg", "graph_llm_extraction").settings(cat)) is not None


# ---------------------------------------------------------------------------
# T5 — phase execution (``IngestionPhase.process``)
# ---------------------------------------------------------------------------


def test_phase_run_returns_none_for_unknown_phase():
    """An invalid status payload is a truthful error, never a silent success."""
    from catgraphrag_phase import phases

    result = phases._to_result("graph_spacy_extraction", {"status": "weird"})
    assert result.status == "error", result
    assert "invalid status" in (result.reason or ""), result


def test_phase_run_returns_done_when_not_graphrag_handler():
    """B1: a registered-but-inactive phase is skipped, never an ERROR."""
    class _OtherHandler:
        pass

    cat = _FakeCat(_OtherHandler())
    result = asyncio.run(_phase("pkg", "graph_spacy_extraction").process(_src(), cat))
    assert result.status == "skipped", result
    assert "inactive" in (result.reason or ""), result


def test_phase_run_dispatches_spacy_to_run_spacy_extraction_phase():
    """T7: graph_spacy_extraction must dispatch to the handler's
    run_spacy_extraction_phase (awaited, with the cat for LLM resolution)."""
    handler = _make_handler()
    cat = _FakeCat(handler)
    with patch.object(type(handler), "run_spacy_extraction_phase",
                      new=AsyncMock(return_value={"status": "done"})) as run:
        result = asyncio.run(
            _phase("pkg", "graph_spacy_extraction").process(_src(), cat)
        )
    assert result.status == "done", result
    run.assert_awaited_once_with("file.pdf", cat=cat)


def test_phase_run_dispatches_llm_to_run_llm_extraction_phase():
    """T7: graph_llm_extraction must dispatch to the handler's
    run_llm_extraction_phase (awaited, with the cat for LLM resolution)."""
    handler = _make_handler()
    cat = _FakeCat(handler)
    with patch.object(type(handler), "run_llm_extraction_phase",
                      new=AsyncMock(return_value={"status": "done"})) as run:
        result = asyncio.run(
            _phase("pkg", "graph_llm_extraction").process(_src(), cat)
        )
    assert result.status == "done", result
    run.assert_awaited_once_with("file.pdf", cat=cat)


def test_spacy_phase_not_ready_when_points_absent():
    handler = _make_handler()
    cat = _FakeCat(handler)
    with patch.object(type(handler), "get_source_points_by_collection", new=AsyncMock(return_value={})), \
         patch.object(type(handler), "drain_source_tasks", new=AsyncMock()) as drain:
        result = asyncio.run(
            _phase("pkg", "graph_spacy_extraction").process(_src(), cat)
        )
    assert result.status == "not_ready", result
    assert result.retry_after == 5, result
    drain.assert_awaited_once_with("file.pdf")


def test_spacy_phase_done_when_points_present():
    handler = _make_handler()
    cat = _FakeCat(handler)
    points = [
        _FakePoint(
            "doc1",
            vector=[0.1, 0.2, 0.3],
            payload={"page_content": "We use GraphVault daily.", "metadata": {"source": "file.pdf"}},
        )
    ]
    with patch.object(type(handler), "get_source_points_by_collection",
                      new=AsyncMock(return_value={"declarative": points})), \
         patch.object(type(handler), "drain_source_tasks", new=AsyncMock(return_value=False)) as drain, \
         patch.object(type(handler), "_create_similarity_relationships", new=AsyncMock()) as sim, \
         patch.object(type(handler), "_extract_document_graph", new=AsyncMock(
             return_value={"entities": [{"name": "GraphVault"}], "mentions": [], "relations": []}
         )) as ner, \
         patch.object(type(handler), "_write_source_graph", new=AsyncMock()) as write:
        result = asyncio.run(
            _phase("pkg", "graph_spacy_extraction").process(_src(), cat)
        )
    assert result.status == "done", result
    assert result.processed == 1, result
    drain.assert_awaited_once_with("file.pdf")
    sim.assert_awaited_once_with("doc1", [0.1, 0.2, 0.3], "declarative")
    # the NER step runs AWAITED for the source's points (id, content, metadata)
    ner.assert_awaited_once_with(
        "doc1", "We use GraphVault daily.", {"source": "file.pdf"}
    )


def test_spacy_phase_skips_ner_when_entity_extraction_off():
    handler = _make_handler()
    handler._enable_entity_extraction = False
    handler._entity_extractor = None
    cat = _FakeCat(handler)
    points = [
        _FakePoint(
            "doc1",
            vector=[0.1, 0.2, 0.3],
            payload={"page_content": "text", "metadata": {"source": "file.pdf"}},
        )
    ]
    with patch.object(type(handler), "get_source_points_by_collection",
                      new=AsyncMock(return_value={"declarative": points})), \
         patch.object(type(handler), "drain_source_tasks", new=AsyncMock(return_value=False)), \
         patch.object(type(handler), "_create_similarity_relationships", new=AsyncMock()), \
         patch.object(type(handler), "_extract_document_graph", new=AsyncMock()) as ner:
        result = asyncio.run(
            _phase("pkg", "graph_spacy_extraction").process(_src(), cat)
        )
    assert result.status == "done", result
    # extraction disabled: no product count is produced (current behaviour)
    assert result.processed is None, result
    ner.assert_not_awaited()


def test_llm_phase_not_ready_when_points_absent():
    handler = _make_handler()
    cat = _FakeCat(handler)
    with patch.object(type(handler), "get_source_points_by_collection", new=AsyncMock(return_value={})), \
         patch.object(type(handler), "drain_source_tasks", new=AsyncMock()) as drain:
        result = asyncio.run(
            _phase("pkg", "graph_llm_extraction").process(_src(), cat)
        )
    assert result.status == "not_ready", result
    assert result.retry_after == 5, result
    drain.assert_awaited_once_with("file.pdf")


def test_llm_phase_done_when_points_present():
    handler = _make_handler(kg=True, cr=True)
    cat = _FakeCat(handler)
    points = [_FakePoint("doc1", vector=[0.1, 0.2, 0.3])]
    with patch.object(type(handler), "get_source_points_by_collection",
                      new=AsyncMock(return_value={"declarative": points})), \
         patch.object(type(handler), "drain_source_tasks", new=AsyncMock()) as drain, \
         patch.object(type(handler), "_extract_concept_relations", new=AsyncMock()) as extract:
        result = asyncio.run(
            _phase("pkg", "graph_llm_extraction").process(_src(), cat)
        )
    assert result.status == "done", result
    drain.assert_awaited_once_with("file.pdf")
    extract.assert_awaited_once()
    # the LLM step receives the source's points and the cat (LLM resolution)
    _args = extract.await_args.args
    assert _args[0] == "file.pdf"
    assert _args[1] == points


def test_llm_phase_runs_when_kg_off_display_only():
    """``enable_knowledge_graph`` is display-only (todo 6), so the LLM phase
    still runs when it is off; only ``enable_concept_relations`` gates it."""
    handler = _make_handler(kg=False, cr=True)
    cat = _FakeCat(handler)
    points = [_FakePoint("doc1", vector=[0.1, 0.2, 0.3])]
    with patch.object(type(handler), "get_source_points_by_collection",
                      new=AsyncMock(return_value={"declarative": points})), \
         patch.object(type(handler), "drain_source_tasks", new=AsyncMock()), \
         patch.object(type(handler), "_extract_concept_relations", new=AsyncMock()) as extract:
        result = asyncio.run(
            _phase("pkg", "graph_llm_extraction").process(_src(), cat)
        )
    assert result.status == "done", result
    extract.assert_awaited_once()


# ---------------------------------------------------------------------------
# T5 — single graph writer (the classic callback and its engine gate are gone)
# ---------------------------------------------------------------------------


def test_gate_raises_when_pending_tasks():
    """The old completion gate is gone: the core owns the source lifecycle."""
    from catgraphrag_phase import main

    assert not hasattr(main, "before_ingestion_status_completed"), (
        "the plugin must not register a completion gate"
    )


def test_gate_passes_when_clean():
    from catgraphrag_phase import main

    assert hasattr(main, "rabbithole_ingestion_phases")
    assert not hasattr(main, "before_ingestion_status_completed")


def test_gate_noop_when_not_graphrag_handler():
    from catgraphrag_phase import main

    class _OtherHandler:
        pass

    cat = _FakeCat(_OtherHandler())
    result = asyncio.run(
        _phase("pkg", "graph_spacy_extraction").process(_src(), cat)
    )
    assert result.status == "skipped", result


def test_gate_raises_when_has_pending_source_tasks_true():
    """The handler still exposes the pending-task probe the phase drains."""
    handler = _make_handler()

    with patch.object(type(handler), "has_pending_source_tasks",
                      new=Mock(return_value=True)):
        assert handler.has_pending_source_tasks("file.pdf") is True


def test_gate_does_not_import_core_registry():
    """The plugin must not reach into the core ingestion-status registry."""
    import inspect
    from catgraphrag_phase import main

    src = inspect.getsource(main)
    forbidden = (
        "core_plugins." + "ingestion_status",
        "ingestion_status." + "fingerprints",
        "ingestion_" + "phase_pending",
    )
    for ref in forbidden:
        assert ref not in src, f"forbidden reference found: {ref}"


def test_registered_phases_are_the_only_graph_writer():
    """There is ONE graph writer: the classes returned by the core hook."""
    from catgraphrag_phase import main

    assert hasattr(main, "rabbithole_ingestion_phases"), (
        "the core phase-registration hook must be present"
    )
    registered = {
        name
        for name, value in vars(main).items()
        if isinstance(value, types.FunctionType) and not name.startswith("_")
    }
    assert "rabbithole_ingestion_phases" in registered, registered
    assert not any(name.startswith("ingestion_phase_") for name in registered), (
        f"legacy phase hooks still registered: {registered}"
    )


def test_registration_is_import_safe():
    """Registration returns both core phase classes without an engine plugin."""
    from catgraphrag_phase import main

    handler = _make_handler()
    cat = _FakeCat(handler)
    classes = main.rabbithole_ingestion_phases([], cat)
    assert [cls.id for cls in classes] == [
        "graph_spacy_extraction",
        "graph_llm_extraction",
    ], classes
    from catgraphrag_phase import phases as plugin_phases

    for cls in classes:
        assert issubclass(cls, plugin_phases.IngestionPhase), cls


def test_no_engine_selection_switch_remains():
    """No engine-selection switch remains: the phases always own the graph."""
    import inspect
    from catgraphrag_phase import main

    src = inspect.getsource(main)
    for ref in ("EfficientIngestionConfiguration", "_resolved_system_ingestion_config_name"):
        assert ref not in src, f"retired engine switch still present: {ref}"
    assert hasattr(main, "rabbithole_ingestion_phases")


def test_single_phase_path_has_no_selector():
    """No mode truth table: the core phase machine is the only path."""
    from catgraphrag_phase import main

    assert hasattr(main, "rabbithole_ingestion_phases")
    registered = {
        name
        for name, value in vars(main).items()
        if isinstance(value, types.FunctionType) and not name.startswith("_")
    }
    assert not any(name.startswith("ingestion_phase_") for name in registered)
    assert "_phase_path_active" not in registered
    assert "_resolved_system_ingestion_config_name" not in registered



# ---------------------------------------------------------------------------
# T24 — old-store cleanup on switch
# ---------------------------------------------------------------------------


def test_transfer_cleanup_noop_on_failure():
    from catgraphrag_phase import main

    cat = _FakeCat(_make_handler())
    with patch("catgraphrag_phase.graphrag_handler.wipe_tenant_graph_data", new=AsyncMock()) as wipe:
        asyncio.run(main.after_vector_memory_transfer_on_agent(False, cat))
        wipe.assert_not_awaited()


def test_transfer_cleanup_noop_when_new_handler_is_graphrag():
    from catgraphrag_phase import main

    cat = _FakeCat(_make_handler())
    with patch("catgraphrag_phase.graphrag_handler.wipe_tenant_graph_data", new=AsyncMock()) as wipe:
        asyncio.run(main.after_vector_memory_transfer_on_agent(True, cat))
        wipe.assert_not_awaited()


def test_transfer_cleanup_wipes_on_success_when_new_handler_not_graphrag():
    from catgraphrag_phase import main

    cat = _FakeCat(None)
    with patch("catgraphrag_phase.graphrag_handler.wipe_tenant_graph_data", new=AsyncMock()) as wipe:
        asyncio.run(main.after_vector_memory_transfer_on_agent(True, cat))
        wipe.assert_awaited_once_with("agent_test")


def test_wipe_tenant_graph_data_skips_without_config():
    from catgraphrag_phase import graphrag_handler

    graphrag_handler._last_graphrag_config.pop("agent_test", None)
    with patch.object(graphrag_handler.GraphRAGHandler, "_ensure_connected", new=AsyncMock()) as conn:
        asyncio.run(graphrag_handler.wipe_tenant_graph_data("agent_test"))
        conn.assert_not_awaited()


def test_wipe_tenant_graph_data_wipes_tenant():
    from catgraphrag_phase import graphrag_handler

    graphrag_handler._last_graphrag_config["agent_test"] = {
        "neo4j_uri": "bolt://fake",
        "neo4j_user": "u",
        "neo4j_password": "p",
        "neo4j_database": "neo4j",
        "neo4j_kwargs": {},
    }

    class _FakeSessionCM:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *a):
            return False

    captured = {}

    async def _fake_drop(self, session):
        captured["handler"] = self

    try:
        with patch.object(graphrag_handler.GraphRAGHandler, "_ensure_connected", new=AsyncMock()) as conn, \
             patch.object(graphrag_handler.GraphRAGHandler, "_get_session",
                          new=Mock(return_value=_FakeSessionCM())), \
             patch.object(graphrag_handler.GraphRAGHandler, "_drop_tenant_data_in_session", new=_fake_drop):
                asyncio.run(graphrag_handler.wipe_tenant_graph_data("agent_test"))
        conn.assert_awaited_once()
        # the wipe handler must be scoped to the target agent
        assert captured["handler"].agent_id == "agent_test"
    finally:
        graphrag_handler._last_graphrag_config.pop("agent_test", None)


# ---------------------------------------------------------------------------
# T25 — graph-phase write idempotency (restart-safe re-run)
# ---------------------------------------------------------------------------


class _EmptyResult:
    """Minimal async result: no rows, no single record."""

    async def single(self):
        return None

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class _RecordingTx:
    def __init__(self, session):
        self._session = session

    async def run(self, query, **params):
        self._session.queries.append(str(query))
        return _EmptyResult()


class _RecordingSession:
    """Captures every Cypher statement sent through run/execute_write."""

    def __init__(self):
        self.queries = []

    async def run(self, query, **params):
        self.queries.append(str(query))
        return _EmptyResult()

    async def execute_write(self, fn):
        await fn(_RecordingTx(self))


class _RecordingSessionCM:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *a):
        return False


def _capture_session(handler, session):
    """Patch the handler's _get_session to yield a recording session."""
    from unittest.mock import Mock

    return patch.object(
        type(handler),
        "_get_session",
        new=Mock(return_value=_RecordingSessionCM(session)),
    )


def test_concept_store_rerun_issues_identical_merge_queries():
    """Re-running the LLM concept step for the same source+gen must issue
    identical MERGE-based statements (no bare CREATE, no weight accumulation)."""
    from catgraphrag_phase import graphrag_handler

    handler = _make_handler()
    handler.agent_id = "agent_test"

    relations = {
        "concepts": [
            {"type": "CONCEPT", "text": "Overfitting", "_norm": "overfitting"},
            {"type": "CONCEPT", "text": "Generalization", "_norm": "generalization"},
        ],
        "relations": [
            {
                "type": "CAUSES",
                "origin": "Overfitting",
                "destination": "Generalization",
                "text": "Overfitting causes poor generalization",
            },
        ],
    }

    runs = []
    for _ in range(2):
        rec = _RecordingSession()
        with _capture_session(handler, rec):
            asyncio.run(handler._store_concept_relations(
                "agent_test",
                relations,
                source="file.pdf",
                document_ids=["doc1", "doc2"],
                concept_gen="gen_abc",
            ))
        runs.append(rec.queries)

    assert runs[0], "expected captured Cypher statements"
    # Re-running the same source+gen must issue the SAME statements.
    assert runs[0] == runs[1], "re-run must issue identical queries"

    all_queries = "\n".join(runs[0])
    # No bare CREATE anywhere in the concept-store writes.
    assert "CREATE (" not in all_queries, f"bare CREATE found: {all_queries}"
    # Entity / edge / provenance writes are MERGE-based.
    assert "MERGE (z:Entity" in all_queries
    assert "MERGE (s)-[r:RELATED_TO" in all_queries
    assert "MERGE (d)-[:PROVENANCE]->(s)" in all_queries
    # The weight write is idempotent: no `+ 0.5` accumulation on re-run.
    assert "SET r.weight = coalesce(r.weight, 1.0)" in all_queries
    assert "coalesce(r.weight, 1.0) + 0.5" not in all_queries


def test_similar_rel_query_is_merge_based():
    """The SIMILAR_TO_{gen} write (epoch.py create_similar_rel) must be a
    MERGE in both directions — re-running the graph phase for the same source
    must not duplicate similarity edges."""
    from catgraphrag_phase import graphrag_handler

    handler = _make_handler()
    query = handler._compile_query("create_similar_rel", "v1")
    assert "MERGE (a)-[r1:SIMILAR_TO_v1]->(b)" in query
    assert "MERGE (b)-[r2:SIMILAR_TO_v1]->(a)" in query
    assert "CREATE (" not in query


def test_derived_graph_writes_are_merge_based():
    """The structural derived graph (SourceFile / PART_OF / NEXT / CHILD_OF /
    HAS_SUMMARY) must issue only MERGE statements for a source re-run."""
    from catgraphrag_phase import graphrag_handler

    handler = _make_handler()
    handler.agent_id = "agent_test"

    class _FakePoint:
        def __init__(self, pid, chunk_index, parent_id=None):
            self.id = pid
            self.payload = {
                "metadata": {
                    "source": "file.pdf",
                    "chunk_index": chunk_index,
                    "parent_id": parent_id,
                }
            }

    points = [
        _FakePoint("doc1", 0),
        _FakePoint("doc2", 1, parent_id="doc1"),
        _FakePoint("doc3", 2),
    ]

    rec = _RecordingSession()
    with patch.object(type(handler), "_ensure_connected", new=AsyncMock()), \
         _capture_session(handler, rec):
        asyncio.run(handler.create_derived_graph_for_source("file.pdf", points))

    assert rec.queries, "expected captured Cypher statements"
    all_queries = "\n".join(rec.queries)
    # No bare CREATE in the derived-graph writes (Document CREATE lives in
    # add_point_to_tenant and is guarded by the document_id_unique constraint).
    assert "CREATE (" not in all_queries, f"bare CREATE found: {all_queries}"
    assert "MERGE (sf:SourceFile" in all_queries
    assert "MERGE (d)-[:PART_OF]->(sf)" in all_queries
    assert "MERGE (a)-[:NEXT]->(b)" in all_queries
    assert "MERGE (child)-[:CHILD_OF]->(parent)" in all_queries


def test_spacy_batch_query_keeps_concept_gen_removal():
    """T7 shared-schema guard: the spaCy entity batch write must keep
    REMOVE-ing the LLM-owned concept_gen markers. The Entity-node contract
    (concept_gen / concept_gen_active) stays internal to the plugin: the
    spaCy phase must never resurrect stale LLM generations."""
    import inspect
    from catgraphrag_phase import graphrag_handler

    src = inspect.getsource(graphrag_handler)
    assert "REMOVE e.concept_gen, e.concept_gen_active" in src, (
        "spaCy batch entity query must REMOVE the LLM-owned concept_gen markers"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main():
    _install_stubs()
    _run()


def _run():
    # In-memory package pointing at the repo root: imports the plugin modules
    # directly (relative imports resolved via __path__) without executing the
    # plugin __init__.py side effects.
    _pkg = types.ModuleType("catgraphrag_phase")
    _pkg.__path__ = [REPO_ROOT]
    sys.modules["catgraphrag_phase"] = _pkg

    from catgraphrag_phase import graphrag_handler  # noqa: E402, F401
    from catgraphrag_phase import main as main_mod  # noqa: E402, F401
    from catgraphrag_phase import phases as phases_mod  # noqa: E402, F401

    # These tests target the handler bodies the phases delegate to; the real
    # embedder/extractor preparation is covered by test_embedder_injection.
    async def _no_prepare(handler, cat):
        return None

    phases_mod._prepare_handler = _no_prepare

    tests = [
        test_specs_registers_two_graph_phases,
        test_specs_preserves_prior_accumulator_entries,
        test_specs_returns_accumulator_when_effing_absent,
        test_specs_returns_accumulator_when_phases_import_raises,
        test_marker_returns_spacy_fingerprint,
        test_marker_returns_concept_fingerprint,
        test_marker_returns_none_for_unknown_phase,
        test_marker_returns_none_when_not_graphrag_handler,
        test_phase_run_returns_none_for_unknown_phase,
        test_phase_run_returns_done_when_not_graphrag_handler,
        test_phase_run_dispatches_spacy_to_run_spacy_extraction_phase,
        test_phase_run_dispatches_llm_to_run_llm_extraction_phase,
        test_spacy_phase_not_ready_when_points_absent,
        test_spacy_phase_done_when_points_present,
        test_spacy_phase_skips_ner_when_entity_extraction_off,
        test_llm_phase_not_ready_when_points_absent,
        test_llm_phase_done_when_points_present,
        test_llm_phase_runs_when_kg_off_display_only,
        test_gate_raises_when_pending_tasks,
        test_gate_passes_when_clean,
        test_gate_noop_when_not_graphrag_handler,
        test_gate_raises_when_has_pending_source_tasks_true,
        test_gate_does_not_import_core_registry,
        test_registered_phases_are_the_only_graph_writer,
        test_registration_is_import_safe,
        test_no_engine_selection_switch_remains,
        test_single_phase_path_has_no_selector,
        test_transfer_cleanup_noop_on_failure,
        test_transfer_cleanup_noop_when_new_handler_is_graphrag,
        test_transfer_cleanup_wipes_on_success_when_new_handler_not_graphrag,
        test_wipe_tenant_graph_data_skips_without_config,
        test_wipe_tenant_graph_data_wipes_tenant,
        test_concept_store_rerun_issues_identical_merge_queries,
        test_similar_rel_query_is_merge_based,
        test_derived_graph_writes_are_merge_based,
        test_spacy_batch_query_keeps_concept_gen_removal,
    ]

    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()

    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()