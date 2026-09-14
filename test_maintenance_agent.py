"""Standalone verification for the maintenance agent CLI (todo 26).

Covers the audit of ``maintenance_agent.py`` against the new ingestion
phase lifecycle (gen-based staleness, phase machine driving):

- ``--reembed`` / ``--reingest`` / ``--graph`` mark the target phase stale
  (surgical ``completed_phases`` diary edit via ``_mark_phase_stale``) and
  drive the ONE phase machine (``reembed_sources``) — NO raw status-doc
  deletion, NO manual point wipes;
- ``--wipe-graph`` is the ONLY explicit destructive graph step (A0 Cypher,
  5 tenant-filtered statements) and the only graph op besides ``--reingest``
  that requires ``--yes`` (``_DESTRUCTIVE_STEPS``);
- ``--dry-run`` prints the new plan and exits 2 with no writes; destructive
  ops require ``--yes``; ``--graph``/``--wipe-graph`` only for GraphRAG
  agents (plugin-active intersection + per-agent skip gates);
- ``--verify`` reads ``completed_phases``/gens (no-stale-phases check with
  the real ``phase_generation`` algorithm) and keeps the
  ``concept_gen_active == fingerprint`` check gated on concept relations
  being enabled.

Runnable:  python test_maintenance_agent.py
Pure-stdlib + unittest.mock, plain asserts, no pytest, no model loading. All
external packages (cat, langchain_core, spacy, langdetect, pydantic, neo4j)
are stubbed in sys.modules *before* the plugin module is imported (same
pattern as test_ingestion_phase_machine.py). The core phase-machine helpers
(``cat.core_plugins.ingestion_status.fingerprints`` / ``.registry``) are
stubbed with the real ``phase_generation`` logic and fakes for the Redis
reads.
"""

import asyncio
import hashlib
import io
import json
import os
import sys
import tempfile
import types
from contextlib import redirect_stdout
from enum import Enum
from unittest.mock import AsyncMock, Mock, patch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


class _StubLog:
    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Stub installation (called from main(), never at import time)
# ---------------------------------------------------------------------------


def _phase_generation(phase_id, settings_fp, dep_gens):
    """Mirror of the core's ``phase_generation`` (fingerprints.py)."""
    payload = json.dumps(
        {"phase": phase_id, "fp": settings_fp, "deps": dep_gens},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


#: Module-level so the fake cat and the stub installer share the same enum.
VectorMemoryType = Enum(
    "VectorMemoryType", {"DECLARATIVE": "declarative", "EPISODIC": "episodic"}
)


class _FakeSource:
    def __init__(self, name, metadata=None):
        self.name = name
        self.metadata = metadata or {}


class _FakeEmbedder:
    name = "fake_embedder"
    size = 384


class _FakeChunker:
    name = "fake_chunker"


class _FakeResult:
    def __init__(self, value):
        self._value = value

    async def single(self):
        return {"n": self._value} if self._value is not None else None


class _FakeSession:
    """Records queries; returns a scalar per substring key."""

    def __init__(self, scalars=None):
        self.queries = []
        self.scalars = scalars or {}

    async def run(self, query, **params):
        self.queries.append((query, params))
        value = 0
        for key, v in self.scalars.items():
            if key in query:
                value = v
                break
        return _FakeResult(value)


class _FakeSessionCM:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *args):
        return False


class _FakeGraphRAGHandler:
    """Stand-in for the deployed GraphRAGHandler (isinstance target)."""

    def __init__(self, kg=True, cr=True, derived=True):
        self._enable_knowledge_graph = kg
        self._enable_concept_relations = cr
        self._enable_derived_graph = derived
        self._graphrag_detected = True
        self._concept_relations_enabled = bool(kg and cr)
        self._walk_gen = None
        self.agent_id = "agent_test"
        self._scalars = {}
        self.last_session = None
        self._ensure_connected = AsyncMock()
        self.initialize = AsyncMock()
        self._read_generation = AsyncMock(return_value="gen1")
        self._concept_fingerprint = Mock(return_value="fp1")
        self.delete_tenant_points = AsyncMock()

    def _get_session(self):
        self.last_session = _FakeSession(self._scalars)
        return _FakeSessionCM(self.last_session)


class _FakeCat:
    """Minimal stand-in for CheshireCat."""

    def __init__(self, handler, sources=None, agent_key="agent_test"):
        self.agent_key = agent_key
        self._id = agent_key
        self.vector_memory_handler = handler
        self.chunker = _FakeChunker()
        self.large_language_model = object()
        self._sources = sources or []

    async def embedder(self):
        return _FakeEmbedder()

    async def get_stored_sources_with_metadata(self):
        return {
            VectorMemoryType.DECLARATIVE: self._sources,
            VectorMemoryType.EPISODIC: [],
        }


def _install_cat_stub(plugins_dir):
    cat_mod = types.ModuleType("cat")
    cat_mod.__path__ = []
    cat_mod.hook = lambda *a, **kwargs: (lambda f: f)
    cat_mod.log = _StubLog()
    sys.modules["cat"] = cat_mod

    log_mod = types.ModuleType("cat.log")
    log_mod.log = _StubLog()
    sys.modules["cat.log"] = log_mod

    env_mod = types.ModuleType("cat.env")
    env_mod.get_env = Mock(return_value=None)
    sys.modules["cat.env"] = env_mod

    utils_mod = types.ModuleType("cat.utils")
    utils_mod.get_plugins_path = Mock(return_value=plugins_dir)
    sys.modules["cat.utils"] = utils_mod

    db_mod = types.ModuleType("cat.db")
    db_mod.__path__ = []
    sys.modules["cat.db"] = db_mod

    crud_mod = types.ModuleType("cat.db.crud")
    crud_mod.store = AsyncMock()
    crud_mod.read = AsyncMock(return_value=None)
    crud_mod.delete = AsyncMock()
    sys.modules["cat.db.crud"] = crud_mod

    cruds_mod = types.ModuleType("cat.db.cruds")
    cruds_mod.__path__ = []
    sys.modules["cat.db.cruds"] = cruds_mod

    settings_crud = types.ModuleType("cat.db.cruds.settings")
    settings_crud.get_agents_main_keys = AsyncMock(return_value=["agent_test", "system"])
    settings_crud.get_setting_by_name = AsyncMock(return_value=None)
    settings_crud.get_settings_by_category = AsyncMock(return_value=None)
    sys.modules["cat.db.cruds.settings"] = settings_crud

    plugins_crud = types.ModuleType("cat.db.cruds.plugins")
    plugins_crud.get_agents_plugin_keys = AsyncMock(return_value=["agent_test"])
    sys.modules["cat.db.cruds.plugins"] = plugins_crud

    looking = types.ModuleType("cat.looking_glass")
    looking.__path__ = []
    sys.modules["cat.looking_glass"] = looking

    cheshire = types.ModuleType("cat.looking_glass.cheshire_cat")
    cheshire.CheshireCat = type("CheshireCat", (), {"create": AsyncMock()})
    sys.modules["cat.looking_glass.cheshire_cat"] = cheshire

    services = types.ModuleType("cat.services")
    services.__path__ = []
    sys.modules["cat.services"] = services

    factory = types.ModuleType("cat.services.factory")
    factory.__path__ = []
    sys.modules["cat.services.factory"] = factory

    ingestion_factory = types.ModuleType("cat.services.factory.ingestion")
    ingestion_factory.resolved_config_name = AsyncMock(
        return_value="EfficientIngestionConfiguration"
    )
    sys.modules["cat.services.factory.ingestion"] = ingestion_factory

    memory = types.ModuleType("cat.services.memory")
    memory.__path__ = []
    sys.modules["cat.services.memory"] = memory

    models_mod = types.ModuleType("cat.services.memory.models")
    models_mod.VectorMemoryType = VectorMemoryType
    models_mod.PointStruct = type("PointStruct", (), {})
    sys.modules["cat.services.memory.models"] = models_mod

    core_plugins = types.ModuleType("cat.core_plugins")
    core_plugins.__path__ = []
    sys.modules["cat.core_plugins"] = core_plugins

    ingestion_status = types.ModuleType("cat.core_plugins.ingestion_status")
    ingestion_status.__path__ = []
    sys.modules["cat.core_plugins.ingestion_status"] = ingestion_status

    fingerprints = types.ModuleType("cat.core_plugins.ingestion_status.fingerprints")
    fingerprints.build_chunker_fingerprint = AsyncMock(
        return_value={"name": "fake_chunker", "settings": {}}
    )
    fingerprints.build_embedder_fingerprint = AsyncMock(
        return_value={"name": "fake_embedder", "settings": {}}
    )
    fingerprints.build_graphrag_fingerprint = AsyncMock(
        return_value={"name": "Neo4jGraphRAGConfig", "settings": {}}
    )
    fingerprints.phase_generation = _phase_generation
    sys.modules["cat.core_plugins.ingestion_status.fingerprints"] = fingerprints

    registry = types.ModuleType("cat.core_plugins.ingestion_status.registry")
    registry.get_status = AsyncMock(return_value=None)
    registry.list_statuses = AsyncMock(return_value=[])
    registry.delete_status = AsyncMock()

    def _status_key(agent_id, scope, source):
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        return f"agents:{agent_id}:ingestion:{scope}:{digest}"

    registry.status_key = _status_key
    sys.modules["cat.core_plugins.ingestion_status.registry"] = registry

    efficient = types.ModuleType("cat.core_plugins.efficient_ingestion")
    efficient.__path__ = []
    sys.modules["cat.core_plugins.efficient_ingestion"] = efficient

    reembed = types.ModuleType("cat.core_plugins.efficient_ingestion.reembed")
    reembed.reembed_sources = AsyncMock()
    sys.modules["cat.core_plugins.efficient_ingestion.reembed"] = reembed

    plugins_pkg = types.ModuleType("cat.plugins")
    plugins_pkg.__path__ = []
    sys.modules["cat.plugins"] = plugins_pkg

    plugin_pkg = types.ModuleType("cat.plugins.cat_graphrag")
    plugin_pkg.__path__ = []
    sys.modules["cat.plugins.cat_graphrag"] = plugin_pkg

    gh_mod = types.ModuleType("cat.plugins.cat_graphrag.graphrag_handler")
    gh_mod.GraphRAGHandler = _FakeGraphRAGHandler
    sys.modules["cat.plugins.cat_graphrag.graphrag_handler"] = gh_mod


def _install_stubs(plugins_dir):
    _install_cat_stub(plugins_dir)


def _make_plugins_dir():
    """Temp plugins dir with a fake deployed graphrag_handler.py."""
    tmp = tempfile.mkdtemp(prefix="qa_plugins_")
    folder = os.path.join(tmp, "cat_graphrag")
    os.makedirs(folder)
    with open(os.path.join(folder, "graphrag_handler.py"), "w", encoding="utf-8") as fh:
        fh.write("class GraphRAGHandler:\n    pass\n")
    return tmp


def _completed_doc(entries, status="completed"):
    return {"source": "file1.pdf", "status": status, "completed_phases": list(entries)}


def _current_gens():
    """The three gens a fully-current doc must record (stub fingerprints)."""
    pc = _phase_generation("parsing_chunking", {"name": "fake_chunker", "settings": {}}, {})
    em = _phase_generation(
        "embedding",
        {"name": "fake_embedder", "settings": {}},
        {"parsing_chunking": pc},
    )
    gb = _phase_generation(
        "graph_building",
        {"settings": {"name": "Neo4jGraphRAGConfig", "settings": {}}, "concept": "fp1"},
        {"parsing_chunking": pc, "embedding": em},
    )
    return pc, em, gb


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_parse_args_ops_order_wipe_before_graph():
    from maintenance_agent import _parse_args

    with patch.object(sys, "argv", ["maintenance_agent", "--agent", "a1", "--wipe-graph", "--graph"]):
        args = _parse_args()
    assert args.ops == ["wipe_graph", "graph"], f"got {args.ops}"


def test_parse_args_single_ops():
    from maintenance_agent import _parse_args

    with patch.object(sys, "argv", ["maintenance_agent", "--all", "--reembed"]):
        assert _parse_args().ops == ["reembed"]
    with patch.object(sys, "argv", ["maintenance_agent", "--all", "--reingest"]):
        assert _parse_args().ops == ["reingest"]
    with patch.object(sys, "argv", ["maintenance_agent", "--all", "--graph"]):
        assert _parse_args().ops == ["graph"]
    with patch.object(sys, "argv", ["maintenance_agent", "--all", "--wipe-graph"]):
        assert _parse_args().ops == ["wipe_graph"]


def test_parse_args_no_op_exits_2():
    from maintenance_agent import _parse_args

    with patch.object(sys, "argv", ["maintenance_agent", "--all"]):
        try:
            _parse_args()
            raise AssertionError("expected SystemExit")
        except SystemExit as exc:
            assert exc.code == 2


def test_destructive_steps_only_reingest_and_wipe_graph():
    from maintenance_agent import _DESTRUCTIVE_STEPS

    assert set(_DESTRUCTIVE_STEPS.keys()) == {"reingest", "wipe_graph"}, (
        f"reembed/graph are idempotent recomputes and must NOT require --yes, "
        f"got {sorted(_DESTRUCTIVE_STEPS)}"
    )


def test_mark_phase_stale_removes_only_target_phase():
    from cat.core_plugins.ingestion_status import registry
    from cat.db import crud
    from maintenance_agent import _mark_phase_stale

    pc, em, gb = _current_gens()
    doc = _completed_doc(
        [
            {"phase": "parsing_chunking", "gen": pc},
            {"phase": "embedding", "gen": em},
            {"phase": "graph_building", "gen": gb},
        ]
    )
    registry.get_status.return_value = doc
    registry.get_status.side_effect = None
    crud.store.reset_mock()

    handler = _FakeGraphRAGHandler()
    ccat = _FakeCat(handler, sources=[_FakeSource("file1.pdf")])
    marked = asyncio.run(_mark_phase_stale(ccat, "declarative", "embedding"))

    assert marked == 1
    assert crud.store.await_count == 1
    stored = crud.store.await_args.args[1]
    phases = [e["phase"] for e in stored["completed_phases"]]
    assert phases == ["parsing_chunking", "graph_building"], f"got {phases}"
    assert stored["status"] == "completed"  # only the diary entry is dropped


def test_mark_phase_stale_skips_missing_doc_and_unrecorded_phase():
    from cat.core_plugins.ingestion_status import registry
    from cat.db import crud
    from maintenance_agent import _mark_phase_stale

    pc, em, gb = _current_gens()
    registry.get_status.side_effect = [None, _completed_doc([{"phase": "embedding", "gen": em}])]
    crud.store.reset_mock()

    handler = _FakeGraphRAGHandler()
    ccat = _FakeCat(
        handler,
        sources=[_FakeSource("no-doc.pdf"), _FakeSource("no-graph-phase.pdf")],
    )
    marked = asyncio.run(_mark_phase_stale(ccat, "declarative", "graph_building"))

    assert marked == 0, "no doc and no recorded phase must not write"
    assert crud.store.await_count == 0


def test_op_reembed_marks_embedding_and_drives_machine():
    from cat.core_plugins.efficient_ingestion import reembed
    from cat.core_plugins.ingestion_status import registry
    from cat.db import crud
    from maintenance_agent import _op_reembed

    pc, em, gb = _current_gens()
    registry.get_status.return_value = _completed_doc(
        [
            {"phase": "parsing_chunking", "gen": pc},
            {"phase": "embedding", "gen": em},
            {"phase": "graph_building", "gen": gb},
        ]
    )
    registry.get_status.side_effect = None
    crud.store.reset_mock()
    reembed.reembed_sources.reset_mock()

    handler = _FakeGraphRAGHandler()
    ccat = _FakeCat(handler, sources=[_FakeSource("file1.pdf")])
    ok = asyncio.run(_op_reembed(ccat, handler, "declarative"))

    assert ok is True
    # surgical diary edit: only the embedding entry is dropped
    stored = crud.store.await_args.args[1]
    phases = [e["phase"] for e in stored["completed_phases"]]
    assert phases == ["parsing_chunking", "graph_building"], f"got {phases}"
    # the machine is driven with the collection
    assert reembed.reembed_sources.await_count == 1
    args = reembed.reembed_sources.await_args.args
    assert args[0] is ccat and str(args[1]) == "VectorMemoryType.DECLARATIVE"
    # NO raw deletion anywhere
    assert registry.delete_status.await_count == 0
    assert handler.delete_tenant_points.await_count == 0


def test_op_reingest_marks_parsing_and_drives_machine():
    from cat.core_plugins.efficient_ingestion import reembed
    from cat.core_plugins.ingestion_status import registry
    from cat.db import crud
    from maintenance_agent import _op_reingest

    pc, em, gb = _current_gens()
    registry.get_status.return_value = _completed_doc(
        [
            {"phase": "parsing_chunking", "gen": pc},
            {"phase": "embedding", "gen": em},
            {"phase": "graph_building", "gen": gb},
        ]
    )
    registry.get_status.side_effect = None
    crud.store.reset_mock()
    reembed.reembed_sources.reset_mock()

    handler = _FakeGraphRAGHandler()
    ccat = _FakeCat(handler, sources=[_FakeSource("file1.pdf")])
    ok = asyncio.run(_op_reingest(ccat, handler, "declarative"))

    assert ok is True
    stored = crud.store.await_args.args[1]
    phases = [e["phase"] for e in stored["completed_phases"]]
    assert phases == ["embedding", "graph_building"], f"got {phases}"
    assert reembed.reembed_sources.await_count == 1
    # the parsing phase's clean-sweep IS the wipe: no manual point deletion
    assert handler.delete_tenant_points.await_count == 0
    assert registry.delete_status.await_count == 0


def test_op_graph_marks_graph_building_and_drives_machine():
    from cat.core_plugins.efficient_ingestion import reembed
    from cat.core_plugins.ingestion_status import registry
    from cat.db import crud
    from maintenance_agent import _op_graph

    pc, em, gb = _current_gens()
    registry.get_status.return_value = _completed_doc(
        [
            {"phase": "parsing_chunking", "gen": pc},
            {"phase": "embedding", "gen": em},
            {"phase": "graph_building", "gen": gb},
        ]
    )
    registry.get_status.side_effect = None
    crud.store.reset_mock()
    reembed.reembed_sources.reset_mock()

    handler = _FakeGraphRAGHandler()
    ccat = _FakeCat(handler, sources=[_FakeSource("file1.pdf")])
    ok = asyncio.run(_op_graph(ccat, handler, "declarative"))

    assert ok is True
    stored = crud.store.await_args.args[1]
    phases = [e["phase"] for e in stored["completed_phases"]]
    assert phases == ["parsing_chunking", "embedding"], f"got {phases}"
    assert reembed.reembed_sources.await_count == 1
    # the phase path is idempotent: no wipe, no deletion
    assert handler.last_session is None or handler.last_session.queries == []


def test_op_wipe_graph_runs_5_tenant_filtered_statements():
    from maintenance_agent import _op_wipe_graph

    handler = _FakeGraphRAGHandler()
    handler._walk_gen = "g7"
    ccat = _FakeCat(handler, sources=[_FakeSource("file1.pdf")])
    ok = asyncio.run(_op_wipe_graph(ccat, handler, "declarative"))

    assert ok is True
    queries = [q for q, _ in handler.last_session.queries]
    assert len(queries) == 5, f"expected 5 A0 statements, got {len(queries)}"
    for q in queries:
        assert "{tenant_id" in q, f"statement not tenant-filtered: {q}"
    assert any("SIMILAR_TO_g7" in q for q in queries), "versioned SIMILAR_TO_<gen> missing"


def test_graph_wipe_statements_all_tenant_filtered():
    from maintenance_agent import _graph_wipe_statements

    stmts = _graph_wipe_statements("g1")
    assert len(stmts) == 5
    for s in stmts:
        assert "{tenant_id" in s, f"not tenant-filtered: {s}"
    assert "SIMILAR_TO_g1" in stmts[3]


def test_run_agent_skips_graph_when_not_detected():
    from cat.db.cruds import settings as crud_settings
    from cat.looking_glass.cheshire_cat import CheshireCat
    from maintenance_agent import _run_agent

    handler = _FakeGraphRAGHandler()
    handler._graphrag_detected = False
    ccat = _FakeCat(handler, sources=[_FakeSource("file1.pdf")])
    CheshireCat.create.return_value = ccat
    crud_settings.get_setting_by_name.return_value = None

    from cat.core_plugins.efficient_ingestion import reembed

    reembed.reembed_sources.reset_mock()
    ok = asyncio.run(_run_agent("agent_test", ["graph", "wipe_graph"]))
    assert ok is True, "skips must not fail the agent"
    # both graph ops skipped: no machine drive, no wipe
    assert reembed.reembed_sources.await_count == 0


def test_run_agent_skips_graph_when_concept_relations_off():
    from cat.db.cruds import settings as crud_settings
    from cat.looking_glass.cheshire_cat import CheshireCat
    from maintenance_agent import _run_agent

    handler = _FakeGraphRAGHandler(kg=True, cr=False)
    ccat = _FakeCat(handler, sources=[_FakeSource("file1.pdf")])
    CheshireCat.create.return_value = ccat
    crud_settings.get_setting_by_name.return_value = {
        "value": {"enable_knowledge_graph": True, "enable_concept_relations": False}
    }

    from cat.core_plugins.efficient_ingestion import reembed

    reembed.reembed_sources.reset_mock()
    ok = asyncio.run(_run_agent("agent_test", ["graph"]))
    assert ok is True
    assert reembed.reembed_sources.await_count == 0


def test_run_agent_wipe_graph_runs_without_engine_check():
    from cat.db.cruds import settings as crud_settings
    from cat.looking_glass.cheshire_cat import CheshireCat
    from cat.services.factory import ingestion as ingestion_factory
    from maintenance_agent import _run_agent

    handler = _FakeGraphRAGHandler()
    ccat = _FakeCat(handler, sources=[_FakeSource("file1.pdf")])
    CheshireCat.create.return_value = ccat
    crud_settings.get_setting_by_name.return_value = {
        "value": {"enable_knowledge_graph": True, "enable_concept_relations": True}
    }
    ingestion_factory.resolved_config_name.reset_mock()

    ok = asyncio.run(_run_agent("agent_test", ["wipe_graph"]))
    assert ok is True
    assert len(handler.last_session.queries) == 5
    # the wipe is handler-level: the ingestion engine is never consulted
    assert ingestion_factory.resolved_config_name.await_count == 0


def test_run_agent_engine_refusal_fails_op():
    from cat.db.cruds import settings as crud_settings
    from cat.looking_glass.cheshire_cat import CheshireCat
    from cat.services.factory import ingestion as ingestion_factory
    from maintenance_agent import _run_agent

    handler = _FakeGraphRAGHandler()
    ccat = _FakeCat(handler, sources=[_FakeSource("file1.pdf")])
    CheshireCat.create.return_value = ccat
    crud_settings.get_setting_by_name.return_value = {
        "value": {"enable_knowledge_graph": True, "enable_concept_relations": True}
    }
    ingestion_factory.resolved_config_name.return_value = "BaseIngestionConfiguration"

    from cat.core_plugins.efficient_ingestion import reembed

    reembed.reembed_sources.reset_mock()
    ok = asyncio.run(_run_agent("agent_test", ["reembed"]))
    assert ok is False, "base engine must fail the op"
    assert reembed.reembed_sources.await_count == 0


def test_run_agent_per_op_isolation():
    from cat.core_plugins.efficient_ingestion import reembed
    from cat.core_plugins.ingestion_status import registry
    from cat.db import crud
    from cat.db.cruds import settings as crud_settings
    from cat.looking_glass.cheshire_cat import CheshireCat
    from maintenance_agent import _run_agent

    handler = _FakeGraphRAGHandler()
    ccat = _FakeCat(handler, sources=[_FakeSource("file1.pdf")])
    CheshireCat.create.return_value = ccat
    crud_settings.get_setting_by_name.return_value = {
        "value": {"enable_knowledge_graph": True, "enable_concept_relations": True}
    }
    registry.get_status.return_value = None
    registry.get_status.side_effect = None
    crud.store.reset_mock()
    reembed.reembed_sources.reset_mock()
    # undo the engine-refusal test's pollution: the efficient engine is active
    from cat.services.factory import ingestion as ingestion_factory

    ingestion_factory.resolved_config_name.return_value = "EfficientIngestionConfiguration"

    # reembed raises (no status doc -> nothing marked is fine, but force a
    # failure via the machine), graph must still run
    reembed.reembed_sources.side_effect = [RuntimeError("boom"), None]

    ok = asyncio.run(_run_agent("agent_test", ["reembed", "graph"]))
    assert ok is False, "a failed op must fail the agent"
    assert reembed.reembed_sources.await_count == 2, "graph op must still run after reembed failed"


def test_build_ops_plan_plugin_not_active_skip():
    from cat.db.cruds import plugins as crud_plugins
    from maintenance_agent import _build_ops_plan

    crud_plugins.get_agents_plugin_keys.return_value = []
    args = Mock(agent=None, ops=["graph"], collection="declarative")
    plan = asyncio.run(_build_ops_plan(args))
    assert len(plan) == 1
    assert plan[0]["skip_reason"] == "plugin-not-active"


def test_build_ops_plan_unknown_agent_exits_2():
    from maintenance_agent import _build_ops_plan

    args = Mock(agent="ghost", ops=["reembed"], collection="declarative")
    try:
        asyncio.run(_build_ops_plan(args))
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert exc.code == 2


def test_print_plan_new_steps():
    from maintenance_agent import _PlanEntry, _print_plan

    plan: list[_PlanEntry] = [
        {"agent_id": "agent_test", "ops": ["reingest", "graph", "wipe_graph"], "collection": "declarative"}
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_plan(plan)
    out = buf.getvalue()
    assert "DRY-RUN PLAN" in out
    assert "op=reingest" in out and "destructive=yes" in out
    assert "op=graph" in out and "destructive=no" in out
    assert "op=wipe_graph" in out and "destructive=yes" in out
    assert "mark stale" in out
    assert "reembed_sources" in out
    assert "SIMILAR_TO_<gen>" in out
    assert "(dry-run: exiting 2, nothing was written)" in out


def test_doc_phases_current_true_when_all_match():
    from maintenance_agent import _doc_phases_current

    pc, em, gb = _current_gens()
    ctx = {
        "chunker_fp": {"name": "fake_chunker", "settings": {}},
        "embedder_fp": {"name": "fake_embedder", "settings": {}},
        "graphrag_fp": {"name": "Neo4jGraphRAGConfig", "settings": {}},
        "concept_fp": "fp1",
        "phase_generation": _phase_generation,
    }
    doc = _completed_doc(
        [
            {"phase": "parsing_chunking", "gen": pc},
            {"phase": "embedding", "gen": em},
            {"phase": "graph_building", "gen": gb},
        ]
    )
    assert _doc_phases_current(doc, ctx, _FakeGraphRAGHandler()) is True


def test_doc_phases_current_false_when_embedding_stale():
    from maintenance_agent import _doc_phases_current

    pc, em, gb = _current_gens()
    ctx = {
        "chunker_fp": {"name": "fake_chunker", "settings": {}},
        "embedder_fp": {"name": "fake_embedder", "settings": {}},
        "graphrag_fp": {"name": "Neo4jGraphRAGConfig", "settings": {}},
        "concept_fp": "fp1",
        "phase_generation": _phase_generation,
    }
    doc = _completed_doc(
        [
            {"phase": "parsing_chunking", "gen": pc},
            {"phase": "embedding", "gen": "stale-embedding-gen"},
            {"phase": "graph_building", "gen": gb},
        ]
    )
    assert _doc_phases_current(doc, ctx, _FakeGraphRAGHandler()) is False


def test_doc_phases_current_false_when_graph_building_missing():
    from maintenance_agent import _doc_phases_current

    pc, em, _ = _current_gens()
    ctx = {
        "chunker_fp": {"name": "fake_chunker", "settings": {}},
        "embedder_fp": {"name": "fake_embedder", "settings": {}},
        "graphrag_fp": {"name": "Neo4jGraphRAGConfig", "settings": {}},
        "concept_fp": "fp1",
        "phase_generation": _phase_generation,
    }
    doc = _completed_doc(
        [
            {"phase": "parsing_chunking", "gen": pc},
            {"phase": "embedding", "gen": em},
        ]
    )
    assert _doc_phases_current(doc, ctx, _FakeGraphRAGHandler()) is False


def test_verify_agent_gen_based_check_passes():
    from cat.core_plugins.ingestion_status import registry
    from maintenance_agent import _verify_agent

    pc, em, gb = _current_gens()
    registry.list_statuses.return_value = [
        _completed_doc(
            [
                {"phase": "parsing_chunking", "gen": pc},
                {"phase": "embedding", "gen": em},
                {"phase": "graph_building", "gen": gb},
            ]
        )
    ]
    handler = _FakeGraphRAGHandler()
    handler._scalars = {
        "IS NULL": 0,
        ")-[r:MENTIONS]->()": 3,
        "SIMILAR_TO_gen1": 2,
        "concept_gen_active": "fp1",
        "NOT (e)-[:MENTIONS]-()": 0,
        "SourceFile": 1,
    }
    ccat = _FakeCat(handler, sources=[_FakeSource("file1.pdf")])
    checks = asyncio.run(_verify_agent(ccat, handler, "declarative"))

    names = {name for name, _ in checks}
    assert "status-completed-no-stale-phases" in names
    assert all(ok for _, ok in checks), f"failed checks: {[(n, o) for n, o in checks if not o]}"


def test_verify_agent_stale_gen_fails_check():
    from cat.core_plugins.ingestion_status import registry
    from maintenance_agent import _verify_agent

    pc, em, _ = _current_gens()
    registry.list_statuses.return_value = [
        _completed_doc(
            [
                {"phase": "parsing_chunking", "gen": pc},
                {"phase": "embedding", "gen": "stale"},
            ]
        )
    ]
    handler = _FakeGraphRAGHandler()
    handler._scalars = {
        "IS NULL": 0,
        ")-[r:MENTIONS]->()": 3,
        "SIMILAR_TO_gen1": 2,
        "concept_gen_active": "fp1",
        "NOT (e)-[:MENTIONS]-()": 0,
        "SourceFile": 1,
    }
    ccat = _FakeCat(handler, sources=[_FakeSource("file1.pdf")])
    checks = asyncio.run(_verify_agent(ccat, handler, "declarative"))

    by_name = {name: ok for name, ok in checks}
    assert by_name["status-completed-no-stale-phases"] is False


def test_verify_agent_fallback_name_check_when_fingerprints_missing():
    from cat.core_plugins.ingestion_status import registry
    from maintenance_agent import _verify_agent

    pc, em, gb = _current_gens()
    registry.list_statuses.return_value = [
        _completed_doc(
            [
                {"phase": "parsing_chunking", "gen": pc},
                {"phase": "embedding", "gen": em},
                {"phase": "graph_building", "gen": gb},
            ]
        )
    ]
    handler = _FakeGraphRAGHandler()
    handler._scalars = {
        "IS NULL": 0,
        ")-[r:MENTIONS]->()": 3,
        "SIMILAR_TO_gen1": 2,
        "concept_gen_active": "fp1",
        "NOT (e)-[:MENTIONS]-()": 0,
        "SourceFile": 1,
    }
    ccat = _FakeCat(handler, sources=[_FakeSource("file1.pdf")])

    saved = sys.modules.pop("cat.core_plugins.ingestion_status.fingerprints")
    try:
        checks = asyncio.run(_verify_agent(ccat, handler, "declarative"))
    finally:
        sys.modules["cat.core_plugins.ingestion_status.fingerprints"] = saved

    names = {name for name, _ in checks}
    assert "status-completed-active-embedder-chunker" in names
    assert "status-completed-no-stale-phases" not in names


def test_verify_agent_concept_gen_gated_when_disabled():
    from cat.core_plugins.ingestion_status import registry
    from maintenance_agent import _verify_agent

    pc, em, gb = _current_gens()
    registry.list_statuses.return_value = [
        _completed_doc(
            [
                {"phase": "parsing_chunking", "gen": pc},
                {"phase": "embedding", "gen": em},
                {"phase": "graph_building", "gen": gb},
            ]
        )
    ]
    handler = _FakeGraphRAGHandler(kg=True, cr=False)
    handler._scalars = {
        "IS NULL": 0,
        ")-[r:MENTIONS]->()": 3,
        "SIMILAR_TO_gen1": 2,
        "NOT (e)-[:MENTIONS]-()": 0,
        "SourceFile": 1,
    }
    ccat = _FakeCat(handler, sources=[_FakeSource("file1.pdf")])
    checks = asyncio.run(_verify_agent(ccat, handler, "declarative"))

    names = {name for name, _ in checks}
    assert "concept-gen-active" not in names, "marker check only valid when concept relations are on"


def test_recompute_phase_gens_returns_ctx():
    from maintenance_agent import _recompute_phase_gens

    handler = _FakeGraphRAGHandler()
    ccat = _FakeCat(handler)
    ctx = asyncio.run(_recompute_phase_gens(ccat, handler))
    assert ctx is not None
    assert ctx["chunker_fp"]["name"] == "fake_chunker"
    assert ctx["concept_fp"] == "fp1"
    assert callable(ctx["phase_generation"])


def test_main_dry_run_exits_2_no_writes():
    import maintenance_agent as ma

    plan = [{"agent_id": "agent_test", "ops": ["reingest", "graph"], "collection": "declarative"}]
    with (
        patch.object(ma, "_runtime_guard", return_value=True),
        patch.object(ma, "_build_ops_plan", AsyncMock(return_value=plan)),
        patch.object(ma, "_run_agent", AsyncMock(return_value=True)) as run_agent,
        patch.object(sys, "argv", ["maintenance_agent", "--all", "--reingest", "--graph", "--dry-run"]),
    ):
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                asyncio.run(ma.main())
            raise AssertionError("expected SystemExit 2")
        except SystemExit as exc:
            assert exc.code == 2
    assert "DRY-RUN PLAN" in buf.getvalue()
    run_agent.assert_not_awaited()


def test_main_reingest_without_yes_exits_2():
    import maintenance_agent as ma

    plan = [{"agent_id": "agent_test", "ops": ["reingest"], "collection": "declarative"}]
    with (
        patch.object(ma, "_runtime_guard", return_value=True),
        patch.object(ma, "_build_ops_plan", AsyncMock(return_value=plan)),
        patch.object(ma, "_run_agent", AsyncMock(return_value=True)) as run_agent,
        patch.object(sys, "argv", ["maintenance_agent", "--all", "--reingest"]),
    ):
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                asyncio.run(ma.main())
            raise AssertionError("expected SystemExit 2")
        except SystemExit as exc:
            assert exc.code == 2
    assert "ABORTED" in buf.getvalue()
    run_agent.assert_not_awaited()


def test_main_reembed_without_yes_runs():
    import maintenance_agent as ma

    plan = [{"agent_id": "agent_test", "ops": ["reembed"], "collection": "declarative"}]
    with (
        patch.object(ma, "_runtime_guard", return_value=True),
        patch.object(ma, "_build_ops_plan", AsyncMock(return_value=plan)),
        patch.object(ma, "_run_agent", AsyncMock(return_value=True)) as run_agent,
        patch.object(sys, "argv", ["maintenance_agent", "--all", "--reembed"]),
    ):
        try:
            asyncio.run(ma.main())
        except SystemExit as exc:
            assert exc.code == 0, f"idempotent reembed must exit 0, got {exc.code}"
    run_agent.assert_awaited_once()


def test_main_verify_ok_exits_0():
    import maintenance_agent as ma

    plan = [{"agent_id": "agent_test", "ops": ["reembed"], "collection": "declarative"}]
    handler = _FakeGraphRAGHandler()
    ccat = _FakeCat(handler)
    with (
        patch.object(ma, "_runtime_guard", return_value=True),
        patch.object(ma, "_build_ops_plan", AsyncMock(return_value=plan)),
        patch.object(ma, "_run_agent", AsyncMock(return_value=True)),
        patch.object(ma, "_bootstrap_agent", AsyncMock(return_value=(ccat, handler, None))),
        patch.object(ma, "_verify_agent", AsyncMock(return_value=[("check", True)])),
        patch.object(sys, "argv", ["maintenance_agent", "--all", "--reembed", "--verify"]),
    ):
        try:
            asyncio.run(ma.main())
        except SystemExit as exc:
            assert exc.code == 0, f"verify-ok must exit 0, got {exc.code}"


def test_import_safety_no_cat_at_module_level():
    """Importing the module must not touch the cat runtime (import-safe)."""
    import maintenance_agent as ma

    assert not hasattr(ma, "cat"), "cat must never be imported at module level"
    assert ma._DESTRUCTIVE_STEPS  # module-level constants are fine


def main():
    _install_stubs(_make_plugins_dir())

    # Import the module under test AFTER the stubs are in place.
    sys.path.insert(0, REPO_ROOT)
    import maintenance_agent  # noqa: F401  (import-safety probe)

    tests = [
        test_parse_args_ops_order_wipe_before_graph,
        test_parse_args_single_ops,
        test_parse_args_no_op_exits_2,
        test_destructive_steps_only_reingest_and_wipe_graph,
        test_mark_phase_stale_removes_only_target_phase,
        test_mark_phase_stale_skips_missing_doc_and_unrecorded_phase,
        test_op_reembed_marks_embedding_and_drives_machine,
        test_op_reingest_marks_parsing_and_drives_machine,
        test_op_graph_marks_graph_building_and_drives_machine,
        test_op_wipe_graph_runs_5_tenant_filtered_statements,
        test_graph_wipe_statements_all_tenant_filtered,
        test_run_agent_skips_graph_when_not_detected,
        test_run_agent_skips_graph_when_concept_relations_off,
        test_run_agent_wipe_graph_runs_without_engine_check,
        test_run_agent_engine_refusal_fails_op,
        test_run_agent_per_op_isolation,
        test_build_ops_plan_plugin_not_active_skip,
        test_build_ops_plan_unknown_agent_exits_2,
        test_print_plan_new_steps,
        test_doc_phases_current_true_when_all_match,
        test_doc_phases_current_false_when_embedding_stale,
        test_doc_phases_current_false_when_graph_building_missing,
        test_verify_agent_gen_based_check_passes,
        test_verify_agent_stale_gen_fails_check,
        test_verify_agent_fallback_name_check_when_fingerprints_missing,
        test_verify_agent_concept_gen_gated_when_disabled,
        test_recompute_phase_gens_returns_ctx,
        test_main_dry_run_exits_2_no_writes,
        test_main_reingest_without_yes_exits_2,
        test_main_reembed_without_yes_runs,
        test_main_verify_ok_exits_0,
        test_import_safety_no_cat_at_module_level,
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