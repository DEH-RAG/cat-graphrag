#!/usr/bin/env python3
"""Dual-mode, end-to-end proof on the real MyCAT app.

The same small corpus (``tests/mocks/sample.txt``) is ingested on the real app
(``create_app`` + ``LifespanManager``) through both ingestion engines:

* **classic mode** - the system ``ingestion`` setting is the core engine: the
  GraphRAG classic callback (``after_rabbithole_stored_documents`` ->
  ``create_derived_graph_for_source``) is the only classic writer and the
  EffING phase diary holds no graph phase;
* **efficient mode** - the system ``ingestion`` setting is
  ``EfficientIngestionConfiguration``: the classic callback is inert
  (zero calls) and the two registered graph phases (``graph_spacy_extraction``
  + ``graph_llm_extraction``) do the work, recorded in the source's
  ``completed_phases`` diary with their markers.

Real dependencies
-----------------
* Redis: the dedicated host test db (``CAT_REDIS_DB=1``), never production;
* Qdrant: an in-memory client installed like the host ``conftest``;
* Neo4j: the live instance at ``bolt://127.0.0.1:7687``. Neo4j Community
  supports a single database, so isolation is a unique tenant id
  (``mygraph_dual_<uuid>``) and every node carrying that tenant is deleted in
  ``finally``. If Neo4j is unreachable the test FAILS explicitly (never skips
  silently).

The MyGRAPH plugin is loaded through a temporary ``cat/plugins/cat_graphrag``
symlink (the source folder ``cat-graphrag.my`` is not a valid Python package
name); the symlink is removed in ``finally``. No CAT core file and no plugin
behaviour is changed here.

Standalone script (plugin rules): the host imports every ``.py`` under the
plugin folder, so the ENTIRE body lives inside ``if __name__ == "__main__":``
and importing this file executes zero code. The host pytest fixtures are
reproduced inline (dedicated Redis db, in-memory Qdrant, plugin-folder link,
background-task immediacy) because this suite drives the real host app. Run it
as a plain script from the host checkout:

    cd /root/git/grinning-cat-core
    CAT_REDIS_HOST=<redis host> .venv/bin/python /root/git/plugins/cat-graphrag.my/tests/test_dual_mode.py
"""

if __name__ == "__main__":
    import os
    import sys

    # The app refuses to start without a strong CAT_JWT_SECRET and must never
    # touch the production Redis db: set both BEFORE importing cat.startup
    # (which creates the app at import time), exactly like the host conftest.
    os.environ.setdefault("CAT_JWT_SECRET", "test_only_jwt_secret_0123456789_abcdefghij")
    os.environ["CAT_REDIS_DB"] = "1"

    import asyncio
    import contextlib
    import importlib
    import inspect
    import json
    import shutil
    import time
    import traceback
    import types
    import uuid
    from pathlib import Path

    from asgi_lifespan import LifespanManager
    from httpx import ASGITransport, AsyncClient
    from neo4j import AsyncGraphDatabase
    from qdrant_client import AsyncQdrantClient

    from cat import utils
    from cat.auth import auth_utils
    from cat.env import get_env, get_env_int
    from cat.looking_glass.mad_hatter.plugin import Plugin
    from cat.services.factory.vector_db import QdrantHandler
    from cat.startup import create_app

    TEST_REDIS_DB = "1"
    API_KEY = "meow_http"
    JWT_SECRET = "meow_jwt_test_secret_at_least_32_characters"
    AGENT_ID = "agent_test"

    PLUGIN_SRC = Path("/root/git/plugins/cat-graphrag.my")
    PLUGIN_ID = "cat_graphrag"
    PLUGINS_DIR = Path(utils.get_base_path()) / "plugins"
    PLUGIN_LINK = PLUGINS_DIR / PLUGIN_ID

    NEO4J_URI = "bolt://127.0.0.1:7687"
    NEO4J_USER = "neo4j"
    NEO4J_DATABASE = "neo4j"
    NEO4J_AUTH_CONTAINERS = ("minimal-neo4j-1", "neo4j-db")

    SAMPLE_FILE = Path("tests/mocks/sample.txt")
    SAMPLE_NAME = SAMPLE_FILE.name
    SAMPLE_CONTENT_TYPE = "text/plain"

    CORE_ENGINE = "CoreIngestionConfiguration"
    EFFICIENT_ENGINE = "EfficientIngestionConfiguration"

    # Deterministic product handed to the concept phase by the fake LLM: unique
    # names/text so the graph assertions prove the concepts came from this stub
    # (never from spaCy or from an empty silent success).
    FAKE_CONCEPT_ALPHA = "dual mode alpha"
    FAKE_CONCEPT_BETA = "dual mode beta"
    FAKE_RELATION_TEXT = "dual mode alpha causally affects dual mode beta"

    # Handler instances created after the spies are installed (test-only): the
    # classic background entity extraction is not awaited by the request, so the
    # tests join it explicitly before counting.
    _HANDLER_INSTANCES = []

    # ---------------------------------------------------------------------------
    # Host-fixture reproduction (conftest.py) without pytest
    # ---------------------------------------------------------------------------
    _MISSING = object()

    class _MonkeyPatch:
        """Minimal ``pytest.MonkeyPatch`` (setattr / setitem / undo)."""

        def __init__(self):
            self._undo = []

        def setattr(self, target, name, value=_MISSING, raising=True):
            if value is _MISSING:
                # pytest-style two-argument form: setattr("dotted.path", value)
                value = name
                module_name, _, attr = target.rpartition(".")
                target = importlib.import_module(module_name)
                name = attr
            elif isinstance(target, str):
                module_name, _, attr = target.rpartition(".")
                target = importlib.import_module(module_name)
                name = attr
            had = hasattr(target, name)
            old = getattr(target, name, None)
            setattr(target, name, value)
            if had:
                self._undo.append(lambda t=target, n=name, o=old: setattr(t, n, o))
            else:
                self._undo.append(lambda t=target, n=name: delattr(t, n))

        def setitem(self, mapping, key, value):
            had = key in mapping
            old = mapping.get(key)
            mapping[key] = value
            if had:
                self._undo.append(lambda m=mapping, k=key, o=old: m.__setitem__(k, o))
            else:
                self._undo.append(lambda m=mapping, k=key: m.__delitem__(k))

        def undo(self):
            while self._undo:
                self._undo.pop()()

    def _mock_classes(mp, memory_client):
        def mock_init_vector_database(self, *args, **kwargs):
            super(QdrantHandler, self).__init__()
            self._client = memory_client
            self.save_memory_snapshots = kwargs.get("save_memory_snapshots", False)

        mp.setattr(QdrantHandler, "__init__", mock_init_vector_database)
        mp.setattr(QdrantHandler, "close", lambda self: None)

        def mock_get_redis_kwargs():
            return {
                "host": get_env("CAT_REDIS_HOST"),
                "port": get_env_int("CAT_REDIS_PORT"),
                "db": TEST_REDIS_DB,
                "encoding": "utf-8",
                "decode_responses": True,
            }

        mp.setattr("cat.db.database.get_redis_kwargs", mock_get_redis_kwargs)

        mp.setattr(utils, "get_plugins_path", lambda: "tests/mocks/mock_plugin_folder/")
        mp.setattr(utils, "get_file_manager_root_storage_path", lambda: "tests/data/storage")

        async def mock_install_requirements(self):
            pass

        mp.setattr(Plugin, "_install_requirements", mock_install_requirements)
        mp.setattr(auth_utils, "extract_agent_id_from_request", lambda: AGENT_ID)

        async def immediate_task(background_tasks, func, *args, **kwargs):
            await func(*args, **kwargs) if inspect.iscoroutinefunction(func) else func(*args, **kwargs)

        mp.setattr("cat.routes.routes_utils.run_background_task", immediate_task)

    async def _clean_up():
        from cat.db.database import get_sync_db

        to_be_removed = [
            "tests/mocks/mock_plugin.zip",
            "tests/mocks/mock_plugin_fast_reply.zip",
            "tests/mocks/mock_plugin_agent_fast_reply.zip",
            "tests/mocks/mock_plugin_toggle_listener.zip",
            "tests/mocks/mock_plugin_multimodal_embedder.zip",
            "tests/mocks/mock_plugin_with_dependencies.zip",
            "tests/mocks/mock_plugin/settings.json",
            "tests/mocks/mock_plugin_folder/mock_plugin",
            "tests/mocks/mock_plugin_folder/mock_plugin_fast_reply",
            "tests/mocks/mock_plugin_folder/mock_plugin_agent_fast_reply",
            "tests/mocks/mock_plugin_folder/mock_plugin_consistent",
            "tests/mocks/mock_plugin_folder/mock_plugin_toggle_listener",
            "tests/mocks/mock_plugin_folder/mock_plugin_multimodal_embedder",
            "tests/mocks/mock_plugin_folder/mock_plugin_with_dependencies",
            "tests/mocks/empty_folder",
            "tests/data",
        ]
        for tbr in to_be_removed:
            if os.path.exists(tbr):
                if os.path.isdir(tbr):
                    shutil.rmtree(tbr)
                else:
                    os.remove(tbr)

        db = get_sync_db()
        bound_db = db.connection_pool.connection_kwargs.get("db", os.environ.get("CAT_REDIS_DB"))
        if str(bound_db) != TEST_REDIS_DB:
            raise RuntimeError(
                f"Refusing to flushdb(): bound to Redis db={bound_db!r}, expected test db={TEST_REDIS_DB!r}. "
                "The Database singleton is probably pointing at production. Aborting to prevent data loss."
            )
        db.flushdb()

    async def _run_encapsulated(test_fn):
        utils.singleton.instances.clear()
        memory_client = AsyncQdrantClient(":memory:")
        mp = _MonkeyPatch()
        _mock_classes(mp, memory_client)
        await _clean_up()
        try:
            await test_fn(mp)
        finally:
            try:
                await _clean_up()
            finally:
                utils.singleton.instances.clear()
                mp.undo()
                collections = await memory_client.get_collections()
                for collection in collections.collections:
                    await memory_client.delete_collection(collection.name)
                await memory_client.close()

    @contextlib.asynccontextmanager
    async def _app_client(mp):
        """The real FastAPI app, with the host Redis/Qdrant test doubles."""
        if PLUGIN_LINK.exists() or PLUGIN_LINK.is_symlink():
            target = os.readlink(PLUGIN_LINK) if PLUGIN_LINK.is_symlink() else str(PLUGIN_LINK)
            if Path(target).resolve() != PLUGIN_SRC.resolve():
                raise AssertionError(f"{PLUGIN_LINK} already exists and points to {target}")
        else:
            PLUGIN_LINK.symlink_to(PLUGIN_SRC, target_is_directory=True)

        # the host conftest points the plugin folder at tests/mocks: restore the
        # real one so the plugin is discovered by the real app
        mp.setattr(utils, "get_plugins_path", lambda: str(PLUGINS_DIR))

        previous_api_key = os.environ.get("CAT_API_KEY")
        previous_jwt = os.environ.get("CAT_JWT_SECRET")
        os.environ["CAT_API_KEY"] = API_KEY
        os.environ["CAT_JWT_SECRET"] = JWT_SECRET
        app = None
        try:
            app = create_app()
            async with LifespanManager(app) as manager:
                async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac:
                    ac._fastapi_test_app = app
                    yield ac
        finally:
            # no leftover server: shutdown_app deletes the lizard from the app state
            if app is not None:
                assert not hasattr(app.state, "lizard")
            if previous_api_key is None:
                os.environ.pop("CAT_API_KEY", None)
            else:
                os.environ["CAT_API_KEY"] = previous_api_key
            if previous_jwt is None:
                os.environ.pop("CAT_JWT_SECRET", None)
            else:
                os.environ["CAT_JWT_SECRET"] = previous_jwt
            if PLUGIN_LINK.is_symlink() and Path(os.readlink(PLUGIN_LINK)) == PLUGIN_SRC:
                PLUGIN_LINK.unlink()
            assert not PLUGIN_LINK.exists() and not PLUGIN_LINK.is_symlink()

    # ---------------------------------------------------------------------------
    # Neo4j helpers (real instance; tenant-scoped cleanup in finally)
    # ---------------------------------------------------------------------------
    def _neo4j_password() -> str:
        """Resolve the Neo4j password without committing a secret.

        Priority: ``CAT_NEO4J_PASSWORD`` / ``NEO4J_PASSWORD`` env, then the
        ``NEO4J_AUTH`` environment of a running local Neo4j container (the dev
        stack). A missing password is a hard, explicit failure.
        """
        password = os.environ.get("CAT_NEO4J_PASSWORD") or os.environ.get("NEO4J_PASSWORD")
        if password:
            return password

        import subprocess

        for container in NEO4J_AUTH_CONTAINERS:
            try:
                proc = subprocess.run(
                    ["docker", "inspect", container, "--format", "{{json .Config.Env}}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except Exception:  # noqa: BLE001 - try the next container
                continue
            if proc.returncode != 0:
                continue
            try:
                env = json.loads(proc.stdout)
            except json.JSONDecodeError:
                continue
            for entry in env:
                if entry.startswith("NEO4J_AUTH="):
                    credentials = entry.partition("=")[2]
                    _, _, password = credentials.partition("/")
                    if password:
                        return password

        raise RuntimeError(
            "Neo4j password not found: set CAT_NEO4J_PASSWORD/NEO4J_PASSWORD or run "
            f"one of the local containers {NEO4J_AUTH_CONTAINERS} with NEO4J_AUTH."
        )

    def _neo4j_driver():
        return AsyncGraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, _neo4j_password()), connection_timeout=10
        )

    async def _neo4j_counts(tenant: str) -> dict:
        driver = _neo4j_driver()
        try:
            async with driver.session(database=NEO4J_DATABASE) as session:

                async def one(query):
                    result = await session.run(query, tenant=tenant)
                    record = await result.single()
                    await result.consume()
                    return record["c"] if record else 0

                return {
                    "nodes": await one("MATCH (n {tenant_id: $tenant}) RETURN count(n) AS c"),
                    "relationships": await one(
                        "MATCH (a {tenant_id: $tenant})-[r]->(b {tenant_id: $tenant}) RETURN count(r) AS c"
                    ),
                    "documents": await one("MATCH (d:Document {tenant_id: $tenant}) RETURN count(d) AS c"),
                    "entities": await one("MATCH (e:Entity {tenant_id: $tenant}) RETURN count(e) AS c"),
                    "source_files": await one("MATCH (sf:SourceFile {tenant_id: $tenant}) RETURN count(sf) AS c"),
                }
        finally:
            await driver.close()

    async def _wipe_tenant(tenant: str) -> dict:
        driver = _neo4j_driver()
        try:
            async with driver.session(database=NEO4J_DATABASE) as session:
                result = await session.run("MATCH (n {tenant_id: $tenant}) DETACH DELETE n", tenant=tenant)
                await result.consume()
        finally:
            await driver.close()
        return await _neo4j_counts(tenant)

    async def _settle_counts(tenant: str, timeout: float = 15.0) -> dict:
        """Wait until the graph stops changing (classic extraction is background).

        The 2 s quiescence window also absorbs the Neo4j driver's deadlock-retry
        backoff: the classic background extraction can be retried after ~1 s, so a
        shorter window would occasionally return before the entities are written.
        """
        previous = None
        stable_since = None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            current = await _neo4j_counts(tenant)
            if current != previous:
                previous = current
                stable_since = time.monotonic()
            elif stable_since is not None and time.monotonic() - stable_since >= 2.0:
                return current
            await asyncio.sleep(0.05)
        return previous or await _neo4j_counts(tenant)

    async def _assert_neo4j_reachable() -> None:
        driver = _neo4j_driver()
        try:
            async with driver.session(database=NEO4J_DATABASE) as session:
                result = await session.run("RETURN 1 AS x")
                await result.single()
                await result.consume()
        except Exception as exc:  # noqa: BLE001 - surfaced explicitly, never silently skipped
            raise AssertionError(f"Neo4j is unreachable at {NEO4J_URI}: {exc!r}")
        finally:
            await driver.close()

    # ---------------------------------------------------------------------------
    # Spies on the real plugin handler class (test-only instrumentation)
    # ---------------------------------------------------------------------------
    def _install_spies(mp) -> dict:
        handler_module = importlib.import_module("cat.plugins.cat_graphrag.graphrag_handler")
        cls = handler_module.GraphRAGHandler
        _HANDLER_INSTANCES.clear()

        counts = {
            "classic": 0,  # create_derived_graph_for_source from main.py hook
            "internal": 0,  # create_derived_graph_for_source from add_points_to_tenant
            "spacy_phase": 0,
            "llm_phase": 0,
            "background_extracts": {},  # document_id -> count from add_points_to_tenant
            "phase_rerun_extracts": {},  # document_id -> count from the awaited phase re-run
            "empty_pass_extracts": 0,
        }

        original_init = cls.__init__
        original_classic = cls.create_derived_graph_for_source
        original_spacy = cls.run_spacy_extraction_phase
        original_llm = cls.run_llm_extraction_phase
        original_extract = cls._extract_document_graph

        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            _HANDLER_INSTANCES.append(self)

        async def classic(self, source, stored_points, cat=None):
            caller = inspect.currentframe().f_back.f_globals.get("__name__", "")
            if caller.endswith(".cat_graphrag.main"):
                counts["classic"] += 1
            else:
                counts["internal"] += 1
            return await original_classic(self, source, stored_points, cat)

        async def spacy(self, source, cat=None):
            counts["spacy_phase"] += 1
            # Attribute extraction calls reliably: the phase first drains the
            # background batch, so any extraction seen while the phase body runs is
            # the deterministic re-extraction (restart/retry path), not background.
            self._dual_mode_in_spacy = True
            try:
                return await original_spacy(self, source, cat=cat)
            finally:
                self._dual_mode_in_spacy = False

        async def llm(self, source, cat=None):
            counts["llm_phase"] += 1
            return await original_llm(self, source, cat=cat)

        async def extract(self, document_id, content, metadata):
            if getattr(self, "_dual_mode_in_spacy", False):
                counts["phase_rerun_extracts"][document_id] = counts["phase_rerun_extracts"].get(document_id, 0) + 1
            else:
                counts["background_extracts"][document_id] = counts["background_extracts"].get(document_id, 0) + 1
            if getattr(self, "_dual_mode_empty_pass", False):
                counts["empty_pass_extracts"] += 1
            return await original_extract(self, document_id, content, metadata)

        mp.setattr(cls, "__init__", init)
        mp.setattr(cls, "create_derived_graph_for_source", classic)
        mp.setattr(cls, "run_spacy_extraction_phase", spacy)
        mp.setattr(cls, "run_llm_extraction_phase", llm)
        mp.setattr(cls, "_extract_document_graph", extract)
        return counts

    async def _drain_recorded_handlers(source: str) -> None:
        """Join the classic background extraction the request does not await."""
        for handler in list(_HANDLER_INSTANCES):
            try:
                await handler.drain_source_tasks(source)
            except Exception:  # noqa: BLE001 - the graph assertions are the signal
                pass
        pending = [
            task
            for handler in _HANDLER_INSTANCES
            for task in getattr(handler, "_pending_entity_tasks", [])
            if not task.done()
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def _install_empty_pass_marker(mp) -> None:
        """Flag handler instances while EffING's empty-vector staging pass runs.

        ``_store_empty_vectors`` calls ``add_points_to_tenant`` with empty vectors;
        the fixed handler must perform ZERO entity extractions there.
        """
        reembed = importlib.import_module("cat.core_plugins.cat_efficient_ingestion.reembed")
        original_store = reembed._store_empty_vectors

        async def store_empty_vectors(ccat, *args, **kwargs):
            handler = getattr(ccat, "vector_memory_handler", None)
            if handler is not None:
                handler._dual_mode_empty_pass = True
            try:
                return await original_store(ccat, *args, **kwargs)
            finally:
                if handler is not None:
                    handler._dual_mode_empty_pass = False

        mp.setattr(reembed, "_store_empty_vectors", store_empty_vectors)

    def _install_fake_llm(mp) -> None:
        """Give the concept phase a REAL product through the existing LLM seam.

        Under the new contract a phase with no LLM is a configuration error, so a
        green ``completed`` status requires an actual extraction product. Two
        patches, both on the plugin (the phase body itself is untouched):

        * ``GraphRAGHandler._resolve_agent_llm`` returns a sentinel object, so the
          handler resolves a usable (non-default) LLM;
        * ``StructuredLLM.run`` returns a schema-valid, NON-EMPTY product instead
          of calling a real model. The concept/relation types are taken from the
          handler's dynamic output model, so the stub stays valid against the
          configured whitelist.

        Returns nothing; the product is deterministic and known via the
        ``FAKE_*`` constants, which the graph assertions query for.
        """
        handler_module = importlib.import_module("cat.plugins.cat_graphrag.graphrag_handler")
        structured_module = importlib.import_module("cat.plugins.cat_graphrag.structured_llm")
        cls = handler_module.GraphRAGHandler
        sentinel_llm = object()

        async def resolve_agent_llm(self):
            return sentinel_llm

        async def run(self, prompt, llm, output_model, system_prompt=None):
            schema = output_model.model_json_schema()
            concept_types = schema["$defs"]["Concept"]["properties"]["type"]["enum"]
            relation_types = schema["$defs"]["Relation"]["properties"]["type"]["enum"]
            concept_type = "DEFINITION" if "DEFINITION" in concept_types else concept_types[0]
            relation_type = "CAUSES" if "CAUSES" in relation_types else relation_types[0]
            product = {
                "concepts": [
                    {"type": concept_type, "text": FAKE_CONCEPT_ALPHA},
                    {"type": concept_type, "text": FAKE_CONCEPT_BETA},
                ],
                "relations": [
                    {
                        "type": relation_type,
                        "origin": FAKE_CONCEPT_ALPHA,
                        "destination": FAKE_CONCEPT_BETA,
                        "text": FAKE_RELATION_TEXT,
                    }
                ],
            }
            model = output_model(**product)
            return structured_module.StructuredLLMResult(text=json.dumps(product), model=model)

        mp.setattr(cls, "_resolve_agent_llm", resolve_agent_llm)
        mp.setattr(structured_module.StructuredLLM, "run", run)

    async def _fake_product_counts(tenant: str) -> dict:
        """Count the exact nodes/edges the fake LLM product must have produced."""
        driver = _neo4j_driver()
        try:
            async with driver.session(database=NEO4J_DATABASE) as session:

                async def one(query, **params):
                    result = await session.run(query, tenant=tenant, **params)
                    record = await result.single()
                    await result.consume()
                    return record["c"] if record else 0

                return {
                    "concept_entities": await one(
                        "MATCH (e:Entity {tenant_id: $tenant}) WHERE e.name IN $names "
                        "RETURN count(e) AS c",
                        names=[FAKE_CONCEPT_ALPHA, FAKE_CONCEPT_BETA],
                    ),
                    "concept_relations": await one(
                        "MATCH (a:Entity {tenant_id: $tenant})-[r:RELATED_TO {text: $text}]"
                        "->(b:Entity {tenant_id: $tenant}) RETURN count(r) AS c",
                        text=FAKE_RELATION_TEXT,
                    ),
                }
        finally:
            await driver.close()

    # ---------------------------------------------------------------------------
    # App-level configuration helpers
    # ---------------------------------------------------------------------------
    def _headers(agent_key: str) -> dict:
        return {"X-Agent-ID": agent_key, "Authorization": f"Bearer {API_KEY}"}

    def _neo4j_payload() -> dict:
        return {
            "neo4j_uri": NEO4J_URI,
            "neo4j_user": NEO4J_USER,
            "neo4j_password": _neo4j_password(),
            "neo4j_database": NEO4J_DATABASE,
            "enable_derived_graph": True,
            "enable_entity_extraction": True,
            "spacy_models": {"en": "en_core_web_sm"},
            "enable_knowledge_graph": True,
            "enable_concept_relations": True,
        }

    async def _activate_plugin(lizard, agent_key: str):
        """Activate the plugin system-wide and for the agent, like an install does."""
        if PLUGIN_ID not in lizard.plugin_manager.active_plugins:
            await lizard.plugin_manager.activate_plugin(PLUGIN_ID)
        cat = await lizard.create_cheshire_cat(agent_key)
        if PLUGIN_ID not in cat.plugin_manager.active_plugins:
            await cat.plugin_manager.activate_plugin(PLUGIN_ID)
        assert PLUGIN_ID in cat.plugin_manager.active_plugins
        return cat

    async def _set_neo4j(client, agent_key: str) -> None:
        response = await client.put(
            "/vector_database/settings/Neo4jGraphRAGConfig",
            json=_neo4j_payload(),
            headers=_headers(agent_key),
        )
        assert response.status_code == 200, response.text
        assert response.json()["name"] == "Neo4jGraphRAGConfig"

    async def _set_engine(client, agent_key: str, ingestion_name: str) -> None:
        response = await client.put(
            f"/ingestion/settings/{ingestion_name}", json={}, headers=_headers(agent_key)
        )
        assert response.status_code == 200, response.text

    async def _configure_agent(client, agent_key: str, ingestion_name: str):
        lizard = client._fastapi_test_app.state.lizard
        cat = await _activate_plugin(lizard, agent_key)
        await _set_neo4j(client, agent_key)
        await _set_engine(client, agent_key, ingestion_name)
        return cat

    async def _ingest_sample(client, agent_key: str):
        with SAMPLE_FILE.open("rb") as f:
            return await client.post(
                "/rabbithole/",
                files={"file": (SAMPLE_NAME, f, SAMPLE_CONTENT_TYPE)},
                headers=_headers(agent_key),
            )

    async def _status(agent_key: str) -> dict:
        registry = importlib.import_module("cat.core_plugins.cat_efficient_ingestion.registry")
        return await registry.get_status(agent_key, "agent", SAMPLE_NAME) or {}

    # ---------------------------------------------------------------------------
    # Scenario 1 - classic mode
    # ---------------------------------------------------------------------------
    async def _test_classic_mode_builds_graph_once_and_records_no_phase(mp):
        await _assert_neo4j_reachable()
        agent_key = f"mygraph_dual_{uuid.uuid4().hex[:10]}"
        async with _app_client(mp) as app_client:
            try:
                await _configure_agent(app_client, agent_key, CORE_ENGINE)
                counts = _install_spies(mp)

                response = await _ingest_sample(app_client, agent_key)
                assert response.status_code == 200, response.text
                await _drain_recorded_handlers(SAMPLE_NAME)

                graph = await _settle_counts(agent_key)
                status = await _status(agent_key)

                assert graph["nodes"] > 0
                assert graph["relationships"] > 0
                assert graph["documents"] > 0
                assert counts["classic"] == 1, counts
                assert counts["spacy_phase"] == 0, counts
                assert counts["llm_phase"] == 0, counts
                recorded = set(status.get("completed_phases") or {})
                assert not {p for p in recorded if p.startswith("graph_")}, recorded

                print("CLASSIC", graph, counts)
            finally:
                leftover = await _wipe_tenant(agent_key)
                assert leftover == {
                    "nodes": 0,
                    "relationships": 0,
                    "documents": 0,
                    "entities": 0,
                    "source_files": 0,
                }, leftover

    # ---------------------------------------------------------------------------
    # Scenario 2 - efficient mode (+ scenario 3: one extraction per document)
    # ---------------------------------------------------------------------------
    async def _test_efficient_mode_runs_both_phases_and_never_the_classic_callback(mp):
        await _assert_neo4j_reachable()
        agent_key = f"mygraph_dual_{uuid.uuid4().hex[:10]}"
        async with _app_client(mp) as app_client:
            try:
                await _configure_agent(app_client, agent_key, EFFICIENT_ENGINE)
                counts = _install_spies(mp)
                _install_empty_pass_marker(mp)
                _install_fake_llm(mp)

                response = await _ingest_sample(app_client, agent_key)
                assert response.status_code == 200, response.text
                await _drain_recorded_handlers(SAMPLE_NAME)

                graph = await _settle_counts(agent_key)
                status = await _status(agent_key)
                recorded = status.get("completed_phases") or {}
                fake_product = await _fake_product_counts(agent_key)

                # The LLM phase really produced a product (not an empty silent
                # success): the source is COMPLETED because the concept path wrote
                # the fake concepts and their RELATED_TO edge.
                assert status.get("status") == "completed", status
                assert graph["nodes"] > 0
                assert graph["relationships"] > 0
                assert graph["documents"] > 0
                assert fake_product["concept_entities"] >= 2, fake_product
                assert fake_product["concept_relations"] >= 1, fake_product

                # scenario 2 - the phases, not the classic callback, do the work
                assert counts["classic"] == 0, counts
                assert counts["spacy_phase"] == 1, counts
                assert counts["llm_phase"] == 1, counts

                assert "graph_spacy_extraction" in recorded, recorded
                assert "graph_llm_extraction" in recorded, recorded
                for phase in ("graph_spacy_extraction", "graph_llm_extraction"):
                    marker = recorded[phase].get("marker")
                    assert isinstance(marker, str) and marker, (phase, recorded[phase])
                    assert recorded[phase].get("deps", {}).get("embedding"), recorded[phase]

                handler = (
                    await app_client._fastapi_test_app.state.lizard.get_cheshire_cat(agent_key)
                ).vector_memory_handler
                assert recorded["graph_spacy_extraction"]["marker"] == handler._spacy_fingerprint()
                assert recorded["graph_llm_extraction"]["marker"] == await handler._concept_fingerprint()

                # scenario 3 - one extraction per document on the real-vector
                # storage pass; none on the empty-vector staging pass. The
                # deterministic graph phase drains and reuses the background
                # batch, so it must NOT re-extract a source whose background write
                # already ran.
                document_count = graph["documents"]
                assert counts["empty_pass_extracts"] == 0, counts
                assert len(counts["background_extracts"]) == document_count, counts
                assert all(n == 1 for n in counts["background_extracts"].values()), counts
                assert counts["phase_rerun_extracts"] == {}, counts

                print("EFFICIENT", graph, counts, recorded)
            finally:
                leftover = await _wipe_tenant(agent_key)
                assert leftover == {
                    "nodes": 0,
                    "relationships": 0,
                    "documents": 0,
                    "entities": 0,
                    "source_files": 0,
                }, leftover

    # ---------------------------------------------------------------------------
    # Comparable counts (same corpus, both engines)
    # ---------------------------------------------------------------------------
    async def _test_both_modes_produce_comparable_graphs(mp):
        """Same corpus -> same chunk structure, comparable graph size.

        Document/chunk count and SourceFile count are identical (exact). The total
        node count is allowed to differ by at most 4 because the efficient
        ``graph_llm_extraction`` phase intentionally adds concept entities; the
        relationship count is therefore not expected to match exactly either. The
        asserted tolerance is: same documents, same SourceFile, nodes within 4, and
        the efficient graph never has fewer relationships than the classic one
        (both > 0).
        """
        await _assert_neo4j_reachable()
        agents = []
        async with _app_client(mp) as app_client:
            try:
                classic_agent = f"mygraph_dual_c_{uuid.uuid4().hex[:8]}"
                agents.append(classic_agent)
                await _configure_agent(app_client, classic_agent, CORE_ENGINE)
                _install_spies(mp)
                response = await _ingest_sample(app_client, classic_agent)
                assert response.status_code == 200, response.text
                await _drain_recorded_handlers(SAMPLE_NAME)
                classic = await _settle_counts(classic_agent)

                efficient_agent = f"mygraph_dual_e_{uuid.uuid4().hex[:8]}"
                agents.append(efficient_agent)
                await _configure_agent(app_client, efficient_agent, EFFICIENT_ENGINE)
                response = await _ingest_sample(app_client, efficient_agent)
                assert response.status_code == 200, response.text
                await _drain_recorded_handlers(SAMPLE_NAME)
                efficient = await _settle_counts(efficient_agent)

                assert classic["documents"] == efficient["documents"] > 0
                assert classic["source_files"] == efficient["source_files"] == 1
                assert abs(classic["nodes"] - efficient["nodes"]) <= 4, (classic, efficient)
                assert classic["relationships"] > 0, classic
                assert efficient["relationships"] > 0, efficient
                assert efficient["relationships"] >= classic["relationships"], (classic, efficient)

                print("COMPARE", {"classic": classic, "efficient": efficient})
            finally:
                for agent in agents:
                    leftover = await _wipe_tenant(agent)
                    assert leftover == {
                        "nodes": 0,
                        "relationships": 0,
                        "documents": 0,
                        "entities": 0,
                        "source_files": 0,
                    }, leftover

    # ---------------------------------------------------------------------------
    # Scenario 4 - EffING blocked in-process: plugin imports, classic still builds
    # ---------------------------------------------------------------------------
    def _load_blocked_plugin(package_name: str):
        """Import the real ``main.py`` with EffING masked and a counting handler."""
        import importlib.util

        package = types.ModuleType(package_name)
        package.__path__ = [str(PLUGIN_SRC)]
        package.__package__ = package_name
        sys.modules[package_name] = package

        class FakeGraphRAGHandler:
            def __init__(self):
                self.classic_calls = 0
                self.nodes = 0
                self.relationships = 0

            async def drain_source_tasks(self, source):
                # No background extraction in this fake.
                return False

            async def create_derived_graph_for_source(self, source, stored_points, cat):
                self.classic_calls += 1
                self.nodes += 2
                self.relationships += 1

        handler = types.ModuleType(package_name + ".graphrag_handler")
        handler.GraphRAGHandler = FakeGraphRAGHandler
        handler.Neo4jGraphRAGConfig = type("Neo4jGraphRAGConfig", (), {})
        sys.modules[package_name + ".graphrag_handler"] = handler

        extractor = types.ModuleType(package_name + ".entity_extractor")
        extractor.EntityExtractor = type("EntityExtractor", (), {})
        sys.modules[package_name + ".entity_extractor"] = extractor

        spec = importlib.util.spec_from_file_location(package_name + ".main", PLUGIN_SRC / "main.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name + ".main"] = module
        spec.loader.exec_module(module)
        return module, FakeGraphRAGHandler

    def _cleanup_package(package_name: str) -> None:
        for name in list(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)

    async def _test_effing_blocked_plugin_imports_and_classic_builds_without_phases(mp):
        effing = "cat.core_plugins.cat_efficient_ingestion"
        saved = sys.modules.get(effing)
        mp.setitem(sys.modules, effing, None)  # simulate absence in-process
        module, handler_cls = _load_blocked_plugin("_mygraph_dual_blocked")
        try:
            handler = handler_cls()
            cat = type("Cat", (), {"vector_memory_handler": handler, "agent_key": "blocked"})()
            await module.after_rabbithole_stored_documents.function("sample.txt", [], cat)

            assert handler.classic_calls == 1
            assert handler.nodes > 0 and handler.relationships > 0

            incoming = ["sentinel"]
            assert module.ingestion_phase_specs.function(incoming, cat) is incoming
        finally:
            _cleanup_package("_mygraph_dual_blocked")
            if saved is None:
                sys.modules.pop(effing, None)

    # ---------------------------------------------------------------------------
    # Scenario 4b - classic re-ingest after an efficient ingest (deterministic
    # regression: the empty-pass discriminator must not fire on the classic path)
    # ---------------------------------------------------------------------------
    async def _test_classic_reingest_after_efficient_ingest_builds_graph(mp):
        """EffING ingest -> switch the system engine to core -> classic re-ingest.

        The efficient ingest leaves the source's status row with
        ``phase=parsing_chunking`` and the phase-machine ``resume_owner`` lease
        (the host never clears that lease on a terminal status). The core ingest
        rewrites ``phase=parsing_chunking`` for every ingestion, so a discriminator
        keyed on ``parsing_chunking + resume_owner`` mistook the classic re-ingest
        for the empty-vector staging pass and silently skipped the graph.

        The fixed discriminator additionally requires the intrinsic empty mapping
        ``vector={}``, which the classic core never writes (it always stores the
        embedder's list). Assert the graph is rebuilt and the classic callback fires
        exactly once, while the efficient graph phases do not run again.
        """
        await _assert_neo4j_reachable()
        agent_key = f"mygraph_dual_{uuid.uuid4().hex[:10]}"
        async with _app_client(mp) as app_client:
            try:
                await _configure_agent(app_client, agent_key, EFFICIENT_ENGINE)
                counts = _install_spies(mp)

                # 1) efficient ingest: the phase machine claims the source and builds
                # the graph; the classic callback stays inert.
                response = await _ingest_sample(app_client, agent_key)
                assert response.status_code == 200, response.text
                await _drain_recorded_handlers(SAMPLE_NAME)
                first_graph = await _settle_counts(agent_key)
                first_status = await _status(agent_key)
                assert first_graph["nodes"] > 0, first_graph
                assert counts["classic"] == 0, counts
                assert counts["spacy_phase"] == 1, counts
                assert counts["llm_phase"] == 1, counts
                # bug precondition: the claim lease survives the terminal status
                assert first_status.get("resume_owner"), first_status

                # 2) switch the SYSTEM ingestion engine to the core one.
                await _set_engine(app_client, agent_key, CORE_ENGINE)

                # 3) wipe the tenant's data nodes (keep the Collection the handler
                # initialized) so any Document/entity/SourceFile after this point can
                # only come from the classic re-ingest.
                driver = _neo4j_driver()
                try:
                    async with driver.session(database=NEO4J_DATABASE) as session:
                        result = await session.run(
                            "MATCH (n {tenant_id: $tenant}) WHERE NOT n:Collection "
                            "DETACH DELETE n",
                            tenant=agent_key,
                        )
                        await result.consume()
                finally:
                    await driver.close()
                wiped = await _neo4j_counts(agent_key)
                assert wiped["documents"] == 0, wiped

                # 4) classic re-ingest of the SAME source.
                response = await _ingest_sample(app_client, agent_key)
                assert response.status_code == 200, response.text
                await _drain_recorded_handlers(SAMPLE_NAME)
                second_graph = await _settle_counts(agent_key)
                second_status = await _status(agent_key)

                assert second_graph["nodes"] > 0, second_graph
                assert second_graph["documents"] > 0, second_graph
                # the ENTITY extraction is what the stale discriminator silently
                # skipped; Documents alone would also be created by a skipped pass.
                assert second_graph["entities"] > 0, second_graph
                assert second_graph["relationships"] > 0, second_graph
                assert counts["classic"] == 1, counts
                # the classic callback is the only writer on this ingest: the phase
                # machine must not have run again.
                assert counts["spacy_phase"] == 1, counts
                assert counts["llm_phase"] == 1, counts
                # the stale claim lease is still on the row -> the old discriminator
                # would have skipped this graph build.
                assert second_status.get("resume_owner"), second_status

                print("REINGEST", {"first": first_graph, "second": second_graph}, counts, second_status)
            finally:
                leftover = await _wipe_tenant(agent_key)
                assert leftover == {
                    "nodes": 0,
                    "relationships": 0,
                    "documents": 0,
                    "entities": 0,
                    "source_files": 0,
                }, leftover

    # ---------------------------------------------------------------------------
    # Scenario 5 - cleanup receipt (runs last)
    # ---------------------------------------------------------------------------
    async def _test_zz_cleanup_receipt(mp):
        """No seeded plugin link, no leftovers in the dedicated Redis test db."""
        assert not PLUGIN_LINK.exists() and not PLUGIN_LINK.is_symlink()
        assert not any(p.name == PLUGIN_ID for p in PLUGINS_DIR.iterdir())

        from cat.db.database import get_sync_db

        db = get_sync_db()
        assert str(db.connection_pool.connection_kwargs.get("db")) == "1"
        assert db.dbsize() == 0

    # ---------------------------------------------------------------------------
    # Runner
    # ---------------------------------------------------------------------------
    def _load_effing_testkit():
        """Import the shared ``effing_testkit`` from its sibling plugin checkout."""
        candidates = (
            os.path.join(os.path.dirname(str(PLUGIN_SRC)), "cat-efficient-ingestion"),
            os.path.join(os.path.dirname(str(PLUGIN_SRC)), "cat-efficient-ingestion.my"),
        )
        for path in candidates:
            if os.path.isfile(os.path.join(path, "effing_testkit.py")):
                if path not in sys.path:
                    sys.path.insert(0, path)
                break
        import effing_testkit  # noqa: E402

        return effing_testkit

    async def _main() -> int:
        tests = [
            ("test_classic_mode_builds_graph_once_and_records_no_phase",
             _test_classic_mode_builds_graph_once_and_records_no_phase),
            ("test_efficient_mode_runs_both_phases_and_never_the_classic_callback",
             _test_efficient_mode_runs_both_phases_and_never_the_classic_callback),
            ("test_both_modes_produce_comparable_graphs",
             _test_both_modes_produce_comparable_graphs),
            ("test_effing_blocked_plugin_imports_and_classic_builds_without_phases",
             _test_effing_blocked_plugin_imports_and_classic_builds_without_phases),
            ("test_classic_reingest_after_efficient_ingest_builds_graph",
             _test_classic_reingest_after_efficient_ingest_builds_graph),
            ("test_zz_cleanup_receipt", _test_zz_cleanup_receipt),
        ]
        failed = 0
        for name, fn in tests:
            try:
                await _run_encapsulated(fn)
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001 - report and continue
                failed += 1
                traceback.print_exc()
                print(f"FAIL {name}: {exc}")
        print(f"\ntest_dual_mode: {len(tests) - failed}/{len(tests)} passed")
        return failed

    _testkit = _load_effing_testkit()
    _testkit.install_stub_package()
    try:
        _failed = asyncio.run(_main())
    finally:
        _testkit.uninstall_stub_package()

    raise SystemExit(1 if _failed else 0)
