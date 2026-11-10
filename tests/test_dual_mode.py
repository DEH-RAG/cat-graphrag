#!/usr/bin/env python3
"""MyGRAPH graph phases integrate with the CORE phase machine.

The old classic-vs-efficient dual mode is gone: every source goes through the
core RabbitHole, which collects the built-in phases plus the plugin phases from
``rabbithole_ingestion_phases``, orders them by dependency, tags them from their
``settings(cat)`` (folded with the dependencies' tags) and runs each one until
it is ``done`` / ``skipped`` / ``error`` (``not_ready`` is retried within the
phase budget).

This harness drives those real core functions (``engine.collect_phases``,
``engine.execution_order``, ``phases.compute_tags``, ``engine._run_phase``)
with the real plugin phase classes and a fake handler, so no Redis/Neo4j/spaCy
is required.

Standalone script (plugin rules): the host imports every ``.py`` under the
plugin folder, so the ENTIRE body lives inside ``if __name__ == "__main__":``
and importing this file executes zero code. Run it as a plain script from the
host checkout:

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-graphrag.my/tests/test_dual_mode.py
"""

if __name__ == "__main__":
    import asyncio
    import os
    import sys
    import traceback
    import types
    from pathlib import Path

    PLUGIN_DIR = Path("/root/git/plugins/cat-graphrag.my")

    os.environ.setdefault("CAT_CRYPTO_KEY", "graphrag-test-crypto-key")
    os.environ.setdefault("CAT_CRYPTO_SALT", "graphrag-test-crypto-salt")

    from cat.rabbit_hole import engine
    from cat.rabbit_hole.phases import (
        IngestionPhase,
        IngestionSource,
        PhaseResult,
        compute_tags,
        execution_order,
    )

    class _DependencyPhase(IngestionPhase):
        """Offline stand-in for the built-in parsing/embedding phases."""

        def __init__(self, phase_id):
            self.id = phase_id

        async def process(self, source, cat):
            return PhaseResult(status="done")

        async def delete(self, source, cat):
            return None

    def _builtins():
        return [
            _DependencyPhase("parsing_chunking"),
            _DependencyPhase("embedding"),
        ]

    def _load_phases(package_name):
        package = types.ModuleType(package_name)
        package.__path__ = [str(PLUGIN_DIR)]
        package.__package__ = package_name
        sys.modules[package_name] = package

        class GraphRAGHandler:
            pass

        handler = types.ModuleType(package_name + ".graphrag_handler")
        setattr(handler, "GraphRAGHandler", GraphRAGHandler)
        sys.modules[package_name + ".graphrag_handler"] = handler

        return __import__(package_name + ".phases", fromlist=["phases"])

    phases = _load_phases("_mygraph_dual_core")
    _StubHandler = sys.modules["_mygraph_dual_core.graphrag_handler"].GraphRAGHandler

    class FakeHandler(_StubHandler):
        def __init__(self, spacy="spacy-v1", concept="concept-v1"):
            self.spacy = spacy
            self.concept = concept

        def _spacy_fingerprint(self):
            return self.spacy

        async def _concept_fingerprint(self):
            return self.concept

    def _cat(handler=None):
        class _PluginManager:
            hooks = {}

            async def execute_hook(self, name, *args, caller=None):
                assert name == "rabbithole_ingestion_phases"
                incoming = args[0]
                incoming.append(phases.GraphSpacyPhase)
                incoming.append(phases.GraphConceptPhase)
                return incoming

        return types.SimpleNamespace(
            plugin_manager=_PluginManager(),
            vector_memory_handler=handler or FakeHandler(),
        )

    def _source(name="file.pdf"):
        return IngestionSource("agent_test", None, name)

    async def _test_collect_phases_orders_graph_after_dependencies():
        rabbit_hole = types.SimpleNamespace()
        phases_ = await engine.collect_phases(rabbit_hole, _cat())

        ids = [phase.id for phase in phases_]
        assert ids == [
            "parsing_chunking",
            "embedding",
            "graph_spacy_extraction",
            "graph_llm_extraction",
        ], ids

    async def _test_plugin_contributes_both_core_phases():
        phases_ = await engine.collect_phases(types.SimpleNamespace(), _cat())
        graph = {
            phase.id: phase
            for phase in phases_
            if phase.id.startswith("graph_")
        }
        assert set(graph) == {"graph_spacy_extraction", "graph_llm_extraction"}
        for phase in graph.values():
            assert isinstance(phase, IngestionPhase), phase
            assert phase.depends_on == ("parsing_chunking", "embedding"), (
                phase.depends_on
            )

    async def _test_settings_change_moves_the_phase_tag():
        cat = _cat(FakeHandler())
        ordered = execution_order(
            [
                *_builtins(),
                phases.GraphSpacyPhase(),
                phases.GraphConceptPhase(),
            ]
        )
        first = await compute_tags(ordered, cat)
        again = await compute_tags(ordered, cat)
        assert first == again, "the tag must be stable while settings are unchanged"

        cat.vector_memory_handler.spacy = "spacy-v2"
        changed = await compute_tags(ordered, cat)
        assert changed["graph_spacy_extraction"] != first["graph_spacy_extraction"]
        assert changed["graph_llm_extraction"] == first["graph_llm_extraction"]

    async def _test_execution_order_is_stable_and_dependency_first():
        spacy = phases.GraphSpacyPhase()
        concept = phases.GraphConceptPhase()
        ordered = execution_order([*_builtins(), concept, spacy])
        assert [phase.id for phase in ordered] == [
            "parsing_chunking",
            "embedding",
            "graph_llm_extraction",
            "graph_spacy_extraction",
        ]

    async def _test_not_ready_is_retried_then_error_after_budget():
        class _NeverReady(IngestionPhase):
            id = "graph_never_ready"
            not_ready_budget_seconds = 0.05

            async def process(self, source, cat):
                return PhaseResult(
                    status="not_ready", reason="points missing", retry_after=0.01
                )

            async def delete(self, source, cat):
                return None

        result = await engine._run_phase(_NeverReady(), _source(), _cat())
        assert result.status == "error", result
        assert "Not ready after" in (result.reason or ""), result

    async def _test_exception_is_an_error_with_its_reason():
        class _Boom(IngestionPhase):
            id = "graph_boom"

            async def process(self, source, cat):
                raise RuntimeError("llm exploded")

            async def delete(self, source, cat):
                return None

        result = await engine._run_phase(_Boom(), _source(), _cat())
        assert result.status == "error", result
        assert result.reason == "llm exploded", result

    async def _main() -> int:
        tests = [
            ("test_collect_phases_orders_graph_after_dependencies",
             _test_collect_phases_orders_graph_after_dependencies),
            ("test_plugin_contributes_both_core_phases",
             _test_plugin_contributes_both_core_phases),
            ("test_settings_change_moves_the_phase_tag",
             _test_settings_change_moves_the_phase_tag),
            ("test_execution_order_is_stable_and_dependency_first",
             _test_execution_order_is_stable_and_dependency_first),
            ("test_not_ready_is_retried_then_error_after_budget",
             _test_not_ready_is_retried_then_error_after_budget),
            ("test_exception_is_an_error_with_its_reason",
             _test_exception_is_an_error_with_its_reason),
        ]
        failed = 0
        for name, fn in tests:
            try:
                await fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001 - report and continue
                failed += 1
                traceback.print_exc()
                print(f"FAIL {name}: {exc}")
        print(f"\ntest_dual_mode: {len(tests) - failed}/{len(tests)} passed")
        return failed

    raise SystemExit(1 if asyncio.run(_main()) else 0)
