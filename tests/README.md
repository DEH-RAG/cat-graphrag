# Standalone harnesses

This directory is the plugin's own behavioral suite, including the former
MyCAT integration contract (`test_dual_mode.py`, `test_phase_registration.py`,
`test_transfer_hook.py`). The host repository keeps no MyGRAPH tests.

The files in this directory are import-safe standalone **scripts**: each `.py`
under the plugin is imported by the Cat loader at activation, so every script
keeps its entire body (imports included) inside `if __name__ == "__main__":`
and importing it executes nothing. Run them as scripts, never by pytest, from
the host venv (it has the Cat dependencies) or inside the Cat container:

```bash
cd ../grinning-cat-core
uv run python /path/to/cat-graphrag.my/tests/test_hook_reload_order.py
```

Each script registers the plugin folder as a synthetic package (the folder name
`cat-graphrag.my` cannot be imported directly), stubs the external services it
does not need and prints `PASS <test>` lines. Requirements per script:

- `test_hook_reload_order.py`, `test_kg_master_switch.py`,
  `test_ner_segmentation.py`, `test_terminology_refresh.py`,
  `test_versioned_schema.py`, `test_reindex_swap.py`,
  `test_derived_graph_unconditional.py`, `test_phase_fingerprint.py`,
  `test_classic_path.py`, `test_embedder_injection.py`,
  `test_maintenance_imports.py`, `test_phase_run.py`,
  `test_ingestion_phase_machine.py`, `test_single_extraction.py` —
  self-contained (stubbed Neo4j/spaCy/Cat/tenant lock; the last several load
  the real plugin through a synthetic package), run from the host venv;
- `test_phase_registration.py` — loads the real plugin `main.py` and pins the
  single core hook (`rabbithole_ingestion_phases`). Run from the host venv;
- `test_phase_abc_migration.py` — the core-contract harness (real
  `cat.rabbit_hole.phases.IngestionPhase`, fake handler): applicability,
  source-scoped delete, settings/tags, process delegation, no protocol logic,
  core subclass shape, 300 s concept budget, warning-is-done;
- `test_transfer_hook.py` — pure unit test of the transfer-hook signature and
  wipe policy (the plugin's heavy modules are stubbed); one case resolves the
  ingestion factory, so run it from the host venv;
- `test_dual_mode.py` — drives the REAL core phase machine offline
  (`engine.collect_phases`, `execution_order`, `compute_tags`,
  `engine._run_phase`) with the real plugin phase classes and a fake handler:
  no Redis/Neo4j required. Run from the host venv:

  ```bash
  cd /root/git/grinning-cat-core
  .venv/bin/python /root/git/plugins/cat-graphrag.my/tests/test_dual_mode.py
  ```
- `test_cypher_comments.py` — static guard against `#` comments inside Cypher
  strings, plus a live execution of the concept-relation template when a Neo4j
  password is provided (`NEO4J_PASSWORD`/`--password`); without one it reports
  `SKIP live execution` and the static audit is the fallback;
- `test_backfill_duplicates.py` — a CLI that needs a live Neo4j: pass
  `--password` (plus `--uri`/`--user`/`--tenant` as needed);
- `test_maintenance_lizard_bootstrap.py` — needs the plugin DEPLOYED as
  `cat.plugins.cat_graphrag`, i.e. run it inside the Cat container with the
  plugin installed, not from a bare host checkout.

## Phase model

The two graph phases (`graph_spacy_extraction` / `graph_llm_extraction`) are
`cat.rabbit_hole.phases.IngestionPhase` subclasses registered through the ONE
core hook `rabbithole_ingestion_phases`. The classic ingestion callback and the
old EffING `ingestion_phase_*` hooks are gone. Covered by:

- `test_phase_abc_migration.py` — the class contract against the real core ABC;
- `test_phase_registration.py` — the single hook + exactly two classes;
- `test_phase_run.py` — the status mapping (skipped/done/not_ready/error);
- `test_classic_path.py` — one graph writer, no engine-selection switch;
- `test_embedder_injection.py` — embedder/extractor preparation for a phase;
- `test_dual_mode.py` — real core phase-machine integration;
- `test_ingestion_phase_machine.py` — the handler bodies behind the phases;
- `test_transfer_hook.py` — the vector-memory switch cleanup.

