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
  `test_single_extraction.py` —
  self-contained (stubbed Neo4j/spaCy/Cat/tenant lock; the last seven load the
  real plugin through a synthetic package), run from the host venv;
- `test_phase_registration.py` — loads the real plugin `main.py` and pins the
  canonical EffING `PhaseSpec` identity; the EffING contract comes from the
  shared `effing_testkit` (`install_stub_package`). Run from the host venv;
- `test_transfer_hook.py` — pure unit test of the transfer-hook signature and
  wipe policy (the plugin's heavy modules are stubbed); one case resolves the
  ingestion factory, so run it from the host venv;
- `test_dual_mode.py` — the full host-app integration contract (classic vs
  efficient engine on the real `create_app`, in-memory Qdrant, dedicated
  `CAT_REDIS_DB=1` and a live Neo4j at `bolt://127.0.0.1:7687`). It needs
  `CAT_REDIS_HOST` (and `CAT_NEO4J_PASSWORD`/`NEO4J_PASSWORD` when the local
  Neo4j container is not inspectable) and runs from the host venv:

  ```bash
  cd /root/git/grinning-cat-core
  CAT_REDIS_HOST=<redis host> .venv/bin/python /root/git/plugins/cat-graphrag.my/tests/test_dual_mode.py
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

## Retired: `tests/test_ingestion_phase_machine.py`

Removed. It documented the obsolete single-phase `graph_building` model
(`ingestion_phase_pending` + a diary/generation check in
`before_ingestion_status_completed`) and imported the pre-extraction module
path (the legacy `cat.plugins.*` namespace).

The current two-phase model (`graph_spacy_extraction` /
`graph_llm_extraction`, registered through EffING's `ingestion_phase_specs`)
is covered by:

- `../test_ingestion_phase_machine.py` — the maintained standalone harness at
  the plugin root (registration, markers, tri-state `ingestion_phase_run`,
  completion gate, mode truth table, transfer cleanup);
- the plugin suites in this directory (`test_phase_run.py`,
  `test_classic_path.py`, `test_embedder_injection.py`,
  `test_single_extraction.py`, `test_maintenance_imports.py`);
- `test_dual_mode.py`, `test_phase_registration.py`, `test_transfer_hook.py`
  — the former MyCAT integration contract, now owned here.
