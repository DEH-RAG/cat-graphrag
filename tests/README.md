# Standalone harnesses

The authoritative behavioral suite is the HOST repository at
`grinning-cat-core/tests/mygraph/` (pytest). The files in this directory are
import-safe standalone scripts (run them directly, not via pytest) because the
Cat plugin loader recursively imports every `.py` under the plugin.

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
- `grinning-cat-core/tests/mygraph/` — the authoritative host suite.
