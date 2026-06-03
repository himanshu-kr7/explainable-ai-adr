# Diagnostic Scripts

Investigation and sanity-check scripts used during development. These are
NOT part of the main reproducible pipeline (see the README at the repository
root), but are retained for transparency of the research process.

- `test_contamination.py` — target-leakage diagnostic; verifies the trained
  model is not trivially memorizing target edges. Supports the methodological
  rigor described in the thesis.
- `explain_leak.py` — ablation leak detector; complements the contamination
  test by interrogating a trained model for target leakage.
- `check_graph.py` — trivial graph-structure sanity check (loads the graph
  and prints its metadata).
- `explain_biology_EARLY_PROTOTYPE.py` — an early edge-occlusion explainability
  prototype, superseded by `scripts/phase3_explainability/phase3_subgraph_viz.py`.
- `plot_results.py` — one-off plotting script with hardcoded ablation
  results; regenerates summary bar charts from saved numbers.
