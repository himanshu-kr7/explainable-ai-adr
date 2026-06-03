# Legacy Evaluation Scripts

Single-purpose evaluators superseded by the consolidated
`scripts/phase2_gnn/evaluate_all_adrs.py`, which evaluates all four target
ADRs under both random-negative and hard-negative protocols in a single run.

- `evaluate_model.py` — single-ADR random-negative evaluation.
- `evaluate_model_hard_negatives.py` — single-ADR hard-negative evaluation.
- `evaluate_structural_model.py` — evaluates the early structural-only model
  (`MTP_Model_Param.pth`), which predates the text-attributed (ClinicalBERT)
  approach used in the final pipeline.

Retained as simpler, single-purpose reference implementations and for research
transparency. For current evaluation, use `evaluate_all_adrs.py`.
