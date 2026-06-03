# Explainable ADR Prediction in Polypharmacy

**Author:** Himanshu Kumar
**Affiliation:** Department of Computer Science and Engineering, IIT Patna
**Supervisor:** Dr. Joydeep Chandra
**Date:** May 2026

> MTech thesis project on explainable adverse drug reaction (ADR) prediction in polypharmacy settings, using a heterogeneous biomedical knowledge graph fusing PrimeKG and Decagon, a text-attributed heterogeneous GNN with ClinicalBERT drug embeddings, and counterfactual edge-occlusion explainability.

---

## Headline Result

The trained TAG-GNN models achieve strong performance under the conventional random-negative evaluation protocol (mean ROC-AUC **0.9922** across four ADRs), matching or exceeding Decagon and recent extensions. However, under a more rigorous **hard-negative protocol** in which negatives are drawn from drug pairs causing *other* adverse reactions, the same models collapse to mean ROC-AUC **0.5087** — indistinguishable from random.

| ADR                       | Positives | RND-ROC | RND-PR | HRD-ROC | HRD-PR |
|---------------------------|-----------|---------|--------|---------|--------|
| Thrombocytopenia          | 4,099     | 0.9937  | 0.9913 | 0.5166  | 0.5109 |
| Bleeding                  | 3,697     | 0.9860  | 0.9785 | 0.4970  | 0.4951 |
| Cardiac decompensation    | 4,764     | 0.9957  | 0.9934 | 0.5213  | 0.5295 |
| Kidney failure            | 5,050     | 0.9932  | 0.9901 | 0.4999  | 0.5063 |

For thrombocytopenia, a ClinicalBERT-only Random Forest baseline achieves ROC-AUC **0.81** under hard negatives — outperforming the GNN by 29 percentage points. We interpret this as evidence that the polypharmacy ADR prediction literature may be evaluating against a deceptively easy task, and we release this code as a foundation for future work using hard-negative training protocols.

For full context, see the thesis (`thesis/thesis.pdf`).

---

## Pipeline Overview

Three sequential phases:

1. **Phase 1 — Graph Construction.** Fuses PrimeKG (structural biology) and Decagon (polypharmacy ADRs) via STITCH–DrugBank entity alignment. Result: 90,067 nodes / 5.7M edges / 10 node types / 1,359 relation types.
2. **Phase 2 — TAG-GNN Training.** Two-layer `HeteroConv` (SAGEConv per relation) encoder + parameter-free dot-product decoder. Drug nodes initialized with 768-d ClinicalBERT [CLS] embeddings of description + mechanism-of-action text.
3. **Phase 3 — Counterfactual Explainability.** For each predicted drug-pair–ADR link, extracts a 2-hop local subgraph, performs leave-one-out edge occlusion, ranks edges by counterfactual logit drop, and renders a named subgraph visualization.

---

## Repository Structure

```
.
├── notebooks/              Phase 1 graph construction notebook
├── scripts/                All Python code
│   ├── phase1_baselines/   Random Forest baselines (random + hard negative)
│   ├── phase2_gnn/         Feature encoder, training, consolidated evaluation
│   ├── phase3_explainability/  Subgraph visualization with edge occlusion
│   ├── diagnostics/        Leakage/contamination tests and sanity checks (research artifacts)
│   └── legacy_eval/        Single-ADR evaluators superseded by evaluate_all_adrs.py
├── hpc/                    SLURM batch scripts for training
├── results/                Evaluation outputs and case-study visualizations              
└── docs/                   Data-setup instructions
```

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU recommended for training (CPU works for inference and explainability)
- ~16 GB RAM minimum for graph construction

### Installation

```bash
git clone https://github.com/<your-username>/explainable-ai-adr.git
cd explainable-ai-adr
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data Setup

This repository does **not** include the underlying datasets, because (a) PrimeKG and Decagon have their own license terms and citation requirements, and (b) the constructed graph (`MTP_Graph.pt`) is too large to host on GitHub.

Follow [`docs/DATA_SETUP.md`](docs/DATA_SETUP.md) to obtain:

- **PrimeKG** — `kg.csv` and `nodes.csv` from [`harvardlcs/PrimeKG`](https://github.com/mims-harvard/PrimeKG)
- **Decagon** — `bio-decagon-combo.csv` from [SNAP/Stanford](http://snap.stanford.edu/decagon/)
- **Drug crosswalk** — `drug-mappings.tsv` (STITCH ↔ DrugBank IDs)
- **PrimeKG drug/disease feature tables** — `drug_features.csv`, `disease_features.csv`

Place them in `data/` at the repository root before running the pipeline.

---

## Reproducing the Results

### Phase 1: Build the unified graph

```bash
jupyter notebook notebooks/01-Heterogeneous-Graph-Construction.ipynb
```

Runs end-to-end. Produces `data/MTP_Graph.pt` (~124 MB).

### Phase 2: Encode drug text features with ClinicalBERT

```bash
python scripts/phase2_gnn/feature_encoder.py
```

Produces `data/real_drug_features.pt` (7,957 × 768 tensor; ~25 MB).

### Phase 2: Train models for each target ADR

On HPC (preferred):

```bash
cd hpc/
for ADR in thrombocytopenia Bleeding Cardiacdecompensation kidneyfailure; do
    sbatch --export=ALL,ADR=$ADR train_adr.sbatch
done
```

Local single-target training:

```bash
python scripts/phase2_gnn/train_job.py --target_adr Bleeding
```

Each training run takes ~2 minutes on an NVIDIA A100 (or ~10–15 minutes on CPU). Trained model is saved to `results/Phase2_TAG_Model_<ADR>.pth`.

### Phase 2: Consolidated evaluation across all four ADRs

```bash
python scripts/phase2_gnn/evaluate_all_adrs.py
```
Prints a consolidated table to standard output with both random-negative and hard-negative ROC-AUC and PR-AUC for all four target ADRs. Redirect to a file with `| tee results/multi_adr_results.txt` if you want to save it.

### Phase 3: Generate counterfactual subgraph visualizations

```bash
python scripts/phase3_explainability/phase3_subgraph_viz.py \
    --target_adr thrombocytopenia --hops 2 --top_k 5
```

Repeat for `Bleeding`, `Cardiacdecompensation`, and `kidneyfailure`. Produces named subgraph PNGs in `results/`.

---

## Reproducibility Notes

The evaluation script in this release fixes two subtle bugs that affect any reproduction of the trained models:

1. **Edge-type purging.** The training script removes all `causes_*` edge types except the target before training, leaving 51 edge types (1 target + 50 structural). The evaluation and explainability scripts must reconstruct the model with this same purged metadata, otherwise the encoder has hundreds of randomly initialized relation layers whose noise overwhelms the prediction signal.

2. **Lazy module materialization.** PyTorch Geometric's `Linear(-1, h)` layers are lazy-initialized. When loading a checkpoint into a model with lazy modules, you must first run a dummy forward pass to materialize the parameters, or the loaded weights are silently discarded into placeholder shadow state.

Both fixes are documented inline in `evaluate_all_adrs.py` and `phase3_subgraph_viz.py`. If you implement your own evaluation, follow the same pattern.

---

## Limitations

This work is honest about what it does and does not show. Briefly:

- The trained models do not reliably discriminate between distinct ADRs under hard-negative evaluation. Random-negative performance numbers should be interpreted in this context.
- The Random Forest baseline was evaluated only for thrombocytopenia; extension to other ADRs is left for future work.
- The explainability framework is best positioned as a model-auditing and hypothesis-generation tool, not as a clinical decision-support system.
- All evaluation is retrospective; no prospective patient validation has been performed.

See Chapter 5 of the thesis for a fuller discussion.

---

## Citation

If you use this code, please cite:

```bibtex
@mastersthesis{kumar2026explainable,
  author  = {Kumar, Himanshu},
  title   = {Explainable Adverse Drug Reaction Prediction in
             Polypharmacy using Heterogeneous Biomedical Knowledge Graphs},
  school  = {Indian Institute of Technology Patna},
  year    = {2026},
  type    = {MTech thesis},
  address = {Patna, India}
}
```

And cite the foundational datasets:

```bibtex
@article{zitnik2018decagon,
  title={Modeling polypharmacy side effects with graph convolutional networks},
  author={Zitnik, Marinka and Agrawal, Monica and Leskovec, Jure},
  journal={Bioinformatics}, volume={34}, number={13}, pages={i457--i466}, year={2018}
}

@article{chandak2023primekg,
  title={Building a knowledge graph to enable precision medicine},
  author={Chandak, Payal and Huang, Kexin and Zitnik, Marinka},
  journal={Scientific Data}, volume={10}, pages={67}, year={2023}
}

@inproceedings{alsentzer2019clinicalbert,
  title={Publicly Available Clinical {BERT} Embeddings},
  author={Alsentzer, Emily and Murphy, John R. and Boag, Willie and others},
  booktitle={Clinical NLP Workshop, NAACL-HLT}, year={2019}
}
```

---

## Acknowledgments

This work was conducted as the MTech thesis of Himanshu Kumar at IIT Patna under the supervision of Dr. Joydeep Chandra. Training was performed on the Param Rudra HPC cluster at IIT Patna.

---

## License

Code released under the MIT License. See [`LICENSE`](LICENSE).

Note that this license applies only to the code in this repository. The underlying datasets (PrimeKG, Decagon, MIMIC-III) have their own licenses; users must comply with those terms separately.
