# Data Setup

This repository does not ship raw data files because (1) several have license
terms requiring direct download from official sources and (2) the constructed
graph is too large for GitHub. Follow these steps to assemble the data
directory.

After completing this setup, you should have:

```
data/
├── kg.csv                          (PrimeKG triples; ~150 MB)
├── nodes.csv                       (PrimeKG node metadata; ~5 MB)
├── drug_features.csv               (PrimeKG drug attributes; ~1 MB)
├── disease_features.csv            (PrimeKG disease attributes; ~15 MB)
├── drug-mappings.tsv               (STITCH-DrugBank crosswalk; ~2 MB)
└── bio-decagon-combo.csv           (Decagon polypharmacy edges; ~75 MB)
```

After running Phase 1 (`notebooks/01-Heterogeneous-Graph-Construction.ipynb`):

```
data/MTP_Graph.pt                   (~124 MB; built from above)
```

After running Phase 2 ClinicalBERT encoder (`scripts/phase2_gnn/feature_encoder.py`):

```
data/real_drug_features.pt          (~25 MB; 7957 × 768 tensor)
```

---

## 1. PrimeKG (Chandak et al., 2023)

**Source:** [https://github.com/mims-harvard/PrimeKG](https://github.com/mims-harvard/PrimeKG)

**License:** CC BY 4.0 (free for academic and commercial use with citation).

**Files to download:**

- `kg.csv` — the full triple file
- `nodes.csv` — node metadata (IDs, types, names)
- `drug_features.csv` — drug node attributes (description, mechanism, ATC, etc.)
- `disease_features.csv` — disease node attributes (MONDO descriptions, etc.)

**How:**

```bash
# Option A: Clone the PrimeKG repo and use their data/ folder
git clone https://github.com/mims-harvard/PrimeKG.git
cp PrimeKG/data/kg.csv  data/
cp PrimeKG/data/nodes.csv  data/
cp PrimeKG/data/features/drug_features.csv  data/
cp PrimeKG/data/features/disease_features.csv  data/

# Option B: Download via Harvard Dataverse
# Follow instructions at https://github.com/mims-harvard/PrimeKG#data
```

**Citation:**

```bibtex
@article{chandak2023primekg,
  title={Building a knowledge graph to enable precision medicine},
  author={Chandak, Payal and Huang, Kexin and Zitnik, Marinka},
  journal={Scientific Data}, volume={10}, pages={67}, year={2023}
}
```

---

## 2. Decagon (Zitnik et al., 2018)

**Source:** [http://snap.stanford.edu/decagon/](http://snap.stanford.edu/decagon/)

**License:** Research use only; cite the original paper.

**File to download:**

- `bio-decagon-combo.csv` — multiplex drug–drug–side-effect triples

**How:**

Visit the Decagon dataset page and download `bio-decagon-combo.csv.gz`. Decompress and place in `data/`.

```bash
# Direct link (may change; check the dataset page for current URL)
wget http://snap.stanford.edu/decagon/bio-decagon-combo.csv.gz -O data/bio-decagon-combo.csv.gz
gunzip data/bio-decagon-combo.csv.gz
```

**Citation:**

```bibtex
@article{zitnik2018decagon,
  title={Modeling polypharmacy side effects with graph convolutional networks},
  author={Zitnik, Marinka and Agrawal, Monica and Leskovec, Jure},
  journal={Bioinformatics}, volume={34}, number={13}, pages={i457--i466}, year={2018}
}
```

---

## 3. STITCH–DrugBank Crosswalk (`drug-mappings.tsv`)

**Source:** Several reproductions exist; the one used in this work is the
crosswalk shipped with the STITCH database and re-published as
`drug-mappings.tsv` in earlier Decagon-based code releases.

**Description:** Maps STITCH chemical IDs (used by Decagon for drugs) to
DrugBank IDs (used by PrimeKG for drugs). 22,350 rows with columns including
`stitch_id`, `drugbank_id`, and chemical descriptors.

**How:**

Several Decagon-derivative repos ship this file. Search for `drug-mappings.tsv`
on GitHub or contact the maintainers of the STITCH database directly. Once
obtained, place at `data/drug-mappings.tsv`.

If you cannot locate the file, you can construct one yourself from STITCH's
chemical alias file (`chemicals.aliases.v5.0.tsv`) by filtering for DrugBank
IDs (those starting with `DB`).

---

## 4. ClinicalBERT (downloaded automatically)

The Phase 2 feature encoder downloads
[`emilyalsentzer/Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
from Hugging Face on first run. No manual setup needed beyond installing
`transformers` (in `requirements.txt`).

**Note:** The first download is ~440 MB. Make sure you have network access and
sufficient disk space when first running `feature_encoder.py`.

**Citation:**

```bibtex
@inproceedings{alsentzer2019clinicalbert,
  title={Publicly Available Clinical {BERT} Embeddings},
  author={Alsentzer, Emily and Murphy, John R. and Boag, Willie and others},
  booktitle={Clinical NLP Workshop, NAACL-HLT}, year={2019}
}
```

---

## Verifying the Setup

Once all files are in place, run:

```bash
python scripts/check_retention.py
```

If everything is correctly configured, this should report retention rates of
approximately **49.77%** for drug-level alignment and **29.77%** for
triple-level alignment between PrimeKG and Decagon via the crosswalk.

If your numbers differ, double-check that the files have not been truncated
during download and that you have the same versions cited above (PrimeKG
v2.0+, Decagon original release).

---

## Storage Space Required

Approximate disk usage after full setup:

| Stage                              | Size    |
|------------------------------------|---------|
| Raw inputs (PrimeKG + Decagon)     | ~250 MB |
| Phase 1 output (`MTP_Graph.pt`)    | ~124 MB |
| Phase 2 features (`real_drug_features.pt`) | ~25 MB  |
| Trained models (4 ADRs × ~14.5 MB) | ~58 MB  |
| Visualizations + logs              | ~5 MB   |
| **Total**                          | **~460 MB** |

Plus the ClinicalBERT model (~440 MB) cached in `~/.cache/huggingface/`.
