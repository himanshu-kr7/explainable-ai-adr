# Master's Thesis: Predicting Polypharmacy Side Effects using Heterogeneous GNNs

## 📌 Project Overview
This repository contains the complete codebase and datasets for predicting drug-drug interaction (DDI) side effects (specifically Thrombocytopenia) using Graph Neural Networks.

## 🗂️ Folder Structure
* **`/Data`**: Contains the raw biomedical datasets (CSVs) and the compiled PyTorch Geometric heterogeneous graph (`MTP_Graph.pt`).
* **`/Notebooks`**: Jupyter notebooks detailing the exploratory data analysis and graph mapping process.
* **`/Scripts`**: The core Python training engines (`train_job.py`) and our custom Edge-Occlusion Ablation Study (`explain_biology.py`).
* **`/Results`**: Training logs, the trained `.pth` model weights, and the final publication-ready explainability charts.

## 🚀 Key Methodology
Instead of relying on black-box explainers, this project utilizes a custom **Edge-Occlusion Ablation Study** to mathematically prove the biological pathways (such as Phenotypic Similarity) that the GNN uses to make its predictions.
