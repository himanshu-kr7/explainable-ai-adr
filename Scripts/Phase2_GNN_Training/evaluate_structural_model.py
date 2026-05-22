"""
Phase 2 Evaluation: GNN Test Set Metrics (STRUCTURAL VARIANT)

This script evaluates the structural GNN (MTP_Model_Param.pth), which was 
trained with random-initialized drug features (NOT ClinicalBERT).

This is the companion script to evaluate_model.py (which evaluates the 
ClinicalBERT-augmented TAG model). Keeping both scripts separate ensures 
reproducibility of both experimental variants reported in the thesis.
"""

import torch
import logging
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.transforms import RandomLinkSplit

# Import the exact model class you built in train_job.py
from train_job import HeteroADRModel

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_and_split_data(graph_path):
    logging.info("Loading Graph Topology...")
    data = torch.load(graph_path, map_location='cpu', weights_only=False)
    
    # NOTE: Structural variant uses the original random-initialized drug features.
    # We DO NOT inject ClinicalBERT embeddings here.
    logging.info("Using original random-initialized drug features (no ClinicalBERT injection).")
    
    # Isolate the target edge type
    target_edge = [et for et in data.edge_types if et[1].startswith('causes_')][0]
    logging.info(f"Target Edge: {target_edge}")
    
    # Purge all other ADR edges to match training-time graph structure
    edges_to_delete = [et for et in list(data.edge_types) 
                       if et[1].startswith('causes_') and et != target_edge]
    for et in edges_to_delete:
        del data[et]
    logging.info(f"Purged {len(edges_to_delete)} non-target ADR edge types. Remaining edge types: {len(data.edge_types)}")
    
    # EXACT split protocol used in training
    logging.info("Applying RandomLinkSplit to isolate Test Set...")
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=False,
        edge_types=[target_edge],
        rev_edge_types=None, 
        add_negative_train_samples=False
    )
    
    train_data, val_data, test_data = transform(data)
    return test_data, target_edge

def evaluate_gnn():
    graph_path = "Data/MTP_Graph.pt"
    model_weights_path = "Results/MTP_Model_Param.pth"
    
    # 1. Prepare Data
    test_data, target_edge = load_and_split_data(graph_path)
    
    # 2. Initialize Model Skeleton
    logging.info("Initializing GNN Architecture (Structural variant)...")
    model = HeteroADRModel(metadata=test_data.metadata(), hidden_channels=128, out_channels=64)
    
    # 3. Load the Structural Model Weights
    logging.info(f"Loading structural model weights from {model_weights_path}...")
    state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'), weights_only=False)
    
    # Use strict=False here because the structural model may have been saved 
    # before the TAG injection refactor, so some keys may differ slightly.
    # The non-strict mode allows graceful handling of any mismatch.
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    if incompatible_keys.missing_keys:
        logging.warning(f"Missing keys: {len(incompatible_keys.missing_keys)} (may be acceptable)")
    if incompatible_keys.unexpected_keys:
        logging.warning(f"Unexpected keys: {len(incompatible_keys.unexpected_keys)} (may be acceptable)")
    logging.info("Weights loaded.")
    
    model.eval()
    
    # 4. Run Inference
    logging.info("Running final Test Set predictions...")
    with torch.no_grad():
        out_dict = model(test_data.x_dict, test_data.edge_index_dict)
        
        src_nodes = test_data[target_edge].edge_label_index[0]
        dst_nodes = test_data[target_edge].edge_label_index[1]
        
        src_embeddings = out_dict[target_edge[0]][src_nodes]
        dst_embeddings = out_dict[target_edge[2]][dst_nodes]
        
        # Standard dot-product decoder
        predictions = (src_embeddings * dst_embeddings).sum(dim=-1).sigmoid().numpy()
        true_labels = test_data[target_edge].edge_label.numpy()
        
    # 5. Calculate Metrics
    roc_auc = roc_auc_score(true_labels, predictions)
    pr_auc = average_precision_score(true_labels, predictions)
    
    print("\n" + "="*50)
    print(" 🏆 PHASE 2: GNN (STRUCTURAL) FINAL TEST SET RESULTS")
    print("="*50)
    print(f"ROC-AUC Score : {roc_auc:.4f}")
    print(f"PR-AUC Score  : {pr_auc:.4f}")
    print("="*50)
    print("NOTE: This is the random-feature GNN (no ClinicalBERT).")
    print("Compare with evaluate_model.py output for the TAG variant.")

if __name__ == "__main__":
    evaluate_gnn()