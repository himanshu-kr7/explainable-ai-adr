"""
Phase 2 Evaluation: GNN Test Set Metrics (STRICT MODE)
This script calculates the final Test ROC-AUC and PR-AUC.
All 'strict=False' fallbacks and undirected graph mismatches have been removed.
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

def load_and_split_data(graph_path, feature_path):
    logging.info("Loading Graph Topology...")
    data = torch.load(graph_path, map_location='cpu', weights_only=False)
    
    logging.info("Injecting ClinicalBERT NLP Features...")
    real_features = torch.load(feature_path, map_location='cpu', weights_only=False)
    data['drug'].x = real_features
    
    # Isolate the target edge type
    target_edge = [et for et in data.edge_types if et[1].startswith('causes_')][0]
    logging.info(f"Target Edge: {target_edge}")
    
    # >>> NEW: Purge all other ADR edges to match training-time graph structure <
    edges_to_delete = [et for et in list(data.edge_types) 
                       if et[1].startswith('causes_') and et != target_edge]
    for et in edges_to_delete:
        del data[et]
    logging.info(f"Purged {len(edges_to_delete)} non-target ADR edge types. Remaining edge types: {len(data.edge_types)}")
    
    # EXACT split used in training. is_undirected=False is crucial for biological relations
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
    feature_path = "Data/real_drug_features.pt"
    model_weights_path = "Results/Phase2_TAG_Model.pth"
    
    # 1. Prepare Data
    test_data, target_edge = load_and_split_data(graph_path, feature_path)
    
    # 2. Initialize Model Skeleton
    logging.info("Initializing GNN Architecture...")
    model = HeteroADRModel(metadata=test_data.metadata(), hidden_channels=128, out_channels=64)
    
    # 3. Load the Supercomputer Weights (STRICT MODE ENFORCED)
    logging.info(f"Loading trained weights from {model_weights_path}...")
    state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'), weights_only=False)
    
    # FIXED: We removed the try-except fallback. It must load perfectly, or crash loudly.
    model.load_state_dict(state_dict, strict=True)
    logging.info("✅ Perfect Match: Weights loaded with 100% strictness. No random layers.")
    
    model.eval() # CRITICAL: Set model to evaluation mode (turns off dropout)
    
    # 4. Run Inference
    logging.info("Running final Test Set predictions...")
    with torch.no_grad():
        # Pass the data through the model to get embeddings
        out_dict = model(test_data.x_dict, test_data.edge_index_dict)
        
        src_nodes = test_data[target_edge].edge_label_index[0]
        dst_nodes = test_data[target_edge].edge_label_index[1]
        
        src_embeddings = out_dict[target_edge[0]][src_nodes]
        dst_embeddings = out_dict[target_edge[2]][dst_nodes]
        
        # Determine if the model uses a custom decode method or pure dot product
        if hasattr(model, 'decode'):
            logging.info("Using Model's built-in decoder...")
            predictions = model.decode(src_embeddings, dst_embeddings).sigmoid().numpy()
        else:
            logging.info("Using standard Dot Product decoder...")
            predictions = (src_embeddings * dst_embeddings).sum(dim=-1).sigmoid().numpy()
            
        true_labels = test_data[target_edge].edge_label.numpy()
        
    # 5. Calculate Metrics
    roc_auc = roc_auc_score(true_labels, predictions)
    pr_auc = average_precision_score(true_labels, predictions)
    
    print("\n" + "="*50)
    print(" 🏆 PHASE 2: GNN FINAL TEST SET RESULTS (CORRECTED)")
    print("="*50)
    print(f"ROC-AUC Score : {roc_auc:.4f}")
    print(f"PR-AUC Score  : {pr_auc:.4f}")
    print("="*50)

if __name__ == "__main__":
    evaluate_gnn()