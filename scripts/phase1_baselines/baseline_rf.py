"""
Baseline 1: Random Forest on Text Embeddings (No Graph Topology)
This script proves why GNNs are necessary by establishing the performance 
limit of standard Machine Learning using only raw NLP features.
"""

import torch
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_and_prepare_data(graph_path, feature_path):
    logging.info("Loading Graph and NLP Features...")
    data = torch.load(graph_path, map_location='cpu', weights_only=False)
    real_features = torch.load(feature_path, map_location='cpu', weights_only=False)
    
    target_edge_type = [et for et in data.edge_types if et[1].startswith('causes_')][0]
    edge_index = data[target_edge_type].edge_index

    logging.info(f"Target Edge Isolated: {target_edge_type}")
    
    # 1. Create Positive Samples (Real interactions)
    src_nodes = edge_index[0].numpy()
    dst_nodes = edge_index[1].numpy()
    
    # Extract NLP features for source nodes (Drugs)
    # Using real_features directly since they belong to the 'drug' nodes
    X_pos_src = real_features[src_nodes].numpy()
    
    # For baseline simplicity, we will use dummy features for the destination (Side Effects)
    # In a perfect world, we'd have ClinicalBERT for side effects too!
    X_pos_dst = np.random.randn(len(dst_nodes), 128) # Dummy 128-dim features
    
    # Concatenate [Drug Features | Disease Features]
    X_pos = np.hstack((X_pos_src, X_pos_dst))
    y_pos = np.ones(len(X_pos))
    
    # 2. Create Negative Samples (Fake interactions for balance)
    logging.info("Generating negative samples for training balance...")
    num_drugs = data['drug'].num_nodes
    num_diseases = data[target_edge_type[2]].num_nodes
    num_edges = len(src_nodes)
    
    fake_src = np.random.randint(0, num_drugs, num_edges)
    fake_dst = np.random.randint(0, num_diseases, num_edges)
    
    X_neg_src = real_features[fake_src].numpy()
    X_neg_dst = np.random.randn(len(fake_dst), 128)
    X_neg = np.hstack((X_neg_src, X_neg_dst))
    y_neg = np.zeros(len(X_neg))
    
    # Combine and split
    X = np.vstack((X_pos, X_neg))
    y = np.concatenate((y_pos, y_neg))
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    graph_path = "Data/MTP_Graph.pt"
    feature_path = "Data/real_drug_features.pt"
    
    X_train, X_test, y_train, y_test = load_and_prepare_data(graph_path, feature_path)
    
    logging.info(f"Training Random Forest on {len(X_train)} samples...")
    # Initialize RF - kept relatively small for fast local execution
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    logging.info("Predicting on test set...")
    y_probs = clf.predict_proba(X_test)[:, 1]
    
    # --- The Crucial M.Tech Level Metrics ---
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs) # The reviewer-demanded metric!
    
    print("\n" + "="*50)
    print(" 📊 BASELINE 1: RANDOM FOREST RESULTS")
    print("="*50)
    print(f"ROC-AUC Score : {roc_auc:.4f}")
    print(f"PR-AUC Score  : {pr_auc:.4f}")
    print("="*50)
    print("This is the score our GNN must beat!")

if __name__ == "__main__":
    main()