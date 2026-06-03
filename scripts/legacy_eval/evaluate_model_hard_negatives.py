"""
Phase 2 Evaluation: GNN Test Set Metrics under HARD-NEGATIVE protocol

This is the methodologically consistent companion to baseline_rf_hard_negatives.py.

Both models (RF baseline and GNN) are now evaluated under the SAME 
hard-negative sampling protocol:
- Positives: drug pairs causing thrombocytopenia
- Negatives: drug pairs causing OTHER side effects (not thrombocytopenia)

This produces a fair, apples-to-apples comparison for thesis reporting.
"""

import torch
import logging
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from train_job import HeteroADRModel

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def load_data_with_hard_negatives(graph_path, feature_path):
    logging.info("Loading Graph and Features...")
    data = torch.load(graph_path, map_location='cpu', weights_only=False)
    real_features = torch.load(feature_path, map_location='cpu', weights_only=False)
    data['drug'].x = real_features

    # Identify target edge
    target_edge = [et for et in data.edge_types if et[1] == 'causes_thrombocytopenia'][0]
    pos_edge_index = data[target_edge].edge_index
    logging.info(f"Target edge: {target_edge}")
    logging.info(f"Positive examples: {pos_edge_index.size(1)}")

    # =========================================================
    # Build hard negatives BEFORE purging other ADR edges
    # =========================================================
    logging.info("Sampling hard negatives from OTHER causes_* edges...")

    pos_pair_set = set(zip(pos_edge_index[0].tolist(), pos_edge_index[1].tolist()))
    pos_pair_set_with_rev = pos_pair_set | {(d, s) for s, d in pos_pair_set}

    other_causes_edges = [et for et in data.edge_types
                          if et[1].startswith('causes_') and et != target_edge]

    hard_neg_pairs = []
    needed = pos_edge_index.size(1)
    per_edge_type = max(1, needed // len(other_causes_edges))

    np.random.seed(42)
    for et in other_causes_edges:
        ei = data[et].edge_index
        n = ei.size(1)
        if n == 0:
            continue
        sample_size = min(per_edge_type, n)
        idxs = np.random.choice(n, sample_size, replace=False)
        for i in idxs:
            src, dst = ei[0, i].item(), ei[1, i].item()
            if (src, dst) not in pos_pair_set_with_rev:
                hard_neg_pairs.append((src, dst))
            if len(hard_neg_pairs) >= needed:
                break
        if len(hard_neg_pairs) >= needed:
            break

    logging.info(f"Collected {len(hard_neg_pairs)} hard negatives")
    hard_neg_pairs = hard_neg_pairs[:needed]
    neg_src = torch.tensor([p[0] for p in hard_neg_pairs], dtype=torch.long)
    neg_dst = torch.tensor([p[1] for p in hard_neg_pairs], dtype=torch.long)

    # =========================================================
    # Now purge other causes_* edges so model can load cleanly
    # =========================================================
    for et in list(data.edge_types):
        if et[1].startswith('causes_') and et != target_edge:
            del data[et]

    # =========================================================
    # Split positives into train/test (same protocol as training)
    # =========================================================
    num_pos = pos_edge_index.size(1)
    perm = torch.randperm(num_pos, generator=torch.Generator().manual_seed(42))
    test_size = int(0.1 * num_pos)
    test_idx = perm[:test_size]

    test_pos_src = pos_edge_index[0, test_idx]
    test_pos_dst = pos_edge_index[1, test_idx]

    # Sample the same number of negatives for test
    neg_test_size = min(test_size, len(hard_neg_pairs))
    test_neg_src = neg_src[:neg_test_size]
    test_neg_dst = neg_dst[:neg_test_size]

    logging.info(f"Test set: {test_size} positives + {neg_test_size} hard negatives")

    return data, target_edge, test_pos_src, test_pos_dst, test_neg_src, test_neg_dst


def evaluate_gnn():
    graph_path = "Data/MTP_Graph.pt"
    feature_path = "Data/real_drug_features.pt"
    model_weights_path = "Results/Phase2_TAG_Model.pth"

    data, target_edge, test_pos_src, test_pos_dst, test_neg_src, test_neg_dst = \
        load_data_with_hard_negatives(graph_path, feature_path)

    logging.info("Initializing GNN Architecture...")
    model = HeteroADRModel(metadata=data.metadata(), hidden_channels=128, out_channels=64)

    logging.info(f"Loading trained weights from {model_weights_path}...")
    state_dict = torch.load(model_weights_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    logging.info("✅ Weights loaded with strict=True")

    model.eval()

    # =========================================================
    # Run inference on positives + hard negatives
    # =========================================================
    logging.info("Running inference on test set (positives + hard negatives)...")
    with torch.no_grad():
        out_dict = model(data.x_dict, data.edge_index_dict)

        # Positives
        pos_src_emb = out_dict[target_edge[0]][test_pos_src]
        pos_dst_emb = out_dict[target_edge[2]][test_pos_dst]
        pos_scores = (pos_src_emb * pos_dst_emb).sum(dim=-1).sigmoid().numpy()

        # Hard negatives
        neg_src_emb = out_dict[target_edge[0]][test_neg_src]
        neg_dst_emb = out_dict[target_edge[2]][test_neg_dst]
        neg_scores = (neg_src_emb * neg_dst_emb).sum(dim=-1).sigmoid().numpy()

    predictions = np.concatenate([pos_scores, neg_scores])
    true_labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    roc_auc = roc_auc_score(true_labels, predictions)
    pr_auc = average_precision_score(true_labels, predictions)

    print("\n" + "=" * 55)
    print(" 🏆 PHASE 2: GNN TEST SET RESULTS (HARD NEGATIVES)")
    print("=" * 55)
    print(f"ROC-AUC Score : {roc_auc:.4f}")
    print(f"PR-AUC Score  : {pr_auc:.4f}")
    print("=" * 55)
    print(f"Test: {len(pos_scores)} positives + {len(neg_scores)} hard negatives")
    print("Negatives = drug pairs causing OTHER side effects.")
    print("=" * 55)


if __name__ == "__main__":
    evaluate_gnn()