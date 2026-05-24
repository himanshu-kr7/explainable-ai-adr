"""
Baseline 1 (HARD NEGATIVES): Random Forest on Text Embeddings
=============================================================
This is the methodologically improved version of baseline_rf.py.

Original problem: random negative sampling produces drug pairs that
don't interact at all, making the task trivially easy (just detect
"is this a real interaction"). This inflates the baseline to ~0.99.

Fix: Use HARD NEGATIVES — drug pairs that DO interact but cause a
DIFFERENT side effect, not the target (thrombocytopenia). This
forces the model to actually learn what's specific to thrombocytopenia.

This produces a more honest, defensible baseline for thesis reporting.
"""

import torch
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def load_and_prepare_data(graph_path, feature_path):
    logging.info("Loading Graph and NLP Features...")
    data = torch.load(graph_path, map_location='cpu', weights_only=False)
    real_features = torch.load(feature_path, map_location='cpu', weights_only=False)

    # Identify target edge type
    target_edge_type = [et for et in data.edge_types if et[1] == 'causes_thrombocytopenia'][0]
    target_edge_index = data[target_edge_type].edge_index
    logging.info(f"Target Edge: {target_edge_type}")
    logging.info(f"Positive examples: {target_edge_index.size(1)}")

    # ========================================================
    # POSITIVES: real drug pairs that cause thrombocytopenia
    # ========================================================
    pos_src = target_edge_index[0].numpy()
    pos_dst = target_edge_index[1].numpy()

    X_pos_src = real_features[pos_src].numpy()
    X_pos_dst = real_features[pos_dst].numpy()
    X_pos = np.hstack((X_pos_src, X_pos_dst))
    y_pos = np.ones(len(X_pos))

    # ========================================================
    # HARD NEGATIVES: drug pairs that interact (cause SOME side effect)
    # but NOT thrombocytopenia specifically
    # ========================================================
    logging.info("Collecting hard negatives from OTHER causes_* edges...")

    # Build set of positive pairs to exclude from negatives
    pos_pair_set = set(zip(pos_src.tolist(), pos_dst.tolist()))
    # Also exclude reverse direction
    pos_pair_set_with_rev = pos_pair_set | {(d, s) for s, d in pos_pair_set}

    # Collect drug pairs from all OTHER causes_* edges
    hard_neg_pairs = []
    other_causes_edges = [et for et in data.edge_types
                          if et[1].startswith('causes_') and et != target_edge_type]
    logging.info(f"Sampling from {len(other_causes_edges)} other side-effect edge types...")

    # Sample uniformly across all other side effects
    needed = len(X_pos)  # match positive count for balance
    per_edge_type = max(1, needed // len(other_causes_edges))

    for et in other_causes_edges:
        ei = data[et].edge_index
        n = ei.size(1)
        if n == 0:
            continue
        # Sample some pairs from this edge type
        sample_size = min(per_edge_type, n)
        idxs = np.random.choice(n, sample_size, replace=False)
        for i in idxs:
            src, dst = ei[0, i].item(), ei[1, i].item()
            # Exclude any pair that's actually a positive
            if (src, dst) not in pos_pair_set_with_rev:
                hard_neg_pairs.append((src, dst))
            if len(hard_neg_pairs) >= needed:
                break
        if len(hard_neg_pairs) >= needed:
            break

    logging.info(f"Collected {len(hard_neg_pairs)} hard negatives")

    if len(hard_neg_pairs) < needed:
        logging.warning(f"Only got {len(hard_neg_pairs)} hard negatives; need {needed}.")

    hard_neg_pairs = hard_neg_pairs[:needed]
    neg_src = np.array([p[0] for p in hard_neg_pairs])
    neg_dst = np.array([p[1] for p in hard_neg_pairs])

    X_neg_src = real_features[neg_src].numpy()
    X_neg_dst = real_features[neg_dst].numpy()
    X_neg = np.hstack((X_neg_src, X_neg_dst))
    y_neg = np.zeros(len(X_neg))

    # Combine
    X = np.vstack((X_pos, X_neg))
    y = np.concatenate((y_pos, y_neg))

    logging.info(f"Total samples: {len(X)} (pos: {len(X_pos)}, neg: {len(X_neg)})")

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def main():
    np.random.seed(42)
    graph_path = "Data/MTP_Graph.pt"
    feature_path = "Data/real_drug_features.pt"

    X_train, X_test, y_train, y_test = load_and_prepare_data(graph_path, feature_path)

    logging.info(f"Training Random Forest on {len(X_train)} samples...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    logging.info("Predicting on test set...")
    y_probs = clf.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)

    print("\n" + "=" * 55)
    print(" 📊 BASELINE 1 (HARD NEGATIVES): RANDOM FOREST")
    print("=" * 55)
    print(f"ROC-AUC Score : {roc_auc:.4f}")
    print(f"PR-AUC Score  : {pr_auc:.4f}")
    print("=" * 55)
    print("Negatives sampled from OTHER causes_* edge types,")
    print("forcing the RF to learn task-specific signal,")
    print("not just 'is this a real interaction'.")
    print("=" * 55)


if __name__ == "__main__":
    main()