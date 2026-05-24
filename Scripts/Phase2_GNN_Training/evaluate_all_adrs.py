"""
Multi-ADR Consolidated Evaluation Script

Evaluates all four trained TAG-GNN models on:
  1. Random-negative protocol (standard literature benchmark)
  2. Hard-negative protocol (negatives sampled from other ADR edges)

"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import SAGEConv, HeteroConv, Linear
from sklearn.metrics import roc_auc_score, average_precision_score

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
ADR_LIST = [
    ("thrombocytopenia", "Results/Phase2_TAG_Model.pth"),
    ("Bleeding", "Results/Phase2_TAG_Model_Bleeding.pth"),
    ("Cardiacdecompensation", "Results/Phase2_TAG_Model_Cardiacdecompensation.pth"),
    ("kidneyfailure", "Results/Phase2_TAG_Model_kidneyfailure.pth"),
]
GRAPH_PATH = "Data/MTP_Graph.pt"
FEATURE_PATH = "Data/real_drug_features.pt"
HIDDEN_DIM = 128
OUT_DIM = 64
NEG_RATIO = 1.0
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# -----------------------------------------------------------
# Model definitions (must match training script)
# -----------------------------------------------------------
class HeteroADRModel(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for _ in range(2):
            conv_dict = {et: SAGEConv((-1, -1), hidden_channels) for et in metadata[1]}
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        self.lin_out = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: self.lin_dict[k](x) for k, x in x_dict.items()}
        x_dict = self.convs[0](x_dict, edge_index_dict)
        x_dict = {k: F.relu(x) for k, x in x_dict.items()}
        x_dict = {k: F.dropout(x, p=0.3, training=self.training) for k, x in x_dict.items()}
        x_dict = self.convs[1](x_dict, edge_index_dict)
        return {k: self.lin_out(x) for k, x in x_dict.items()}


def load_encoder(model_path, metadata, device):
    encoder = HeteroADRModel(metadata, HIDDEN_DIM, OUT_DIM)
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    if any(k.startswith('encoder.') for k in state_dict.keys()):
        state_dict = {k.replace('encoder.', '', 1): v for k, v in state_dict.items()}
    encoder_keys = set(encoder.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in encoder_keys}
    encoder.load_state_dict(filtered, strict=False)
    encoder.to(device).eval()
    return encoder


def get_target_edge_type(data, adr_name):
    target_name = f"causes_{adr_name}"
    matching = [et for et in data.edge_types if et[1] == target_name]
    if not matching:
        raise ValueError(f"Edge type '{target_name}' not found in graph")
    return matching[0]


def compute_embeddings(encoder, data, target_edge_type, device):
    edge_index_dict = {
        et: data[et].edge_index.to(device)
        for et in data.edge_types
        if et != target_edge_type
    }
    x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
    with torch.no_grad():
        z_dict = encoder(x_dict, edge_index_dict)
    return z_dict['drug']


def evaluate_random_negatives(z_drug, pos_edges, num_drugs, device):
    num_pos = pos_edges.size(1)
    num_neg = int(num_pos * NEG_RATIO)

    pos_set = set()
    for i in range(num_pos):
        u, v = pos_edges[0, i].item(), pos_edges[1, i].item()
        pos_set.add((u, v))
        pos_set.add((v, u))

    neg_src = torch.randint(0, num_drugs, (num_neg,))
    neg_dst = torch.randint(0, num_drugs, (num_neg,))
    keep = []
    for i in range(num_neg):
        u, v = neg_src[i].item(), neg_dst[i].item()
        if u != v and (u, v) not in pos_set:
            keep.append(i)
    neg_src = neg_src[keep]
    neg_dst = neg_dst[keep]
    num_neg = len(keep)

    z_pos_src = z_drug[pos_edges[0]]
    z_pos_dst = z_drug[pos_edges[1]]
    pos_scores = (z_pos_src * z_pos_dst).sum(dim=-1).cpu().numpy()

    z_neg_src = z_drug[neg_src.to(device)]
    z_neg_dst = z_drug[neg_dst.to(device)]
    neg_scores = (z_neg_src * z_neg_dst).sum(dim=-1).cpu().numpy()

    y_true = np.concatenate([np.ones(num_pos), np.zeros(num_neg)])
    y_score = np.concatenate([pos_scores, neg_scores])
    return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)


def evaluate_hard_negatives(z_drug, pos_edges, data, target_edge_type, device):
    num_pos = pos_edges.size(1)
    target_set = set()
    for i in range(num_pos):
        u, v = pos_edges[0, i].item(), pos_edges[1, i].item()
        target_set.add((u, v))
        target_set.add((v, u))

    other_pairs = []
    for et in data.edge_types:
        if et == target_edge_type:
            continue
        if not et[1].startswith('causes_'):
            continue
        edge_idx = data[et].edge_index
        for i in range(edge_idx.size(1)):
            u, v = edge_idx[0, i].item(), edge_idx[1, i].item()
            if (u, v) not in target_set and (v, u) not in target_set:
                other_pairs.append((u, v))

    if len(other_pairs) == 0:
        return float('nan'), float('nan')

    np.random.shuffle(other_pairs)
    hard_negs = other_pairs[:num_pos]
    neg_src = torch.tensor([p[0] for p in hard_negs])
    neg_dst = torch.tensor([p[1] for p in hard_negs])

    z_pos_src = z_drug[pos_edges[0]]
    z_pos_dst = z_drug[pos_edges[1]]
    pos_scores = (z_pos_src * z_pos_dst).sum(dim=-1).cpu().numpy()

    z_neg_src = z_drug[neg_src.to(device)]
    z_neg_dst = z_drug[neg_dst.to(device)]
    neg_scores = (z_neg_src * z_neg_dst).sum(dim=-1).cpu().numpy()

    y_true = np.concatenate([np.ones(num_pos), np.zeros(len(hard_negs))])
    y_score = np.concatenate([pos_scores, neg_scores])
    return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n", flush=True)

    print("Loading graph and ClinicalBERT features...", flush=True)
    data = torch.load(GRAPH_PATH, map_location='cpu', weights_only=False)
    real_features = torch.load(FEATURE_PATH, map_location='cpu', weights_only=False)
    data['drug'].x = real_features
    num_drugs = data['drug'].num_nodes
    print(f"  Graph loaded. Drug nodes: {num_drugs}\n", flush=True)

    results = []
    for adr_name, model_path in ADR_LIST:
        print(f"========================================", flush=True)
        print(f"Evaluating ADR: {adr_name}", flush=True)
        print(f"  Model: {model_path}", flush=True)

        if not os.path.exists(model_path):
            print(f"  [ERROR] Model file not found, skipping", flush=True)
            results.append({"adr": adr_name, "error": "model file missing"})
            continue

        try:
            target_edge_type = get_target_edge_type(data, adr_name)
            pos_edges = data[target_edge_type].edge_index.to(device)
            num_pos = pos_edges.size(1)
            print(f"  Target edge: {target_edge_type}", flush=True)
            print(f"  Positive examples: {num_pos}", flush=True)

            encoder = load_encoder(model_path, data.metadata(), device)
            z_drug = compute_embeddings(encoder, data, target_edge_type, device)

            rnd_roc, rnd_pr = evaluate_random_negatives(z_drug, pos_edges, num_drugs, device)
            print(f"  Random-neg:  ROC-AUC = {rnd_roc:.4f}, PR-AUC = {rnd_pr:.4f}", flush=True)

            hrd_roc, hrd_pr = evaluate_hard_negatives(z_drug, pos_edges, data, target_edge_type, device)
            print(f"  Hard-neg:    ROC-AUC = {hrd_roc:.4f}, PR-AUC = {hrd_pr:.4f}", flush=True)

            results.append({
                "adr": adr_name, "num_pos": num_pos,
                "rnd_roc": rnd_roc, "rnd_pr": rnd_pr,
                "hrd_roc": hrd_roc, "hrd_pr": hrd_pr,
                "error": None
            })
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            results.append({"adr": adr_name, "error": str(e)})
        print(flush=True)

    print("\n", flush=True)
    print("=" * 78, flush=True)
    print("CONSOLIDATED RESULTS - ALL ADRs, BOTH PROTOCOLS", flush=True)
    print("=" * 78, flush=True)
    print(f"{'ADR':<25} {'Pos':>6} {'RND-ROC':>9} {'RND-PR':>9} {'HRD-ROC':>9} {'HRD-PR':>9}", flush=True)
    print("-" * 78, flush=True)
    for r in results:
        if r.get("error"):
            print(f"{r['adr']:<25} ERROR: {r['error']}", flush=True)
            continue
        print(
            f"{r['adr']:<25} "
            f"{r['num_pos']:>6} "
            f"{r['rnd_roc']:>9.4f} "
            f"{r['rnd_pr']:>9.4f} "
            f"{r['hrd_roc']:>9.4f} "
            f"{r['hrd_pr']:>9.4f}",
            flush=True
        )
    print("=" * 78, flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()