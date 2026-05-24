"""
Phase 3: Counterfactual Edge-Occlusion Subgraph Visualization
"""

import os
import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from torch_geometric.nn import SAGEConv, HeteroConv, Linear
from torch_geometric.data import HeteroData
import torch.nn.functional as F


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


class LinkPredictor(torch.nn.Module):
    """Parameter-free dot-product decoder for link prediction."""
    def __init__(self, in_channels=None):
        super().__init__()

    def forward(self, z_src, z_dst):
        return (z_src * z_dst).sum(dim=-1)


class FullMTPModel(torch.nn.Module):
    """Wraps encoder + decoder for inference and explainability."""
    def __init__(self, metadata, hidden_dim, out_dim):
        super().__init__()
        self.encoder = HeteroADRModel(metadata, hidden_dim, out_dim)
        self.decoder = LinkPredictor(out_dim)

    def forward(self, x_dict, edge_index_dict, edge_label_index, target_edge_type):
        z_dict = self.encoder(x_dict, edge_index_dict)
        src_type, _, dst_type = target_edge_type
        z_src = z_dict[src_type][edge_label_index[0]]
        z_dst = z_dict[dst_type][edge_label_index[1]]
        # Inference-time temperature scaling. Monotonic, so does not
        # change ranking of edge importances; included to keep logit
        # magnitudes in a humanly readable range.
        return self.decoder(z_src, z_dst) / 15.0


def build_purged_graph(data, target_edge_type):
    """Build a HeteroData containing all node types and only:
        - non-causes_* (structural) edge types, plus
        - the single target causes_* edge type.
    This matches the purging that train_job.py performs.
    """
    data_purged = HeteroData()
    for nt in data.node_types:
        if hasattr(data[nt], 'x') and data[nt].x is not None:
            data_purged[nt].x = data[nt].x
        data_purged[nt].num_nodes = data[nt].num_nodes
    purged_count = 0
    for et in data.edge_types:
        if et[1].startswith('causes_') and et != target_edge_type:
            purged_count += 1
            continue
        data_purged[et].edge_index = data[et].edge_index
    return data_purged, purged_count


def build_name_lookup(nodes_csv_path, primekg_csv_path):
    """Construct a (node_type, type_local_idx) -> human-readable name lookup."""
    print("Building name lookup...")
    nodes_df = pd.read_csv(nodes_csv_path)
    nodes_df['node_id'] = nodes_df['node_id'].astype(str).str.strip()
    nodes_df['node_name'] = nodes_df['node_name'].astype(str).str.strip()
    id_to_name = dict(zip(nodes_df['node_id'], nodes_df['node_name']))

    primekg_df = pd.read_csv(primekg_csv_path, low_memory=False)
    primekg_df['x_id'] = primekg_df['x_id'].astype(str).str.strip()
    primekg_df['y_id'] = primekg_df['y_id'].astype(str).str.strip()

    all_nodes = pd.concat([
        primekg_df[['x_id', 'x_type']].rename(columns={'x_id': 'id', 'x_type': 'type'}),
        primekg_df[['y_id', 'y_type']].rename(columns={'y_id': 'id', 'y_type': 'type'})
    ]).drop_duplicates(subset='id')

    per_type_idx_to_name = defaultdict(dict)
    for node_type in all_nodes['type'].unique():
        nodes_of_type = all_nodes[all_nodes['type'] == node_type]['id'].values
        for i, node_id in enumerate(nodes_of_type):
            name = id_to_name.get(node_id, node_id)
            if len(name) > 28:
                name = name[:25] + "..."
            per_type_idx_to_name[node_type][i] = name

    print(f"  Built name lookup for {len(per_type_idx_to_name)} node types")
    return per_type_idx_to_name


def extract_khop_subgraph(data_purged, seed_nodes_init, num_hops, per_hop_limit,
                          target_edge_type):
    """K-hop expansion from seed nodes within the purged graph."""
    seed_nodes = {nt: set(s) for nt, s in seed_nodes_init.items()}

    for hop in range(num_hops):
        new_nodes = {nt: set(s) for nt, s in seed_nodes.items()}
        for edge_type in data_purged.edge_types:
            src_type, rel, dst_type = edge_type
            # Skip the target causes_* edge during expansion so the
            # explanation is in terms of biological pathways, not the
            # ADR layer itself.
            if edge_type == target_edge_type:
                continue
            if src_type not in seed_nodes and dst_type not in seed_nodes:
                continue
            edge_idx = data_purged[edge_type].edge_index
            if src_type in seed_nodes:
                src_tensor = torch.tensor(list(seed_nodes[src_type]))
                mask = torch.isin(edge_idx[0], src_tensor)
                dsts = edge_idx[1, mask].tolist()
                new_nodes.setdefault(dst_type, set()).update(dsts[:per_hop_limit])
            if dst_type in seed_nodes:
                dst_tensor = torch.tensor(list(seed_nodes[dst_type]))
                mask = torch.isin(edge_idx[1], dst_tensor)
                srcs = edge_idx[0, mask].tolist()
                new_nodes.setdefault(src_type, set()).update(srcs[:per_hop_limit])
        seed_nodes = new_nodes
        print(f"  After hop {hop+1}: "
              + ", ".join(f"{nt}={len(ns)}" for nt, ns in seed_nodes.items()))

    return seed_nodes


def build_local_subgraph(data_purged, seed_nodes, target_src, target_dst,
                         target_edge_type):
    """Build a HeteroData subgraph containing only nodes in seed_nodes."""
    batch = HeteroData()
    node_remap = {}
    for nt, nodes in seed_nodes.items():
        nodes_list = sorted(nodes)
        node_remap[nt] = {g: l for l, g in enumerate(nodes_list)}
        if hasattr(data_purged[nt], 'x') and data_purged[nt].x is not None:
            batch[nt].x = data_purged[nt].x[nodes_list]
        batch[nt].n_id = torch.tensor(nodes_list, dtype=torch.long)
        batch[nt].num_nodes = len(nodes_list)

    for edge_type in data_purged.edge_types:
        src_type, _, dst_type = edge_type
        if src_type not in seed_nodes or dst_type not in seed_nodes:
            continue
        src_set = set(seed_nodes[src_type])
        dst_set = set(seed_nodes[dst_type])
        edge_idx = data_purged[edge_type].edge_index
        src_tensor = torch.tensor(list(src_set))
        dst_tensor = torch.tensor(list(dst_set))
        mask = torch.isin(edge_idx[0], src_tensor) & torch.isin(edge_idx[1], dst_tensor)
        kept = edge_idx[:, mask]
        if kept.size(1) > 0:
            src_remap = torch.tensor([node_remap[src_type][g.item()] for g in kept[0]])
            dst_remap = torch.tensor([node_remap[dst_type][g.item()] for g in kept[1]])
            batch[edge_type].edge_index = torch.stack([src_remap, dst_remap])

    local_src = node_remap['drug'][target_src]
    local_dst = node_remap['drug'][target_dst]
    batch[target_edge_type].edge_label_index = torch.tensor([[local_src], [local_dst]])
    return batch, local_src, local_dst


def edge_occlusion_ablation(model, batch, target_edge_type, baseline_logit, top_k):
    """For each non-target edge in the subgraph, occlude it and measure
    the drop in prediction logit. Return the top-k edges by positive
    importance score.
    """
    clean_edges = {
        k: batch[k].edge_index
        for k in batch.edge_types
        if hasattr(batch[k], 'edge_index') and batch[k].edge_index.size(1) > 0
    }
    all_scores = []
    for edge_type, edge_index in clean_edges.items():
        if edge_type == target_edge_type:
            continue
        num_edges = edge_index.size(1)
        for i in range(num_edges):
            ablated = clean_edges.copy()
            mask = torch.ones(num_edges, dtype=torch.bool)
            mask[i] = False
            ablated[edge_type] = edge_index[:, mask]
            with torch.no_grad():
                new_logit = model(
                    batch.x_dict, ablated,
                    batch[target_edge_type].edge_label_index,
                    target_edge_type
                ).item()
            importance = baseline_logit - new_logit
            if importance > 0:
                src_local = edge_index[0, i].item()
                dst_local = edge_index[1, i].item()
                all_scores.append((importance, edge_type, src_local, dst_local))

    all_scores.sort(reverse=True, key=lambda x: x[0])
    return all_scores[:top_k]


def render_subgraph(G, src_drug_name, dst_drug_name, target_adr, top_k, output_path):
    """Render the explanation subgraph with named nodes and importance-weighted edges."""
    plt.figure(figsize=(16, 11))
    pos = nx.spring_layout(G, k=2.8, iterations=100, seed=42)

    type_colors = {
        'drug_target': '#E74C3C',
        'drug': '#F39C12',
        'effect/phenotype': '#3498DB',
        'disease': '#27AE60',
        'gene/protein': '#9B59B6',
        'pathway': '#1ABC9C',
        'anatomy': '#95A5A6',
        'biological_process': '#16A085',
        'molecular_function': '#8E44AD',
        'cellular_component': '#2C3E50',
        'exposure': '#D35400',
    }

    for ntype, color in type_colors.items():
        nodes_of_type = [n for n in G.nodes if G.nodes[n].get('ntype') == ntype]
        if not nodes_of_type:
            continue
        size = 2000 if ntype == 'drug_target' else 1200
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodes_of_type,
            node_color=color, node_size=size,
            edgecolors='black', linewidths=1.5, alpha=0.95
        )

    importance_edges = [
        (u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
        if not d.get('is_prediction')
    ]
    pred_edges = [
        (u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
        if d.get('is_prediction')
    ]

    if importance_edges:
        weights = [G[u][v][k]['weight'] for u, v, k in importance_edges]
        max_w = max(weights) if weights else 1
        widths = [1 + 4 * (w / max_w) for w in weights]
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v) for u, v, k in importance_edges],
            width=widths, edge_color='#34495E',
            arrows=True, arrowsize=18, alpha=0.7,
            connectionstyle="arc3,rad=0.08"
        )

    if pred_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v) for u, v, k in pred_edges],
            width=3, edge_color='#E74C3C',
            style='dashed', arrows=True, arrowsize=22,
            connectionstyle="arc3,rad=0.2"
        )

    labels = {n: G.nodes[n]['label'] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold')

    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C',
               markersize=15, label='Target drug pair'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#F39C12',
               markersize=12, label='Other drug'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB',
               markersize=12, label='Phenotype'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#27AE60',
               markersize=12, label='Disease'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#9B59B6',
               markersize=12, label='Gene/Protein'),
        Line2D([0], [0], color='#34495E', lw=2,
               label='Important edge (thickness ' + r'$\propto$' + ' importance)'),
        Line2D([0], [0], color='#E74C3C', lw=2, linestyle='dashed',
               label=f'Predicted: causes_{target_adr}'),
    ]
    plt.legend(handles=legend_items, loc='lower center',
               bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9, frameon=True)

    plt.title(
        f"Top-{top_k} Most Important Biological Edges for Predicting\n"
        f"'{src_drug_name}' + '{dst_drug_name}' "
        + r"$\rightarrow$" + f" causes_{target_adr}",
        fontsize=14, fontweight='bold', pad=20
    )
    plt.axis('off')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def load_model_with_materialization(model, model_path, x_dict, edge_dict, device):
    """Materialize lazy Linear(-1, h) layers via a dummy forward pass,
    then load the checkpoint with strict=True.

    This is the critical fix from the evaluate_all_adrs.py debugging:
    PyG lazy modules silently discard loaded weights if not materialized.
    """
    model.to(device).eval()
    # STEP 1: Dummy forward to materialize lazy layers
    with torch.no_grad():
        _ = model.encoder(x_dict, edge_dict)
    print("  Lazy layers materialized via dummy forward")

    # STEP 2: Load checkpoint
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    print(f"  Checkpoint has {len(state_dict)} keys")

    # Add 'encoder.' prefix if needed
    sample_key = next(iter(state_dict.keys()))
    if not sample_key.startswith('encoder.'):
        state_dict = {f'encoder.{k}': v for k, v in state_dict.items()}
        print(f"  Added 'encoder.' prefix to checkpoint keys")

    # STEP 3: Load with strict=True
    result = model.load_state_dict(state_dict, strict=True)
    print(f"  Loaded strict=True (no missing/unexpected keys)")
    return model


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Target ADR: {args.target_adr}")
    print(f"Hops: {args.hops}, per-hop limit: {args.per_hop_limit}, top-K: {args.top_k}")

    # 1. Load graph and ClinicalBERT drug features
    print("\n[1/9] Loading graph and ClinicalBERT features...")
    data = torch.load(args.graph_path, map_location='cpu', weights_only=False)
    real_features = torch.load(args.feature_path, map_location='cpu', weights_only=False)
    data['drug'].x = real_features

    # 2. Identify target edge by exact name match
    print("\n[2/9] Identifying target edge type...")
    target_edge_name = f"causes_{args.target_adr}"
    matching = [et for et in data.edge_types if et[1] == target_edge_name]
    if not matching:
        available = sorted([et[1].replace('causes_', '')
                          for et in data.edge_types if et[1].startswith('causes_')])[:20]
        raise ValueError(
            f"Edge type '{target_edge_name}' not found.\n"
            f"First 20 available: {available}"
        )
    target_edge_type = matching[0]
    print(f"  Target edge: {target_edge_type}")

    # 3. PURGE non-target ADR edge types (CRITICAL: matches training)
    print("\n[3/9] Purging non-target causes_* edge types...")
    data_purged, purged_count = build_purged_graph(data, target_edge_type)
    print(f"  Purged {purged_count} non-target ADR edge types. "
          f"Kept {len(data_purged.edge_types)} edge types.")

    # 4. Resolve trained model path
    model_path = args.model_path
    if model_path is None:
        candidate1 = f"Results/Phase2_TAG_Model_{args.target_adr}.pth"
        candidate2 = "Results/Phase2_TAG_Model.pth"  # legacy filename
        if os.path.exists(candidate1):
            model_path = candidate1
        elif os.path.exists(candidate2) and args.target_adr == "thrombocytopenia":
            model_path = candidate2
        else:
            raise FileNotFoundError(
                f"Trained model not found. Tried: {candidate1}, {candidate2}"
            )
    print(f"  Model: {model_path}")

    # 5. Pick a representative target drug pair (first positive edge)
    print("\n[4/9] Selecting target drug pair (first positive edge)...")
    target_src = data_purged[target_edge_type].edge_index[0, 0].item()
    target_dst = data_purged[target_edge_type].edge_index[1, 0].item()
    print(f"  Drug pair: drug_{target_src} + drug_{target_dst}")

    # 6. Build model with PURGED metadata and load weights properly
    print("\n[5/9] Loading trained model (purged metadata + lazy materialization)...")
    model = FullMTPModel(data_purged.metadata(), 128, 64)
    x_dict_full = {nt: data_purged[nt].x.to(device) for nt in data_purged.node_types}
    edge_dict_full = {et: data_purged[et].edge_index.to(device)
                      for et in data_purged.edge_types}
    model = load_model_with_materialization(
        model, model_path, x_dict_full, edge_dict_full, device
    )

    # 7. Extract k-hop subgraph from purged graph
    print(f"\n[6/9] Extracting {args.hops}-hop subgraph "
          f"(per-hop limit={args.per_hop_limit})...")
    seed_nodes_init = {'drug': {target_src, target_dst}}
    seed_nodes = extract_khop_subgraph(
        data_purged, seed_nodes_init,
        num_hops=args.hops, per_hop_limit=args.per_hop_limit,
        target_edge_type=target_edge_type
    )

    # 8. Build local subgraph and compute baseline logit
    print("\n[7/9] Building local subgraph and computing baseline logit...")
    batch, local_src, local_dst = build_local_subgraph(
        data_purged, seed_nodes, target_src, target_dst, target_edge_type
    )
    batch = batch.to(device)

    total_nodes = sum(batch[nt].num_nodes for nt in batch.node_types)
    total_edges = sum(
        batch[et].edge_index.size(1)
        for et in batch.edge_types if hasattr(batch[et], 'edge_index')
    )
    print(f"  Subgraph: {total_nodes} nodes, {total_edges} edges")

    clean_edges_dict = {
        k: batch[k].edge_index
        for k in batch.edge_types
        if hasattr(batch[k], 'edge_index') and batch[k].edge_index.size(1) > 0
    }
    with torch.no_grad():
        baseline_logit = model(
            batch.x_dict, clean_edges_dict,
            batch[target_edge_type].edge_label_index,
            target_edge_type
        ).item()
    print(f"  Baseline logit: {baseline_logit:.4f}")

    # 9. Run counterfactual edge-occlusion ablation
    print(f"\n[8/9] Running edge-occlusion ablation over {total_edges} edges...")
    top_edges = edge_occlusion_ablation(
        model, batch, target_edge_type, baseline_logit, args.top_k
    )
    if not top_edges:
        print("  WARNING: No edges with positive importance found.")
        return
    print(f"  Top {len(top_edges)} edges identified. "
          f"Best importance: {top_edges[0][0]:.4f}")

    # 10. Build name lookup and assemble visualization
    print("\n[9/9] Building name lookup and rendering visualization...")
    per_type_idx_to_name = build_name_lookup(args.nodes_csv, args.primekg_csv)

    def get_name(node_type, local_idx):
        if local_idx < batch[node_type].n_id.size(0):
            global_idx = batch[node_type].n_id[local_idx].item()
            return per_type_idx_to_name.get(node_type, {}).get(
                global_idx, f"{node_type}_{global_idx}"
            )
        return f"{node_type}_unk"

    G = nx.MultiDiGraph()
    src_drug_name = get_name('drug', local_src)
    dst_drug_name = get_name('drug', local_dst)
    G.add_node(f"drug::{local_src}", label=src_drug_name, ntype='drug_target')
    G.add_node(f"drug::{local_dst}", label=dst_drug_name, ntype='drug_target')

    for importance, edge_type, src_local, dst_local in top_edges:
        src_type, rel, dst_type = edge_type
        src_name = get_name(src_type, src_local)
        dst_name = get_name(dst_type, dst_local)
        src_key = f"{src_type}::{src_local}"
        dst_key = f"{dst_type}::{dst_local}"
        if not G.has_node(src_key):
            G.add_node(src_key, label=src_name, ntype=src_type)
        if not G.has_node(dst_key):
            G.add_node(dst_key, label=dst_name, ntype=dst_type)
        G.add_edge(src_key, dst_key, weight=importance, rel=rel)

    # Predicted edge (dashed red)
    G.add_edge(
        f"drug::{local_src}", f"drug::{local_dst}",
        weight=0, rel='PREDICTED', is_prediction=True
    )

    # Resolve output path
    output_path = args.output
    if output_path is None:
        output_path = f"Results/subgraph_{args.target_adr}.png"

    render_subgraph(G, src_drug_name, dst_drug_name,
                    args.target_adr, args.top_k, output_path)

    print(f"\n[Done] Visualization saved to: {output_path}")
    print(f"  Drug pair: '{src_drug_name}' + '{dst_drug_name}'")
    print(f"  ADR: causes_{args.target_adr}")
    print(f"  Top edge importance: {top_edges[0][0]:.4f}")
    print(f"  Subgraph size: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges (in viz)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Counterfactual edge-occlusion explainability "
                    "for trained polypharmacy GNN models (v3, bug-fixed)."
    )

    parser.add_argument("--target_adr", type=str, required=True,
                        help="Target ADR name (without 'causes_' prefix).")
    parser.add_argument("--hops", type=int, default=2, choices=[1, 2],
                        help="Number of hops for local subgraph (default: 2)")
    parser.add_argument("--per_hop_limit", type=int, default=5,
                        help="Max neighbors per source-relation per hop (default: 5)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top edges to visualize (default: 10)")

    parser.add_argument("--graph_path", type=str, default="Data/MTP_Graph.pt")
    parser.add_argument("--feature_path", type=str, default="Data/real_drug_features.pt")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained model (auto-discovered if None)")
    parser.add_argument("--nodes_csv", type=str, default="Data/nodes.csv")
    parser.add_argument("--primekg_csv", type=str, default="Data/kg.csv")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path (auto-named if None)")

    args = parser.parse_args()
    main(args)