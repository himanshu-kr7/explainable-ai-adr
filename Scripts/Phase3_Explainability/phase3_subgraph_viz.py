"""
Phase 3 Subgraph Visualization
"""

import os
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
    def __init__(self, in_channels):
        super().__init__()
        self.lin = Linear(in_channels, in_channels)

    def forward(self, z_src, z_dst):
        return (z_src * z_dst).sum(dim=-1)


class FullMTPModel(torch.nn.Module):
    def __init__(self, metadata, hidden_dim, out_dim):
        super().__init__()
        self.encoder = HeteroADRModel(metadata, hidden_dim, out_dim)
        self.decoder = LinkPredictor(out_dim)

    def forward(self, x_dict, edge_index_dict, edge_label_index, target_edge_type):
        z_dict = self.encoder(x_dict, edge_index_dict)
        src_type, _, dst_type = target_edge_type
        z_src = z_dict[src_type][edge_label_index[0]]
        z_dst = z_dict[dst_type][edge_label_index[1]]
        return self.decoder(z_src, z_dst) / 15.0


def build_name_lookup(nodes_csv_path, primekg_csv_path):
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

    print(f"Built name lookup for {len(per_type_idx_to_name)} node types")
    return per_type_idx_to_name


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    GRAPH_PATH = "Data/MTP_Graph.pt"
    FEATURE_PATH = "Data/real_drug_features.pt"
    MODEL_PATH = "Results/Phase2_TAG_Model.pth"
    NODES_CSV = "Data/nodes.csv"
    PRIMEKG_CSV = "Data/kg.csv"
    OUTPUT_PATH = "Results/subgraph_visualization.png"
    TOP_K = 10

    # 1. Load graph + features
    print("Loading graph...")
    data = torch.load(GRAPH_PATH, map_location='cpu', weights_only=False)
    real_features = torch.load(FEATURE_PATH, map_location='cpu', weights_only=False)
    data['drug'].x = real_features

    # 2. Identify target edge type (keep all edge types for model compatibility)
    target_edge_type = [et for et in data.edge_types if et[1] == 'causes_thrombocytopenia'][0]
    print(f"Target edge: {target_edge_type}")

    # 3. Load model with flexible key matching
    print("Loading trained model...")
    model = FullMTPModel(data.metadata(), 128, 64)
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)

    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())
    need_prefix = any(k.startswith('encoder.') for k in model_keys) and not any(k.startswith('encoder.') for k in ckpt_keys)
    remove_prefix = not any(k.startswith('encoder.') for k in model_keys) and any(k.startswith('encoder.') for k in ckpt_keys)

    if need_prefix:
        state_dict = {f'encoder.{k}': v for k, v in state_dict.items()}
        print("Added 'encoder.' prefix to checkpoint keys")
    elif remove_prefix:
        state_dict = {k.replace('encoder.', '', 1): v for k, v in state_dict.items()}
        print("Removed 'encoder.' prefix")

    model_keys_set = set(model.state_dict().keys())
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys_set}
    model.load_state_dict(filtered_state_dict, strict=False)
    print(f"Loaded {len(filtered_state_dict)}/{len(state_dict)} weight tensors")
    model.to(device).eval()

    # 4. Extract 2-hop subgraph manually
    print("Extracting 2-hop subgraph manually...")
    target_src = data[target_edge_type].edge_index[0, 0].item()
    target_dst = data[target_edge_type].edge_index[1, 0].item()
    print(f"Target drug pair: drug_{target_src} + drug_{target_dst}")

    seed_nodes = {'drug': {target_src, target_dst}}
    PER_HOP_LIMIT = 5  # Limit neighbors per edge type per hop

    # Only do 1 hop (drug -> immediate biological neighbors)
    # This keeps subgraph small but still biologically meaningful
    for hop in range(1):
        new_nodes = {nt: set(s) for nt, s in seed_nodes.items()}
        for edge_type in data.edge_types:
            src_type, _, dst_type = edge_type
            # Skip causes_* edges - we don't want side-effect shortcuts in the explanation
            if edge_type[1].startswith('causes_'):
                continue
            if src_type not in seed_nodes and dst_type not in seed_nodes:
                continue
            edge_idx = data[edge_type].edge_index
            if src_type in seed_nodes:
                src_tensor = torch.tensor(list(seed_nodes[src_type]))
                mask = torch.isin(edge_idx[0], src_tensor)
                dsts = edge_idx[1, mask].tolist()
                new_nodes.setdefault(dst_type, set()).update(dsts[:PER_HOP_LIMIT])
            if dst_type in seed_nodes:
                dst_tensor = torch.tensor(list(seed_nodes[dst_type]))
                mask = torch.isin(edge_idx[1], dst_tensor)
                srcs = edge_idx[0, mask].tolist()
                new_nodes.setdefault(src_type, set()).update(srcs[:PER_HOP_LIMIT])
        seed_nodes = new_nodes
        

    # Build batch HeteroData
    batch = HeteroData()
    node_remap = {}
    for nt, nodes in seed_nodes.items():
        nodes_list = sorted(nodes)
        node_remap[nt] = {g: l for l, g in enumerate(nodes_list)}
        if hasattr(data[nt], 'x') and data[nt].x is not None:
            batch[nt].x = data[nt].x[nodes_list]
        batch[nt].n_id = torch.tensor(nodes_list, dtype=torch.long)
        batch[nt].num_nodes = len(nodes_list)

    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        if src_type not in seed_nodes or dst_type not in seed_nodes:
            continue
        src_set = set(seed_nodes[src_type])
        dst_set = set(seed_nodes[dst_type])
        edge_idx = data[edge_type].edge_index
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

    batch = batch.to(device)

    total_nodes = sum(batch[nt].num_nodes for nt in batch.node_types)
    total_edges = sum(batch[et].edge_index.size(1) for et in batch.edge_types if hasattr(batch[et], 'edge_index'))
    print(f"Subgraph: {total_nodes} nodes, {total_edges} edges")

    target_drug_src_idx = local_src
    target_drug_dst_idx = local_dst

    # 5. Build name lookup
    per_type_idx_to_name = build_name_lookup(NODES_CSV, PRIMEKG_CSV)

    # 6. Baseline confidence
    clean_edges = {k: batch[k].edge_index for k in batch.edge_types if hasattr(batch[k], 'edge_index') and batch[k].edge_index.size(1) > 0}
    with torch.no_grad():
        baseline_logit = model(batch.x_dict, clean_edges,
                                batch[target_edge_type].edge_label_index,
                                target_edge_type).item()
    print(f"Baseline logit: {baseline_logit:.2f}")

    # 7. Edge-occlusion ablation
    print("Running ablation...")
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
                new_logit = model(batch.x_dict, ablated,
                                   batch[target_edge_type].edge_label_index,
                                   target_edge_type).item()
            importance = baseline_logit - new_logit
            if importance > 0:
                src_local = edge_index[0, i].item()
                dst_local = edge_index[1, i].item()
                all_scores.append((importance, edge_type, src_local, dst_local))

    all_scores.sort(reverse=True, key=lambda x: x[0])
    top_edges = all_scores[:TOP_K]
    print(f"Top {len(top_edges)} edges selected. Best drop: {top_edges[0][0]:.2f}" if top_edges else "No edges")

    # 8. Map local subgraph indices to global → name
    def get_name(node_type, local_idx):
        if local_idx < batch[node_type].n_id.size(0):
            global_idx = batch[node_type].n_id[local_idx].item()
            return per_type_idx_to_name.get(node_type, {}).get(global_idx, f"{node_type}_{global_idx}")
        return f"{node_type}_unk"

    # 9. Build NetworkX graph
    G = nx.MultiDiGraph()
    src_drug_name = get_name('drug', target_drug_src_idx)
    dst_drug_name = get_name('drug', target_drug_dst_idx)
    G.add_node(f"drug::{target_drug_src_idx}", label=src_drug_name, ntype='drug_target')
    G.add_node(f"drug::{target_drug_dst_idx}", label=dst_drug_name, ntype='drug_target')

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

    src_key = f"drug::{target_drug_src_idx}"
    dst_key = f"drug::{target_drug_dst_idx}"
    G.add_edge(src_key, dst_key, weight=0, rel='PREDICTED', is_prediction=True)

    print(f"Visualization: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 10. Render
    print("Rendering...")
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
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_of_type,
                                node_color=color, node_size=size,
                                edgecolors='black', linewidths=1.5, alpha=0.95)

    importance_edges = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if not d.get('is_prediction')]
    pred_edges = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if d.get('is_prediction')]

    if importance_edges:
        weights = [G[u][v][k]['weight'] for u, v, k in importance_edges]
        max_w = max(weights) if weights else 1
        widths = [1 + 4 * (w / max_w) for w in weights]
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, k in importance_edges],
                                width=widths, edge_color='#34495E',
                                arrows=True, arrowsize=18, alpha=0.7,
                                connectionstyle="arc3,rad=0.08")

    if pred_edges:
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, k in pred_edges],
                                width=3, edge_color='#E74C3C',
                                style='dashed', arrows=True, arrowsize=22,
                                connectionstyle="arc3,rad=0.2")

    labels = {n: G.nodes[n]['label'] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold')

    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C', markersize=15, label='Target drug pair'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#F39C12', markersize=12, label='Other drug'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB', markersize=12, label='Phenotype'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#27AE60', markersize=12, label='Disease'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#9B59B6', markersize=12, label='Gene/Protein'),
        Line2D([0], [0], color='#34495E', lw=2, label='Important edge (thickness = importance)'),
        Line2D([0], [0], color='#E74C3C', lw=2, linestyle='dashed', label='Predicted: causes_thrombocytopenia'),
    ]
    plt.legend(handles=legend_items, loc='lower center', bbox_to_anchor=(0.5, -0.08),
                ncol=2, fontsize=9, frameon=True)

    plt.title(f"Top-{TOP_K} Most Important Biological Edges for Predicting\n"
              f"'{src_drug_name}' + '{dst_drug_name}' " + r"$\rightarrow$" + " causes_thrombocytopenia",
              fontsize=14, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Saved to {OUTPUT_PATH}")
    print(f"   Drug pair: '{src_drug_name}' + '{dst_drug_name}'")


if __name__ == "__main__":
    main()