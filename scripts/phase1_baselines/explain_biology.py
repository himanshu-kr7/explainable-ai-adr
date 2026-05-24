"""
Edge-Occlusion Ablation Study for Heterogeneous GNNs.
Phase 2: Text-Attributed Graph (TAG) Integration
"""

import argparse
import logging
import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, HeteroConv, Linear

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class HeteroADRModel(torch.nn.Module):
    """Must perfectly match the architecture in train_job.py"""
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

        logits = self.decoder(z_src, z_dst)
        return logits / 15.0 # Temperature scaling

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # 1. Load the Data
    logging.info("Loading heterogeneous graph topology...")
    data = torch.load(args.graph_path, map_location='cpu', weights_only=False)
    
    # --- SURGICAL INJECTION: Load ClinicalBERT Features ---
    if args.use_real_features:
        logging.info("🚨 INJECTING CLINICALBERT EMBEDDINGS FOR XAI 🚨")
        real_features = torch.load(args.feature_path, map_location='cpu', weights_only=False)
        data['drug'].x = real_features
        logging.info(f"✅ Replaced dummy features with {data['drug'].x.shape[1]}-dim NLP embeddings.")

    # Isolate the target edge
    target_edge_type = [et for et in data.edge_types if et[1].startswith('causes_')][0]
    for et in list(data.edge_types):
        if et[1].startswith('causes_') and et != target_edge_type:
            del data[et]

    # 2. Initialize and Load Model
    logging.info("Initializing and loading trained model weights...")
    model = FullMTPModel(data.metadata(), 128, 64)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu', weights_only=True), strict=True)
    model.to(device)
    model.eval()

    # 3. Extract Subgraph Context
    logging.info(f"Extracting sub-graph context for {target_edge_type[1]}...")
    explanation_loader = LinkNeighborLoader(
        data,
        num_neighbors=[10, 5],
        batch_size=1,
        edge_label_index=(target_edge_type, data[target_edge_type].edge_index[:, :1]),
        shuffle=False
    )
    
    subgraph_batch = next(iter(explanation_loader)).to(device)
    logging.info(f"Context isolated. Nodes: {subgraph_batch.num_nodes}, Edges: {subgraph_batch.num_edges}")

    # 4. Setup Edge-Occlusion Ablation
    logging.info("Initiating Edge-Occlusion Ablation Study...")
    clean_edge_index_dict = {
        k: v for k, v in subgraph_batch.edge_index_dict.items() if v.size(1) > 0
    }

    with torch.no_grad():
        baseline_logit = model(
            subgraph_batch.x_dict, 
            clean_edge_index_dict, 
            subgraph_batch[target_edge_type].edge_label_index, 
            target_edge_type
        ).item()

    logging.info(f"Baseline Confidence Score: {baseline_logit:.4f}")
    
    # 5. Execute Ablation Loop
    all_scores = []
    for edge_type, edge_index in clean_edge_index_dict.items():
        if edge_type == target_edge_type: continue 
        
        num_edges = edge_index.size(1)
        for i in range(num_edges):
            ablated_dict = clean_edge_index_dict.copy()
            mask = torch.ones(num_edges, dtype=torch.bool)
            mask[i] = False
            ablated_dict[edge_type] = edge_index[:, mask]
            
            with torch.no_grad():
                new_logit = model(
                    subgraph_batch.x_dict, 
                    ablated_dict, 
                    subgraph_batch[target_edge_type].edge_label_index, 
                    target_edge_type
                ).item()
            
            importance = baseline_logit - new_logit
            if importance > 0: 
                all_scores.append((importance, edge_type, i))

    # 6. Format and Output Results
    print("\n" + "="*60)
    print(" 🔬 XAI RESULTS: TOP BIOLOGICAL PATHWAYS")
    print("="*60)
    
    all_scores.sort(reverse=True, key=lambda x: x[0])

    if not all_scores:
        print("Result: Topological structure yielded no impact. Model relied solely on text features.")
    else:
        max_score = all_scores[0][0]
        for rank in range(min(args.top_k, len(all_scores))):
            raw_score, edge_type, _ = all_scores[rank]
            rel_score = (raw_score / max_score) if max_score > 0 else 0.0
            
            print(f"[{rank+1}] Relative Importance: {rel_score:.2f} | Confidence Drop: {raw_score:.2f}")
            print(f"    Pathway: {edge_type[0]} -> ({edge_type[1]}) -> {edge_type[2]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 2 Edge-Occlusion xAI Study.")
    parser.add_argument("--graph_path", type=str, default="Data/MTP_Graph.pt")
    parser.add_argument("--feature_path", type=str, default="Data/real_drug_features.pt")
    parser.add_argument("--model_path", type=str, default="Results/Phase2_TAG_Model.pth")
    parser.add_argument("--use_real_features", type=bool, default=True)
    parser.add_argument("--top_k", type=int, default=5)
    
    args = parser.parse_args()
    main(args)