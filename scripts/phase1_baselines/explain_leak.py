"""
Ablation Leak Detector.

This diagnostic script interrogates a trained heterogeneous graph neural network 
for multiplex edge contamination. By systematically occluding entire edge types 
(macro-ablation) and monitoring global confidence degradation, this tool 
identifies if the model improperly memorized parallel pathways (data leakage)
instead of learning true structural topology.
"""

import argparse
import logging
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear

# Configure professional diagnostic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

class HeteroADRModel(torch.nn.Module):
    """Encoder module for heterogeneous node representation learning."""
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
    """Decoder module computing interaction probability via dot-product."""
    def __init__(self, in_channels):
        super().__init__()
        self.lin = Linear(in_channels, in_channels)
        
    def forward(self, z_src, z_dst):
        return (z_src * z_dst).sum(dim=-1)

class FullMTPModel(torch.nn.Module):
    """End-to-end architecture combining the encoder and link predictor."""
    def __init__(self, metadata, hidden_dim, out_dim):
        super().__init__()
        self.encoder = HeteroADRModel(metadata, hidden_dim, out_dim)
        self.decoder = LinkPredictor(out_dim)
        
    def forward(self, x_dict, edge_index_dict, edge_label_index, target_edge_type):
        z_dict = self.encoder(x_dict, edge_index_dict)
        src_type, _, dst_type = target_edge_type
        
        z_src = z_dict[src_type][edge_label_index[0]]
        z_dst = z_dict[dst_type][edge_label_index[1]]
        return self.decoder(z_src, z_dst)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Initializing Ablation Leak Detector on device: {device}")

    # 1. Load Data and Model
    logging.info(f"Loading Graph Data from {args.graph_path}...")
    try:
        data = torch.load(args.graph_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        logging.error(f"Critical: {args.graph_path} not found.")
        return

    logging.info("Loading Trained Model Weights...")
    model = FullMTPModel(data.metadata(), args.hidden_dim, args.output_dim)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True), strict=False)
    model.to(device)
    model.eval()

    # Find target edge dynamically
    target_edge_type = [et for et in data.edge_types if 'causes' in et[1]][0]
    logging.info(f"Target Edge Type: {target_edge_type}")
    
    # Test a batch of edges to get a solid average
    edge_label_index = data[target_edge_type].edge_index[:, :args.test_samples].to(device)
    
    with torch.no_grad():
        baseline_preds = model(data.x_dict, data.edge_index_dict, edge_label_index, target_edge_type)
        baseline_score = torch.sigmoid(baseline_preds).mean().item()
        
    logging.info(f"Baseline Confidence (With all edges): {baseline_score * 100:.2f}%")
    logging.info("🚨 RUNNING ABLATION: REMOVING EDGE TYPES ONE BY ONE 🚨")
    print("-" * 70)
    
    # 2. The Ablation Loop
    for edge_type in data.edge_types:
        if edge_type == target_edge_type:
            continue
            
        # Backup the true edges
        backup_edges = data.edge_index_dict[edge_type].clone()
        
        # Temporarily delete the edges from the graph
        data.edge_index_dict[edge_type] = torch.empty((2, 0), dtype=torch.long, device=device)
        
        # Test the model's confidence without them
        with torch.no_grad():
            preds = model(data.x_dict, data.edge_index_dict, edge_label_index, target_edge_type)
            score = torch.sigmoid(preds).mean().item()
            
        drop = baseline_score - score
        
        if drop > args.threshold: 
            logging.warning(f"MASSIVE LEAK FOUND! Hiding '{edge_type[1]}' caused confidence to drop by {drop*100:.1f}%.")
        else:
            logging.info(f"Hiding '{edge_type[1]}': No major change (Confidence: {score*100:.1f}%)")
            
        # Put the edges back for the next test
        data.edge_index_dict[edge_type] = backup_edges

    print("-" * 70)
    logging.info("Ablation Leak Detection Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ablation Leak Detector.")
    parser.add_argument("--graph_path", type=str, default="MTP_Graph.pt", help="Path to compiled .pt graph")
    parser.add_argument("--model_path", type=str, default="MTP_Model_Param.pth", help="Path to trained weights")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden channel dimension")
    parser.add_argument("--output_dim", type=int, default=64, help="Output channel dimension")
    parser.add_argument("--test_samples", type=int, default=500, help="Number of sample edges")
    parser.add_argument("--threshold", type=float, default=0.15, help="Confidence drop leak threshold")
    
    args = parser.parse_args()
    main(args)