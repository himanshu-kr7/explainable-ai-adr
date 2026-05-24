"""
Target Leakage Diagnostic (Contamination Test).

This script assesses a trained heterogeneous graph neural network for target 
edge leakage. It evaluates whether the model relies heavily on the presence 
of the target edge within the message-passing topology to make its prediction, 
or if it has successfully learned underlying biological generalizations.
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
    logging.info(f"Initializing Target Leakage Diagnostic on device: {device}")

    # 1. Initialization
    try:
        data = torch.load(args.graph_path, map_location=device, weights_only=False)
        logging.info(f"Graph topology loaded successfully from {args.graph_path}")
    except FileNotFoundError:
        logging.error(f"Critical: {args.graph_path} not found.")
        return

    model = FullMTPModel(data.metadata(), args.hidden_dim, args.output_dim)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True), strict=False)
    model.to(device)
    model.eval()

    target_edge_type = [et for et in data.edge_types if 'causes' in et[1]][0]
    logging.info(f"Target prediction edge isolated: {target_edge_type}")
    # Purge all other ADR edges to match training-time graph structure
    edges_to_delete = [et for et in list(data.edge_types) 
                   if et[1].startswith('causes_') and et != target_edge_type]
    for et in edges_to_delete:
        del data[et]
    logging.info(f"Purged {len(edges_to_delete)} non-target ADR edge types.")
    # Extract test cohort
    edge_label_index = data[target_edge_type].edge_index[:, :args.test_samples].to(device)
    
    # ==========================================
    # Phase 1: Target-Inclusion Assessment (Baseline)
    # ==========================================
    logging.info("Executing Phase 1: Target-Inclusion Assessment (Message-passing intact)...")
    with torch.no_grad():
        preds_included = model(data.x_dict, data.edge_index_dict, edge_label_index, target_edge_type)
        score_included = torch.sigmoid(preds_included).mean().item()
    
    logging.info(f"Baseline Confidence (Target Included): {score_included * 100:.2f}%")

    # ==========================================
    # Phase 2: Target-Blinded Assessment (Strict Evaluation)
    # ==========================================
    logging.info("Executing Phase 2: Target-Blinded Assessment (Isolating test cohort)...")
    
    # Exclude the test edges from the message-passing index
    blind_edges = data.edge_index_dict[target_edge_type][:, args.test_samples:] 
    
    # Temporarily replace the graph's target edges with the blinded version
    data[target_edge_type].edge_index = blind_edges

    with torch.no_grad():
        preds_blinded = model(data.x_dict, data.edge_index_dict, edge_label_index, target_edge_type)
        score_blinded = torch.sigmoid(preds_blinded).mean().item()
        
    logging.info(f"Strict Evaluation Confidence (Target Blinded): {score_blinded * 100:.2f}%")
    
    # ==========================================
    # Final Diagnostic Analysis
    # ==========================================
    drop = score_included - score_blinded
    print("\n" + "="*60)
    print(" TARGET LEAKAGE DIAGNOSTIC REPORT")
    print("="*60)
    print(f" Absolute Confidence Drop: {drop * 100:.2f}%")
    
    if drop > 0.05: # 5% drop warning
        logging.warning("Moderate to high confidence degradation detected. The model relies heavily on target presence.")
    else:
        logging.info("Minimal confidence degradation. Model demonstrates robust biological generalization, proving zero target leakage.")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Target Leakage Diagnostic Test.")
    parser.add_argument("--graph_path", type=str, default="Data/MTP_Graph.pt", help="Path to compiled .pt graph")
    parser.add_argument("--model_path", type=str, default="Results/MTP_Model_Param.pth", help="Path to trained weights")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden channel dimension")
    parser.add_argument("--output_dim", type=int, default=64, help="Output channel dimension")
    parser.add_argument("--test_samples", type=int, default=500, help="Number of sample edges to isolate for testing")
    
    args = parser.parse_args()
    main(args)