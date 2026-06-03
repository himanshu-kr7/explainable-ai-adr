"""
Heterogeneous Graph Neural Network Training Pipeline for Polypharmacy Side Effects.
Phase 2: Text-Attributed Graph (TAG) Integration

VERSION: 2 (multi-ADR support)
Changes from v1:
  - Explicit target ADR selection via --target_adr CLI argument
  - No reliance on dictionary ordering for target edge selection
  - Cleaned LinkPredictor (removed unused Linear layer)
"""

import os
import time
import argparse
import logging
import torch
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, HeteroConv, Linear
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score

# Optimize multiprocessing for HPC environments
torch.multiprocessing.set_sharing_strategy("file_system")

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


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
    """Parameter-free dot-product decoder for link prediction.

    Computes score(u, v) = z_u^T z_v where z_u, z_v are the GNN output
    embeddings of the source and destination drugs. No learnable
    parameters: this is a pure dot product followed by sigmoid (applied
    by the BCE-with-logits loss during training).
    """
    def __init__(self, in_channels=None):
        super().__init__()
        # in_channels retained for API compatibility but unused.

    def forward(self, z_src, z_dst):
        return (z_src * z_dst).sum(dim=-1)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Initializing Phase 2 training protocol on device: {device}")
    logging.info(f"Target ADR: {args.target_adr}")

    # 1. Load Data
    if not os.path.exists(args.graph_path):
        raise FileNotFoundError(f"Critical Error: Topology file {args.graph_path} not found.")

    data = torch.load(args.graph_path, weights_only=False)
    logging.info(f"Graph topology loaded. Nodes: {data.num_nodes}, Edges: {data.num_edges}")

    # --- ClinicalBERT Text-Attributed Graph Injection ---
    if args.use_real_features:
        logging.info("Loading ClinicalBERT text-attributed drug features...")
        if not os.path.exists(args.feature_path):
            raise FileNotFoundError(f"Missing ClinicalBERT features at {args.feature_path}")

        real_features = torch.load(args.feature_path, weights_only=False)
        data['drug'].x = real_features
        logging.info(f"Replaced drug features with {data['drug'].x.shape[1]}-dim ClinicalBERT embeddings.")

    # 2. Explicit Target ADR Selection (no longer relies on dict ordering)
    target_edge_name = f"causes_{args.target_adr}"
    matching_edges = [et for et in data.edge_types if et[1] == target_edge_name]

    if not matching_edges:
        available = sorted([et[1].replace('causes_', '')
                          for et in data.edge_types
                          if et[1].startswith('causes_')])[:20]
        raise ValueError(
            f"Target edge type '{target_edge_name}' not found in graph.\n"
            f"First 20 available ADRs: {available}"
        )

    target_edge_type = matching_edges[0]
    num_positives = data[target_edge_type].edge_index.size(1)
    logging.info(f"Target edge type: {target_edge_type}")
    logging.info(f"Positive examples in dataset: {num_positives}")

    # 3. Purge other causes_* edge types (prevents multiplex leakage)
    purged_count = 0
    for et in list(data.edge_types):
        if et[1].startswith('causes_') and et != target_edge_type:
            del data[et]
            purged_count += 1
    logging.info(f"Purged {purged_count} non-target ADR edge types. "
                 f"Remaining edge types: {len(data.edge_types)}")

    # 4. Random link split (80/10/10)
    logging.info("Applying RandomLinkSplit (80/10/10)...")
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True,
        edge_types=target_edge_type
    )
    train_data, val_data, test_data = transform(data)

    # 5. Create high-throughput loaders
    kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'persistent_workers': True}

    train_loader = LinkNeighborLoader(
        train_data, num_neighbors=[20, 10],
        edge_label_index=(target_edge_type, train_data[target_edge_type].edge_label_index),
        edge_label=train_data[target_edge_type].edge_label, shuffle=True, **kwargs
    )
    val_loader = LinkNeighborLoader(
        val_data, num_neighbors=[20, 10],
        edge_label_index=(target_edge_type, val_data[target_edge_type].edge_label_index),
        edge_label=val_data[target_edge_type].edge_label, shuffle=False, **kwargs
    )

    # 6. Initialize architecture
    model = HeteroADRModel(data.metadata(), args.hidden_dim, args.output_dim).to(device)
    decoder = LinkPredictor(args.output_dim).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(decoder.parameters()), lr=args.lr
    )

    # 7. Training loop
    def train_epoch():
        model.train()
        decoder.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict)

            edge_label_index = batch[target_edge_type].edge_label_index
            z_src = out['drug'][edge_label_index[0]]
            z_dst = out['drug'][edge_label_index[1]]

            pred = decoder(z_src, z_dst)
            loss = F.binary_cross_entropy_with_logits(
                pred, batch[target_edge_type].edge_label
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        decoder.eval()
        preds, targets = [], []
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            edge_label_index = batch[target_edge_type].edge_label_index
            z_src = out['drug'][edge_label_index[0]]
            z_dst = out['drug'][edge_label_index[1]]

            preds.append(decoder(z_src, z_dst).sigmoid().cpu())
            targets.append(batch[target_edge_type].edge_label.cpu())
        return roc_auc_score(torch.cat(targets), torch.cat(preds))

    logging.info(f"Starting training: {args.epochs} epochs on '{args.target_adr}'")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch()
        val_auc = evaluate(val_loader)
        logging.info(
            f"Epoch: {epoch:02d} | Loss: {loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | Time: {time.time()-t0:.1f}s"
        )

    # 8. Save artifacts
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    logging.info(f"Training complete. Model weights saved to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train HeteroADRModel on a specified polypharmacy ADR target"
    )

    # Target ADR (NEW in v2)
    parser.add_argument(
        "--target_adr", type=str, default="thrombocytopenia",
        help="Target ADR name (without 'causes_' prefix). "
             "Examples: thrombocytopenia, Bleeding, Cardiacdecompensation, kidneyfailure"
    )

    # Paths
    parser.add_argument("--graph_path", type=str, default="Data/MTP_Graph.pt")
    parser.add_argument("--feature_path", type=str, default="Data/real_drug_features.pt")
    parser.add_argument("--save_path", type=str, default="results/Phase2_TAG_Model.pth")

    # Phase 2 flags
    parser.add_argument("--use_real_features", type=bool, default=True)
    parser.add_argument("--full_graph", type=bool, default=True)

    # Hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--output_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()
    main(args)
