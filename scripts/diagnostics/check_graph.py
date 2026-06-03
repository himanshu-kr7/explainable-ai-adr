import torch
from torch_geometric.data import HeteroData

print("Loading graph...")
# Adjusting the path since you are running it from the main folder
graph = torch.load("MTP_Graph.pt", map_location='cpu', weights_only=False)

print("\nGraph Blueprint:")
print(graph)