import os
import torch
import pandas as pd
from torch_geometric.explain import Explainer, GNNExplainer

# We will import your specific HeteroADRModel from Phase 2 later
# from Scripts.Phase2_GNN_Training.train_job import HeteroADRModel 

def load_medical_dictionaries(data_dir='Data'):
    """
    Loads the dictionaries needed to translate raw GNN node IDs
    back into human-readable medical terms (Drugs, Proteins, Diseases).
    """
    print("📚 Loading Medical Dictionaries from Knowledge Graph...")
    
    try:
        # Load the main nodes file (which has standard IDs and names)
        nodes_df = pd.read_csv(os.path.join(data_dir, 'nodes.csv'))
        
        # Create a dictionary mapping the integer Node ID -> Actual Biological Name
        id_to_name = dict(zip(nodes_df['node_index'], nodes_df['name']))
        node_types = dict(zip(nodes_df['node_index'], nodes_df['type']))
        
        print(f"✅ Successfully loaded mapping for {len(id_to_name)} biological entities.")
        return id_to_name, node_types
        
    except Exception as e:
        print(f"⚠️ Error loading dictionaries. Check your nodes.csv file structure: {e}")
        return None, None

def setup_explainer(model):
    """
    Configures the GNNExplainer to find the most important 
    biological pathways (edges) for a given prediction.
    """
    print("🚀 Initializing Phase 3 GNNExplainer...")
    
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes', # Explains the ClinicalBERT/SMILES features
        edge_mask_type='object',     # Explains the specific biological edges
        model_config=dict(
            mode='binary_classification',
            task_level='edge',       # We are explaining a specific Drug-Drug side effect link
            return_type='probs',
        ),
    )
    return explainer

def run_explanation(explainer, data, target_edge_index, id_to_name):
    print(f"🔍 Analyzing biological pathways for prediction at edge index: {target_edge_index}")
    
    # Generate the Explanation mask
    explanation = explainer(data.x_dict, data.edge_index_dict, index=target_edge_index)
    
    print("✅ Explanation mask generated! Ready for clinical translation.")
    return explanation

if __name__ == "__main__":
    print("--- MTP Phase 3: Explainable AI Engine ---")
    
    # 1. Load the Dictionaries
    id_to_name, node_types = load_medical_dictionaries(data_dir='Data')
    
    # 2. Verify Graph Data Exists
    graph_path = os.path.join('Data', 'MTP_Graph.pt')
    if os.path.exists(graph_path):
        print(f"📂 Verified graph data structure at {graph_path}")
    else:
        print("⚠️ Graph data not found. Check path.")

    print("⏳ System Ready: Waiting to integrate Param Rudra model weights...")