import pandas as pd
import gc
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

print("--- 🧬 Initializing ClinicalBERT Feature Encoder ---")

# 1. Load the pre-trained Clinical Text Model
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
print(f"Downloading/Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Use GPU if available (Supercomputer prep)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# 2. Load the Raw Data
data_path = './Data/drug_features.csv'
print(f"Loading raw clinical data from {data_path}...")
df = pd.read_csv(data_path)

# Fill empty text with a blank space to prevent errors
df['mechanism_of_action'] = df['mechanism_of_action'].fillna("")
df['description'] = df['description'].fillna("")

# 3. Combine text columns into a single rich clinical profile
df['clinical_text'] = df['description'] + " " + df['mechanism_of_action']

# 4. Encoding Loop
batch_size = 8 # Pushing the laptop safely!
all_embeddings = []

print("Encoding text into 768-dimensional biological vectors...")
with torch.no_grad():
    for i in tqdm(range(0, len(df), batch_size)):
        batch_texts = df['clinical_text'].iloc[i:i+batch_size].tolist()
        
        # Tokenize the text
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        
        # Pass through BioBERT
        outputs = model(**inputs)
        
        # We take the [CLS] token embedding
        embeddings = outputs.last_hidden_state[:, 0, :].cpu()
        all_embeddings.append(embeddings)
        
        # --- THE FIX: Force RAM Cleanup after every batch ---
        del inputs
        del outputs
        gc.collect()

# 5. Save the final features
final_tensor = torch.cat(all_embeddings, dim=0)
save_path = 'Data/real_drug_features.pt'
torch.save(final_tensor, save_path)

print(f"\n✅ Success! Encoded {final_tensor.size(0)} drugs into {final_tensor.size(1)}-dim features.")
print(f"Saved to {save_path}. Ready for the Supercomputer!")