import matplotlib.pyplot as plt
import seaborn as sns

# 1. The Exact Data from your Ablation Study Output
pathways = [
    "effect/phenotype -> drug (Path A)",
    "disease -> effect/phenotype (Path A)",
    "effect/phenotype -> drug (Path B)",
    "disease -> effect/phenotype (Path B)",
    "effect/phenotype -> drug (Path C)"
]
logit_drops = [65.3998, 52.3265, 47.2101, 46.8358, 45.5336]
rel_power = [1.0000, 0.8001, 0.7219, 0.7161, 0.6962]

# 2. Set up a beautiful, publication-ready aesthetic
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Define colors based on the pathway type
colors = ['#4C72B0' if 'disease' in p else '#55A868' for p in pathways]

# 3. Create the Horizontal Bar Chart
bars = plt.barh(pathways, logit_drops, color=colors, edgecolor='black', height=0.6)

# 4. Add the labels and titles
plt.xlabel("Importance to Model (Logit Drop)", fontsize=12, fontweight='bold')
plt.title("Edge-Occlusion Ablation Study: Top 5 Biological Pathways\nPredicting Drug-Induced Thrombocytopenia", 
          fontsize=14, fontweight='bold', pad=15)

# 5. Add the exact numbers to the end of each bar
for bar, power, logit in zip(bars, rel_power, logit_drops):
    plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
             f"Drop: {logit:.1f} (Rel: {power:.2f})", 
             va='center', ha='left', fontsize=10, color='black')

# 6. Clean up the look
plt.gca().invert_yaxis() # Put the #1 result at the top
plt.xlim(0, max(logit_drops) + 15) # Leave room for the text
plt.tight_layout()

# 7. Save it in high resolution
plt.savefig("results/ablation_results.png", dpi=300, bbox_inches='tight')
print("Saved publication chart as 'results/ablation_results.png'!")