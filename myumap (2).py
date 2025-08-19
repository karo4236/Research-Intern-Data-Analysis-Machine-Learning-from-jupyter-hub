import pandas as pd
import umap
import matplotlib.pyplot as plt

# 🔹 1. Load real and synthetic CYMO files
real_df = pd.read_csv("ann.balanced_control_part1_subset.csv")
synthetic_df = pd.read_csv("ann.cleaned_mdd_inf_zero_7b.csv")

# 🔹 2. Add labels (1 = real, 0 = synthetic)
real_df["label"] = 1
synthetic_df["label"] = 0

# 🔹 3. Combine datasets
combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)

# 🔹 4. Extract features and labels
labels = combined_df["label"].values
features = combined_df.drop(columns=["label"]).values

# 🔹 5. Run UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(features)

# 🔹 6. Plot and save
plt.figure(figsize=(8,6))
plt.scatter(embedding[labels==1, 0], embedding[labels==1, 1], label='Real', alpha=0.6)
plt.scatter(embedding[labels==0, 0], embedding[labels==0, 1], label='Synthetic', alpha=0.6)
plt.legend()
plt.title("UMAP Projection of CYMO Features (Real vs. Synthetic)")
plt.savefig("umap_real_vs_synthetic.png", dpi=300, bbox_inches='tight')
plt.show()
