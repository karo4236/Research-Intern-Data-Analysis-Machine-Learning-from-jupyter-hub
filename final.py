import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import UMAP
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

# Load CYMO feature vectors for real and synthetic text
real_data = pd.read_csv("cymo_features_real.csv")      # shape: [n_samples, n_features]
synthetic_data = pd.read_csv("cymo_features_synth.csv")  # same shape, same columns

# Compute UMAP for structure visualization
def plot_umap(real, synth):
    combined = pd.concat([real, synth])
    labels = ['Real'] * len(real) + ['Synthetic'] * len(synth)
    reducer = UMAP(random_state=42)
    embedding = reducer.fit_transform(combined)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=labels, alpha=0.6)
    plt.title("UMAP Projection of CYMO Features")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.legend(); plt.tight_layout()
    plt.show()

plot_umap(real_data, synthetic_data)

# Compute feature-wise JS and Wasserstein distances
def compute_divergence_metrics(real, synth):
    js_divergences = {}
    wass_distances = {}
    for col in real.columns:
        real_col = real[col].dropna().values
        synth_col = synth[col].dropna().values

        # Normalize histograms for JS
        hist_real, _ = np.histogram(real_col, bins=50, range=(min(real_col.min(), synth_col.min()), max(real_col.max(), synth_col.max())), density=True)
        hist_synth, _ = np.histogram(synth_col, bins=50, range=(min(real_col.min(), synth_col.min()), max(real_col.max(), synth_col.max())), density=True)

        # Add small constant to avoid log(0)
        js = jensenshannon(hist_real + 1e-8, hist_synth + 1e-8)
        wd = wasserstein_distance(real_col, synth_col)

        js_divergences[col] = js
        wass_distances[col] = wd
    return js_divergences, wass_distances

js_scores, wd_scores = compute_divergence_metrics(real_data, synthetic_data)

# Show top 10 features with highest divergence
divergence_df = pd.DataFrame({
    'Feature': list(js_scores.keys()),
    'JS Divergence': list(js_scores.values()),
    'Wasserstein Distance': list(wd_scores.values())
}).sort_values(by='JS Divergence', ascending=False)

print(divergence_df.head(10))

# Density plots for top diverging features
def plot_density_comparison(real, synth, top_features, n=4):
    for feature in top_features[:n]:
        plt.figure(figsize=(6, 4))
        sns.kdeplot(real[feature], label="Real", fill=True, alpha=0.5)
        sns.kdeplot(synth[feature], label="Synthetic", fill=True, alpha=0.5)
        plt.title(f'Density Plot - {feature}')
        plt.legend()
        plt.tight_layout()
        plt.show()

plot_density_comparison(real_data, synthetic_data, divergence_df['Feature'].tolist())
