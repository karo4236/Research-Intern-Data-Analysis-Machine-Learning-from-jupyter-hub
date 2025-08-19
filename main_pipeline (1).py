import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# Step 1: Load CYMO Datasets
# ================================
def load_cymo_datasets():
    file_map = {
        'inf_fewshot_7B': 'ann.few_shot_persona_2077comments_20250717_181329_7b.csv',
        'inf_fewshot_70B': 'ann.few_shot_persona_2077comments_20250717_204349_70b.csv',
        'atb_fewshot_mdd_70B': 'ann.usingfewshotmdd_persona2077comments_70b_20250718_051344.csv',
        'atb_zeroshot_mdd_70B': 'ann.usingmdd_persona2077comments_atb_zero_70b20250718_034747.csv',
        'inf_zeroshot_7B': 'ann.zero_shot_persona_2077comments_20250717_173356_7b.csv',
        'inf_zeroshot_70B': 'ann.zero_shot_persona_2077comments_20250717_194411_70b.csv',
        'atb_zeroshot_7B': 'ann.atb_zero_7b.csv',
    }

    datasets = {}
    for key, path in file_map.items():
        df = pd.read_csv(path)
        datasets[key] = df.head(2077)  # Trim to 2077 rows
    return datasets

# ================================
# Step 2: Fidelity Evaluation
# ================================
def evaluate_distribution_alignment(datasets):
    if 'real' not in datasets:
        print(" No real dataset found for comparison.")
        return

    real = datasets['real'].select_dtypes(include=[np.number])

    for name, df in datasets.items():
        if name == 'real':
            continue

        synth = df.select_dtypes(include=[np.number])
        print(f"\n▶ Distributional Alignment for: {name}")

        for col in real.columns.intersection(synth.columns):
            try:
                js = jensenshannon(real[col], synth[col])
                wd = wasserstein_distance(real[col], synth[col])
                print(f"{col}: JS={js:.4f}, Wasserstein={wd:.4f}")
            except Exception:
                continue

# ================================
# Step 3: UMAP Visualization
# ================================
def plot_umap(df, title):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print(f" No numeric features in {title}")
        return
    scaled = StandardScaler().fit_transform(numeric_df)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(scaled)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1])
    plt.title(f't-SNE for {title}')
    plt.tight_layout()
    plt.show()

# ================================
# Step 4: Classifier Indistinguishability
# ================================
def train_indistinguishability_classifier(real_df, synth_df, name):
    real = real_df.select_dtypes(include=[np.number]).copy()
    synth = synth_df.select_dtypes(include=[np.number]).copy()
    real['label'] = 0
    synth['label'] = 1

    combined = pd.concat([real, synth])
    combined = shuffle(combined, random_state=42)

    X = combined.drop(columns=['label'])
    y = combined['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"\n▶ Classifier Evaluation (Real vs. {name})")
    print(classification_report(y_test, y_pred, digits=4))

# ================================
# Step 5: Mental Health Detection
# ================================
def train_mh_model(X_train, y_train, X_test, y_test, title=""):
    # Filter numeric only
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

    # Align feature columns
    X_train, X_test = X_train.align(X_test, join="inner", axis=1)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f" {title} → Accuracy: {acc:.4f} | F1: {f1:.4f}")

# ================================
# Main Runner
# ================================
if __name__ == "__main__":
    datasets = load_cymo_datasets()

    # Add real dataset
    if 'real' not in datasets:
        real_path = 'real_cymo.csv'
        if os.path.exists(real_path):
            datasets['real'] = pd.read_csv(real_path).head(2077)
        else:
            print(" 'real_cymo.csv' not found.")

    # Fidelity: Distribution metrics
    evaluate_distribution_alignment(datasets)

    # Fidelity: UMAP and Classifier Evaluation
    for name, df in datasets.items():
        if name != 'real':
            plot_umap(df, title=name)
            train_indistinguishability_classifier(datasets['real'], df, name)

    # Utility: MH model training (logistic baseline)
    target = 'label'
    real_df = datasets['real']
    for name, df in datasets.items():
        if name == 'real':
            continue

        # Use dummy labels if label not present
        if target not in df.columns:
            df[target] = np.random.randint(0, 2, size=len(df))
        if target not in real_df.columns:
            real_df[target] = np.random.randint(0, 2, size=len(real_df))

        X_train = df.drop(columns=[target])
        y_train = df[target]
        X_test = real_df.drop(columns=[target])
        y_test = real_df[target]

        train_mh_model(X_train, y_train, X_test, y_test, title=f"MH Detection on {name}")
