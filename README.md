# Research Intern: Data Analysis & Machine Learning

This repository contains the code and analyses developed during the research internship focused on synthetic data evaluation, sociodemographic attribute detection, and downstream language-based mental health classification.

---

## **Project Overview**

The project focuses on:

1. **Sociodemographic Attribute Detection**  
   - Detect age, gender, and education for Reddit users.
   - Use both **pattern-based** (explicit and implicit) and **automatic estimation** methods.
   - Generate distributional comparisons to validate results.

2. **Synthetic Data Evaluation**  
   - Generate **8 synthetic datasets**:
     - Persona Definition: Attribute-Controlled vs. Inferred
     - Prompting Setup: Zero-shot vs. Few-shot
     - Model Size: 7B vs. 70B
   - Evaluate **fidelity and realism**:
     - UMAP (global structure)
     - Jensen-Shannon Divergence, Wasserstein Distance (local structure)
     - Feature-wise density comparisons

3. **Classifier-Based Indistinguishability**  
   - Train a **binary classifier** to discriminate authentic vs. synthetic texts.
   - Compute **Accuracy, Precision, Recall, F1 Score**.
   - Inspect feature importance using **CYMO features**.

4. **Utility Evaluation for Mental Health Detection**  
   - Train models on:
     1. Authentic data only  
     2. Synthetic data only  
     3. Authentic + Synthetic (Data Augmentation)  
   - Evaluate **effect of dataset size** and **trait-specific debiasing**.
   - Analyze errors specifically for **underrepresented sociodemographic groups**.

---

## Directory Structure

```text
llama-local/
├── notebooks/        # Jupyter notebooks
├── scripts/          # Python scripts for analysis
├── data/             # Processed data files (small)
├── README.md
├── requirements.txt  # Dependencies
└── .gitignore        # Ignored files (venv, checkpoints, etc.)

```

---

## **Setup Instructions**

1. Clone the repository:

```bash
git clone https://github.com/karo4236/Research-Intern-Data-Analysis-Machine-Learning-from-jupyter-hub.git
cd llama-local
```

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate


pip install -r requirements.txt


Usage

Notebooks in notebooks/ demonstrate:

Sociodemographic attribute detection

Synthetic data generation and evaluation

Downstream classification experiments

Scripts in scripts/ provide reusable modules for:

Feature extraction using CYMO

Training classifiers

Generating evaluation metrics

References

Giuffrè, M., & Shung, D. L. (2023). Harnessing the power of synthetic data in healthcare: innovation, application, and privacy. NPJ Digital Medicine, 6(1), 186.

Chen, R. J., et al. (2021). Synthetic data in machine learning for medicine and healthcare. Nature Biomedical Engineering, 5(6), 493–497.

Smolyak, D., et al. (2024). Large language models and synthetic health data: progress and prospects. JAMIA Open, 7(4), ooae114.

Woo, E.G., et al. (2025). Synthetic data distillation enables the extraction of clinical information at scale. NPJ Digital Medicine, 8, 267.


