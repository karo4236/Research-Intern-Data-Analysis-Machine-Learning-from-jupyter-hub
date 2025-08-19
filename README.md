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

## **Directory Structure**

│
├─ notebooks/ # Jupyter notebooks
├─ scripts/ # Python scripts for analysis
├─ data/ # Processed data files (small)
├─ README.md
├─ requirements.txt # Dependencies
└─ .gitignore # Ignored files (venv, checkpoints, etc.)

