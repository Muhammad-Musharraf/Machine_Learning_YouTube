<div align="center">

# 🤖 Machine Learning YouTube Tutorials

### A complete, hands-on Machine Learning repository — companion code for the YouTube series by Muhammad Musharraf

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

> *From linear regression to ensemble methods and unsupervised learning — structured notebooks, handwritten notes, and clean code, all in one place.*

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Topics Covered](#-topics-covered)
  - [Supervised Learning](#-supervised-learning)
  - [Unsupervised Learning](#-unsupervised-learning)
  - [Ensemble Methods](#-ensemble-methods)
  - [Imbalanced Data](#-imbalanced-data)
  - [ML Notes & PDFs](#-ml-notes--pdfs)
- [Getting Started](#-getting-started)
- [Prerequisites](#-prerequisites)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Overview

This repository is the official companion codebase for the **Machine Learning YouTube tutorial series** by **Muhammad Musharraf**. Every notebook maps directly to a video episode and is written to be:

- **Self-contained** — each notebook runs independently with minimal setup
- **Beginner-friendly** — step-by-step code with inline explanations
- **Comprehensive** — spans the full ML spectrum: preprocessing → modeling → evaluation
- **Google Colab ready** — open any notebook in Colab and run it immediately

Whether you're taking your first steps in Machine Learning or brushing up on advanced ensemble techniques, this series provides structured, practical content at every level.

---

## 📁 Repository Structure

```
Machine_Learning_YouTube/
│
├── 📂 Supervised Machine Learning Topics/
│   ├── Simple Linear Regression
│   ├── Multi-Linear Regression
│   ├── Polynomial Regression
│   ├── Regularization (Ridge / Lasso)
│   ├── Logistic Regression (Binary & Multi-class)
│   ├── Decision Tree (Classification & Regression)
│   ├── K-Nearest Neighbours (Classification & Regression)
│   ├── Support Vector Machine / SVR
│   ├── Naive Bayes
│   ├── Hyperparameter Tuning
│   ├── Cross Validation
│   ├── Confusion Matrix
│   ├── Encoding (Label, OneHot, Ordinal)
│   ├── Feature Scaling (Normalization, Standardization)
│   ├── Transformers (Column, Function, Power)
│   ├── Outlier Detection & Removal
│   └── Duplicate Removal & Type Casting
│
├── 📂 UnSupervised Learning/
│   ├── Clustering/
│   │   ├── K-Means Clustering
│   │   ├── Hierarchical Clustering
│   │   ├── DBSCAN Clustering
│   │   └── Silhouette Score
│   ├── Association/
│   │   ├── Apriori Algorithm
│   │   └── FP-Growth Algorithm
│   └── PCA/
│       └── Principal Component Analysis
│
├── 📂 Others/
│   ├── Ensemble Learning/
│   │   ├── Max Voting
│   │   ├── Averaging & Weighted Average (Classification)
│   │   └── Averaging & Weighted Average (Regression)
│   └── Bagging/
│       ├── Bagging Meta-Estimator + Random Forest (Classification)
│       └── Bagging Meta-Estimator + Random Forest (Regression)
│
├── 📂 Imbalanced DATA/
│   └── Imbalanced Data Handling (XGBoost, SMOTE, class_weight)
│
├── 📂 ML Notes/
│   ├── Machine Learning Handwritten Notes.pdf
│   ├── Machine Learning Notes.pdf
│   ├── Imbalanced Data Techniques Document.pdf
│   └── Unsupervised Learning/
│       ├── Clustering_Unsupervised_Learning.pdf
│       ├── Association Learning Document.pdf
│       └── PCA.pdf
│
└── README.md
```

---

## 📚 Topics Covered

### 🔷 Supervised Learning

A complete walkthrough of supervised ML algorithms — from foundational regression to advanced classifiers. Each notebook covers theory, implementation, and evaluation.

| Notebook | Topic | Key Concepts |
|----------|-------|-------------|
| `Simple Linear Regression` | Regression | OLS, slope/intercept, R² |
| `Multi-Linear Regression` | Regression | Multivariate inputs, coefficients |
| `Polynomial Regression` | Regression | Feature transformation, overfitting |
| `Regularization` | Regression | Ridge (L2), Lasso (L1), ElasticNet |
| `Logistic Regression (Binary)` | Classification | Sigmoid, decision boundary |
| `Logistic Regression (Multi-class)` | Classification | One-vs-Rest, softmax |
| `Logistic Regression (Polynomial)` | Classification | Non-linear boundaries |
| `Decision Tree` | Classification | Gini impurity, pruning |
| `Decision Tree (Regression)` | Regression | MSE split criterion |
| `KNN (Classification)` | Classification | Distance metrics, k selection |
| `KNN (Regression)` | Regression | Neighbor averaging |
| `SVM (Classification)` | Classification | Hyperplane, margin, kernels |
| `SVM (Polynomial Kernel)` | Classification | RBF, poly kernel tricks |
| `SVR (Regression)` | Regression | Epsilon-insensitive loss |
| `Naive Bayes` | Classification | Bayes theorem, conditional probability |
| `Hyperparameter Tuning` | Model Optimization | GridSearchCV, RandomizedSearchCV |
| `Cross Validation` | Model Evaluation | K-Fold, StratifiedKFold |
| `Confusion Matrix` | Evaluation Metrics | Precision, Recall, F1, ROC |

**Preprocessing & Feature Engineering:**

| Notebook | Topic |
|----------|-------|
| `Label Encoding` | Ordinal categorical to integer |
| `OneHot Encoding` | Nominal categorical to binary |
| `Ordinal Encoding` | Custom-ordered categories |
| `Normalization` | Min-Max scaling |
| `Standardization` | Z-score scaling |
| `Column Transformer` | Heterogeneous feature pipelines |
| `Function Transformer` | Custom feature transformations |
| `Power Transformer` | Box-Cox / Yeo-Johnson |
| `Outlier Detection` | IQR, Z-score, visualization |
| `Duplicate Removal & Type Casting` | Data cleaning essentials |

---

### 🔶 Unsupervised Learning

#### Clustering

| Notebook | Algorithm | Key Concepts |
|----------|-----------|-------------|
| `K-Means Clustering` | K-Means | Centroids, inertia, Elbow method |
| `Hierarchical Clustering` | Agglomerative | Dendrograms, linkage methods |
| `DBSCAN Clustering` | Density-Based | Epsilon, min_samples, noise points |
| `Silhouette Score` | Evaluation | Cluster quality metric |

#### Association Rule Learning

| Notebook | Algorithm | Key Concepts |
|----------|-----------|-------------|
| `Apriori Algorithm` | Apriori | Support, confidence, lift |
| `FP-Growth Algorithm` | FP-Growth | Frequent Pattern trees, scalability |

#### Dimensionality Reduction

| Notebook | Algorithm | Key Concepts |
|----------|-----------|-------------|
| `PCA` | Principal Component Analysis | Variance explained, eigenvectors, scree plot |

---

### 🟣 Ensemble Methods

| Notebook | Technique | Task |
|----------|-----------|------|
| `Max Voting` | Hard Voting Classifier | Classification |
| `Averaging & Weighted Average` | Soft Voting | Classification & Regression |
| `Bagging + Random Forest` | Bootstrap Aggregating | Classification |
| `Bagging + Random Forest` | Bootstrap Aggregating | Regression |

---

### 🔴 Imbalanced Data

| Notebook | Techniques Covered |
|----------|--------------------|
| `Imbalanced Data Handling` | `class_weight`, SMOTE oversampling, XGBoost `scale_pos_weight`, ROC-AUC evaluation |

**Strategies demonstrated:**
- Class weight adjustment
- Oversampling with SMOTE (`imbalanced-learn`)
- Undersampling techniques
- Evaluation using ROC curve and AUC score instead of accuracy

---

### 📄 ML Notes & PDFs

Curated study notes to supplement the code:

| File | Contents |
|------|----------|
| `Machine Learning Handwritten Notes.pdf` | Concept explanations written by hand — great for revision |
| `Machine Learning Notes.pdf` | Typed, structured ML theory reference |
| `Imbalanced Data Techniques Document.pdf` | Deep-dive into handling class imbalance |
| `Clustering_Unsupervised_Learning.pdf` | Clustering algorithms: theory and math |
| `Association Learning Document.pdf` | Apriori and FP-Growth explained |
| `PCA.pdf` | Linear algebra behind Principal Component Analysis |

---

## 🛠️ Getting Started

### Option A — Google Colab *(Recommended for beginners)*

1. Open any `.ipynb` file on GitHub
2. Click **"Open in Colab"** at the top of the notebook
3. Go to `Runtime → Run all`

No installation needed. Runs entirely in the browser with free GPU access.

---

### Option B — Local Setup

**1. Clone the repository**

```bash
git clone https://github.com/Muhammad-Musharraf/Machine_Learning_YouTube.git
cd Machine_Learning_YouTube
```

**2. Create a virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Launch Jupyter Notebook**

```bash
jupyter notebook
```

Then open any notebook from the browser interface.

---

## 📦 Prerequisites

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computing |
| `pandas` | Data manipulation |
| `matplotlib` | Plotting & visualization |
| `seaborn` | Statistical data visualization |
| `scikit-learn` | Core ML algorithms |
| `xgboost` | Gradient boosting |
| `imbalanced-learn` | SMOTE & resampling tools |
| `tensorflow` / `keras` | Deep learning |
| `jupyter` | Notebook environment |

Install all at once:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn tensorflow jupyter
```

---

## 🗺️ Learning Roadmap

Follow this sequence if you're starting from scratch:

```
1. Data Cleaning & Preprocessing
      ↓
2. Feature Encoding & Scaling
      ↓
3. Supervised Learning — Regression
      ↓
4. Supervised Learning — Classification
      ↓
5. Model Evaluation (Cross-Validation, Confusion Matrix, ROC)
      ↓
6. Hyperparameter Tuning
      ↓
7. Ensemble Methods (Voting → Bagging → Random Forest)
      ↓
8. Imbalanced Data Handling
      ↓
9. Unsupervised Learning — Clustering & Association
      ↓
10. Dimensionality Reduction (PCA)
```

---

## 🤝 Contributing

Contributions are welcome! If you'd like to fix a bug, improve a notebook, or add a new topic:

1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/your-topic`
3. **Commit** your changes: `git commit -m "Add: topic description"`
4. **Push**: `git push origin feature/your-topic`
5. **Open a Pull Request** with a clear description

Please keep notebooks clean, well-commented, and consistent with the existing style.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). You are free to use, adapt, and share the code with attribution.

---

## 👨‍💻 Author

<div align="center">

**Muhammad Musharraf**
*Machine Learning Engineer & Educator*

[![GitHub](https://img.shields.io/badge/GitHub-Muhammad--Musharraf-181717?style=for-the-badge&logo=github)](https://github.com/Muhammad-Musharraf)
[![YouTube](https://img.shields.io/badge/YouTube-Subscribe-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://youtube.com)

</div>

---

<div align="center">

### ⭐ If this helped you, give the repo a star — it helps others discover it!

*Built with ❤️ for the ML community*

</div>
