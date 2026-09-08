# 🤖 ML-Library

> A personal Machine Learning & Artificial Intelligence learning library.

Dépôt personnel dédié à l'étude, l'implémentation et l'expérimentation en **Machine Learning** et **Intelligence Artificielle**.

Ce repository n'est pas une roadmap de choses à apprendre : c'est la **trace structurée de ce que j'ai étudié et implémenté**, des fondamentaux du ML jusqu'au Deep Learning, aux modèles génératifs, au Reinforcement Learning et à la mise en application (AI Engineering).

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

---

## 📂 Structure du dépôt

```text
ML-Library/
├── 0_intro_ml/          Fondamentaux du Machine Learning
├── 1_python_ml/         Python appliqué au ML (NumPy, Pandas, Matplotlib)
├── 2_deep_learning/     Bases des réseaux de neurones
├── 3_ai_engineering/    ML avancé + Deep Learning appliqué (21 modules)
├── 4_ml_algorithms/     Parcours principal d'implémentation par famille d'algorithmes
└── 5_projects/          Projets de bout en bout
```

---

## 📚 Contenu détaillé

### 0️⃣ [`0_intro_ml`](./0_intro_ml) — Introduction to Machine Learning

Notions fondamentales : data preprocessing, feature scaling, régression linéaire / multiple / polynomiale, régression logistique, SVM et Kernel SVM, K-Means, classification, clustering, évaluation de modèles.

### 1️⃣ [`1_python_ml`](./1_python_ml) — Python for Machine Learning

Manipulation de données et workflows ML en Python : NumPy, Pandas, Matplotlib, préparation de données, visualisation, premiers modèles de régression et de classification sous Jupyter.

### 2️⃣ [`2_deep_learning`](./2_deep_learning) — Deep Learning Foundations

Réseaux de neurones artificiels, MLP, forward propagation, backpropagation, descente de gradient, premiers CNN.

### 3️⃣ [`3_ai_engineering`](./3_ai_engineering) — ML avancé & Deep Learning appliqué

La section la plus dense du dépôt (21 modules progressifs), qui va des techniques d'ensemble jusqu'aux modèles de langage :

| Bloc | Modules |
|---|---|
| **Ensemble Learning** | Ensemble Learning Basic, Bagging, Boosting, Boosting Extreme, Boosting ++ |
| **Robustesse & évaluation** | Données déséquilibrées, régularisation, cross-validation, optimisation et ajustement d'un modèle final, ingénierie des caractéristiques |
| **Deep Learning appliqué** | Theory Deep Learning, CIFAR-10 (TensorFlow), CIFAR-100, CNN avec PyTorch |
| **NLP & séquences** | RNNs, Seq2Seq, Sentiment Analysis |
| **Transformers & LLM** | Transformers, BERT & GPT, projet Transformers |
| **Transfer Learning** | Transfer Learning in Vision, Transfer Learning in NLP |

### 4️⃣ [`4_ml_algorithms`](./4_ml_algorithms) — Machine Learning Algorithms

Parcours principal d'implémentation, organisé par famille d'apprentissage :

- **1. Supervisé** — Régression · Classification · Ensemble Learning
- **2. Non-supervisé** — Clustering · Réduction de dimensionnalité
- **3. Semi-supervisé** — Self-Training · Co-Training · Graph-Based Semi-Supervised Learning
- **4. Anomaly Detection** — Détection d'anomalies
- **5. Deep Learning** — MLP · CNN · RNN · Transformers · Variational Autoencoders · GAN · Diffusion Models
- **6. Reinforcement Learning** — Value-Based Methods · Policy-Based Methods (REINFORCE) · Actor-Critic

### 5️⃣ [`5_projects`](./5_projects) — Projets

Projets de bout en bout mettant en pratique les notions des sections précédentes :

| # | Projet | Domaine |
|---|---|---|
| 1 | Système de recommandation de films | Recommender Systems |
| 2 | Détection de spam | NLP / Classification |
| 3 | Assistant vocal | Speech / NLP |
| 4 | Reconnaissance faciale | Computer Vision |
| 5 | Chatbot | NLP |
| 6 | Détection d'objets | Computer Vision |
| 7 | Traducteur | NLP / Seq2Seq |
| 8 | Chatbot pour l'orientation académique | NLP / LLM |

---

## 🛠️ Technologies

**Langages & outils** — Python, Jupyter Notebook
**Data & ML** — NumPy, Pandas, Matplotlib, Scikit-Learn
**Deep Learning** — TensorFlow, PyTorch

> La majorité des implémentations de Deep Learning des premières sections utilisent **TensorFlow**. **PyTorch** est introduit dans les parties plus avancées, notamment pour le Reinforcement Learning.

---

## 🚀 Utilisation

```bash
# Cloner le dépôt
git clone https://github.com/Wabaya243/ML-Library.git
cd ML-Library

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate      # Windows : .venv\Scripts\activate

# Installer les dépendances de base
pip install numpy pandas matplotlib scikit-learn jupyter tensorflow torch

# Lancer Jupyter
jupyter notebook
```

Chaque dossier est indépendant : ouvre directement le notebook du sujet qui t'intéresse.

---

## 📈 Learning Path

```text
Python
   ↓
Mathematics & Statistics
   ↓
Machine Learning
   ↓
Supervised Learning
   ↓
Unsupervised Learning
   ↓
Semi-Supervised Learning
   ↓
Anomaly Detection
   ↓
Deep Learning
   ↓
Transformers
   ↓
Generative AI
   ↓
Reinforcement Learning
   ↓
AI Engineering
   ↓
AI Projects
```

---

## 🎯 Objectif

Documenter et centraliser mon parcours technique :

**Machine Learning → Deep Learning → Generative AI → Reinforcement Learning → AI Engineering**

Ce dépôt est à la fois une **bibliothèque personnelle de connaissances**, un espace d'expérimentation et une trace de mon évolution dans le domaine de l'IA.

---

## 👤 Auteur

**Kambale Divin** — [@Wabaya243](https://github.com/Wabaya243)
Master 1 Informatique, Université de Kinshasa (UNIKIN)
