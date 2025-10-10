<div align="center">
  <img src="https://www.simplon.ma/images/Simplon_Maghreb_Rouge.png" alt="Simplon Maghreb Logo" width="300" />
  <br /><br />
  <h1>🔍 Customer Churn Prediction – Machine Learning Project</h1>
  <p>Projet ML réalisé dans le cadre du parcours <strong>Simplon Maghreb</strong></p>
</div>

---

## 🧠 Overview

Le but de ce projet est de développer un modèle de **Machine Learning** capable de prédire si un client est susceptible de **se désabonner (churn)**.  
Le jeu de données contient des informations démographiques, contractuelles et comportementales sur les clients d’un opérateur télécom.

Deux modèles ont été comparés :

- 🔹 **Régression Logistique**
- 🔹 **Support Vector Machine (SVM)**

---

## 🧰 Technologies Utilisées

<p align="center">
  <img src="https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/-Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/-NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/-Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/-Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white"/>
  <img src="https://img.shields.io/badge/-Seaborn-4C9F70?style=for-the-badge&logoColor=white"/>
  <img src="https://img.shields.io/badge/-Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
</p>

---

## 📊 Méthodologie

1. **Chargement et nettoyage des données**
   - Suppression de la colonne `customerID`
   - Encodage des variables catégorielles avec `LabelEncoder`
   - Conversion de la variable cible `Churn` en valeurs numériques (1 = Yes, 0 = No)

2. **Entraînement des modèles**
   - Répartition des données en **train/test (80/20)**
   - Entraînement des deux modèles : Régression Logistique et SVM
   - Évaluation avec les métriques : Accuracy, Precision, Recall, F1 Score, AUC

3. **Validation croisée**
   - Utilisation de la méthode **K-Fold (k=5)** pour mesurer la stabilité du modèle

---

## 📈 Résultats Comparatifs

| Modèle                | Accuracy | Precision | Recall | F1 Score | AUC   |
|-----------------------|-----------|------------|---------|-----------|--------|
| **Régression Logistique** | **0.8176** | 0.6824     | **0.5818** | **0.6281** | **0.8607** |
| **SVM (SVC)**         | 0.8119     | **0.6957** | 0.5147  | 0.5917   | 0.8463 |

### 🔍 Interprétation
- La **Régression Logistique** surpasse le SVM en **Recall** et **F1 Score**, ce qui est crucial pour détecter les churners.  
- Le **SVM** offre une précision légèrement supérieure, mais identifie moins bien les clients à risque.  
- Le score **AUC (0.8607)** confirme la robustesse de la Régression Logistique.

---

## 📉 Visualisation des Résultats

<p align="center">
  <img src="images/confusion_matrix.png" width="400" alt="Confusion Matrix"/>
  <img src="images/roc_curve.png" width="400" alt="ROC Curve"/>
</p>

---

## 🏆 Modèle Retenu : Régression Logistique

### Pourquoi ?
- ✅ Meilleure performance globale (Recall, F1, AUC)  
- ✅ Modèle simple, rapide à entraîner et à interpréter  
- ✅ Idéal pour une première mise en production  

---

## 🚀 Perspectives d’Amélioration

- Expérimenter avec des modèles avancés : **Random Forest**, **XGBoost**, **LightGBM**  
- Optimisation des hyperparamètres (GridSearchCV)  
- Intégration dans une application **Streamlit** pour des prédictions interactives  
- Surveillance continue des performances du modèle en production  

---

## 📚 Structure du Projet

