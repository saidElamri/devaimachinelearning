<div align="center">
  <br />
  <img src="https://www.simplon.ma/images/Simplon_Maghreb_Rouge.png" alt="Simplon Maghreb Logo" width="300"/>
  <br /><br />

  <div>
    <img src="https://img.shields.io/badge/-Python-black?style=for-the-badge&logo=python&logoColor=white&color=3776AB" />
    <img src="https://img.shields.io/badge/-Pandas-black?style=for-the-badge&logo=pandas&logoColor=white&color=150458" />
    <img src="https://img.shields.io/badge/-NumPy-black?style=for-the-badge&logo=numpy&logoColor=white&color=013243" />
    <img src="https://img.shields.io/badge/-Scikit--Learn-black?style=for-the-badge&logo=scikitlearn&logoColor=white&color=F7931E" />
    <img src="https://img.shields.io/badge/-Matplotlib-black?style=for-the-badge&logo=plotly&logoColor=white&color=11557C" />
    <img src="https://img.shields.io/badge/-Seaborn-black?style=for-the-badge&logoColor=white&color=4C9F70" />
    <img src="https://img.shields.io/badge/-Jupyter-black?style=for-the-badge&logo=jupyter&logoColor=white&color=F37626" />
    <img src="https://img.shields.io/badge/-Git-black?style=for-the-badge&logo=git&logoColor=white&color=F05032" />
    <img src="https://img.shields.io/badge/-Jira-black?style=for-the-badge&logo=jira&logoColor=white&color=0052CC" />
  </div>

  <h1>📊 Rapport Technique – Prédiction du Churn Client</h1>
  <p><strong>Projet ML</strong> – Simplon Maghreb</p>
</div>

---

## 🧩 1. Introduction

L’objectif de ce projet est de construire un modèle de Machine Learning capable de prédire le **churn (désabonnement des clients)** à partir d’un ensemble de données clients.  
Deux modèles ont été comparés : la **Régression Logistique** et le **Support Vector Machine (SVM)**.

---

## ⚙️ 2. Modèles Testés

Les deux modèles ont été entraînés sur les mêmes données, avec une séparation **train/test de 80/20**.  
Les métriques principales utilisées pour la comparaison sont :

- **Accuracy** : taux de prédictions correctes globales  
- **Precision** : proportion de prédictions positives correctes  
- **Recall** : capacité à détecter les vrais positifs  
- **F1 Score** : moyenne harmonique entre précision et rappel  
- **AUC (Area Under Curve)** : mesure globale de performance du modèle

---

## 📈 3. Résultats Obtenus

| Modèle                | Accuracy | Precision | Recall | F1 Score | AUC   |
|-----------------------|-----------|------------|---------|-----------|--------|
| **Régression Logistique** | **0.8176** | 0.6824     | **0.5818** | **0.6281** | **0.8607** |
| SVM (SVC)             | 0.8119     | **0.6957** | 0.5147  | 0.5917   | 0.8463 |

### 🔍 Analyse des Résultats

- La **Régression Logistique** offre **une meilleure Recall et F1 Score**, ce qui signifie qu’elle identifie mieux les clients susceptibles de se désabonner.  
- Le **SVM** obtient une précision légèrement supérieure, mais détecte moins bien les cas de churn.  
- En termes de **AUC (0.8607 vs 0.8463)**, la régression logistique montre une **meilleure capacité de distinction** entre les classes.

---

## 📊 4. Visualisations et Interprétation

### 🔹 Matrice de Confusion
- Pour la **Régression Logistique**, la majorité des clients fidèles (classe 0) sont bien prédits, et environ **58 % des churners** (classe 1) sont correctement identifiés.  
- Le **SVM** confond davantage les churners avec les clients fidèles, ce qui diminue son rappel.

### 🔹 Courbe ROC
- La **courbe ROC** de la Régression Logistique se situe légèrement au-dessus de celle du SVM.  
- Cela confirme que le modèle est **plus robuste** sur l’ensemble des seuils de décision.

---

## ✅ 5. Justification du Modèle Retenu

Le modèle retenu pour la mise en production est la **Régression Logistique**, pour les raisons suivantes :

1. **Performance globale supérieure** sur les métriques importantes (Recall, F1 Score, AUC).  
2. **Interprétabilité facile** : chaque coefficient peut être interprété pour comprendre l’influence d’une variable sur la probabilité de churn.  
3. **Simplicité et rapidité d’entraînement**, adaptée à une première mise en production.  
4. **Bonne généralisation** sur les données de test.

---

## 🏁 6. Conclusion

La **Régression Logistique** est le meilleur compromis entre **performance**, **interprétabilité** et **stabilité**.  
Elle sera utilisée comme **modèle principal** pour la mise en production, tout en laissant la possibilité d’explorer d’autres modèles plus complexes (**Random Forest**, **XGBoost**) dans des itérations futures.

---

<div align="center">
  <p >👨‍💻 Projet réalisé par <strong> <a href='https://github.com/saidElamri'> said Elamri </a></strong> | Simplon Maghreb</p>
</div>