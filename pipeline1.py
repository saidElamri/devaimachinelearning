"""
pipeline.py

Pipeline de préparation, entraînement et évaluation pour la prédiction du churn client.
Livrables : fonctions réutilisables pour charger les données, préparer, sélectionner les features,
entraîner plusieurs modèles supervisés (LogisticRegression, RandomForest, XGBoost) et sauvegarder
les résultats et modèles.

Usage (exemple) :
    python pipeline.py --data_path data/churn.csv --target Churn --out_dir results --use_variance_threshold

Dépendances : scikit-learn, pandas, numpy, matplotlib, joblib, xgboost (optionnel)

Fonctions principales exportées :
    - load_data(path)
    - split_data(df, target, test_size, random_state)
    - build_preprocessor(numeric_cols, categorical_cols)
    - apply_variance_threshold(X, threshold)
    - train_models(X_train, y_train, models)
    - evaluate_model(model, X_test, y_test)
    - compare_and_save(models, X_test, y_test, out_dir)

Retour : CSV avec métriques pour chaque modèle, figures ROC/PR, modèles sauvegardés (.joblib)

"""

import os
import argparse
import json
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import joblib

# XGBoost is optional; if unavailable, we'll skip it gracefully
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


def load_data(path: str) -> pd.DataFrame:
    """Charge les données depuis un fichier CSV et renvoie un DataFrame pandas."""
    df = pd.read_csv(path)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def split_data(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Sépare les features et la cible, puis effectue un train/test split stratifié si possible."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe columns")

    X = df.drop(columns=[target])
    y = df[target]

    # si la variable cible n'est pas binaire (ex: 'Yes'/'No'), on essaye d'encodage simple
    if y.dtype == object or y.dtype.name == 'category':
        y = y.astype(str).map(lambda s: 1 if s.strip().lower() in ['yes','y','1','true','t'] else 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )

    print(f"Split: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, list, list]:
    """Construit un ColumnTransformer pour imputation, encodage et scaling.

    Retourne le preprocessor et les listes de colonnes numériques et catégoriques.
    """
    # heuristique simple pour typer les colonnes
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()

    # Columns that might be numeric but stored as object (e.g. 'TotalCharges') -> try to convert
    for col in X.columns:
        if col not in numeric_cols and col not in categorical_cols:
            # attempt conversion
            try:
                X[col] = pd.to_numeric(X[col])
                numeric_cols.append(col)
            except Exception:
                categorical_cols.append(col)

    # Pipelines
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ], remainder='drop')

    print(f"Preprocessor built. Numeric cols: {len(numeric_cols)}, Categorical cols: {len(categorical_cols)}")
    return preprocessor, numeric_cols, categorical_cols


def apply_variance_threshold(X: np.ndarray, threshold: float = 0.0) -> Tuple[np.ndarray, VarianceThreshold]:
    """Applique VarianceThreshold sur la matrice numpy X (features transformées).

    Retourne X_reduced et l'objet fitted VarianceThreshold.
    """
    sel = VarianceThreshold(threshold=threshold)
    X_reduced = sel.fit_transform(X)
    print(f"VarianceThreshold applied: {X.shape[1]} -> {X_reduced.shape[1]} features kept")
    return X_reduced, sel


def train_models(X_train: np.ndarray, y_train: np.ndarray, models: Dict[str, Any]) -> Dict[str, Any]:
    """Entraîne les modèles donnés et renvoie un dict {name: fitted_model}.

    models: dictionnaire {name: estimator}
    """
    fitted = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        fitted[name] = model
        print(f"{name} trained")
    return fitted


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Calcule métriques classiques et renvoie un dict."""
    y_pred = model.predict(X_test)
    y_prob = None
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        # some estimators (like SVC without probability=True) don't support predict_proba
        try:
            y_prob = model.decision_function(X_test)
            # squash to 0-1 using min-max
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
        except Exception:
            y_prob = None

    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
    metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        except Exception:
            metrics['roc_auc'] = np.nan
        try:
            metrics['average_precision'] = average_precision_score(y_test, y_prob)
        except Exception:
            metrics['average_precision'] = np.nan
    else:
        metrics['roc_auc'] = np.nan
        metrics['average_precision'] = np.nan

    print(f"Eval metrics: {metrics}")
    return metrics


def plot_roc_pr(model, X_test, y_test, out_path_prefix: str):
    """Trace et sauve les courbes ROC et PR pour un modèle donné."""
    y_prob = None
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        try:
            y_prob = model.decision_function(X_test)
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
        except Exception:
            print("No probability or decision function available; skipping ROC/PR plots")
            return

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True)
    roc_path = out_path_prefix + '_roc.png'
    plt.savefig(roc_path)
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    pr_path = out_path_prefix + '_pr.png'
    plt.savefig(pr_path)
    plt.close()

    print(f"Saved plots: {roc_path}, {pr_path}")


def compare_and_save(fitted_models: Dict[str, Any], X_test, y_test, out_dir: str):
    """Évalue tous les modèles, sauve métriques dans CSV et modèles sur disque."""
    os.makedirs(out_dir, exist_ok=True)
    metrics_list = []

    for name, model in fitted_models.items():
        print(f"Evaluating {name}")
        metrics = evaluate_model(model, X_test, y_test)
        metrics['model'] = name
        metrics_list.append(metrics)

        # save model
        model_path = os.path.join(out_dir, f"{name}.joblib")
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")

        # plots
        try:
            plot_roc_pr(model, X_test, y_test, os.path.join(out_dir, name))
        except Exception as e:
            print(f"Could not plot for {name}: {e}")

    df_metrics = pd.DataFrame(metrics_list).set_index('model')
    csv_path = os.path.join(out_dir, 'metrics_summary.csv')
    df_metrics.to_csv(csv_path)
    print(f"Saved metrics summary to {csv_path}")
    return df_metrics


def main(args):
    df = load_data(args.data_path)

    X_train_df, X_test_df, y_train, y_test = split_data(df, args.target, test_size=args.test_size, random_state=args.random_state)

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train_df.copy())

    # transformer: fit preprocessor on train only
    X_train_trans = preprocessor.fit_transform(X_train_df)
    X_test_trans = preprocessor.transform(X_test_df)

    # optionally apply VarianceThreshold
    vt = None
    if args.use_variance_threshold:
        X_train_trans, vt = apply_variance_threshold(X_train_trans, threshold=args.vt_threshold)
        # apply same selection to test set
        X_test_trans = vt.transform(X_test_trans)

    # define models
    models = {
        'logistic': LogisticRegression(max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=args.random_state)
    }
    if XGBOOST_AVAILABLE:
        models['xgboost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=args.random_state)
    else:
        print('XGBoost not available; skipping')

    # train
    fitted = train_models(X_train_trans, y_train.values, models)

    # evaluate & save
    df_metrics = compare_and_save(fitted, X_test_trans, y_test.values, args.out_dir)

    # save artifacts: preprocessor and (optional) variance selector
    joblib.dump(preprocessor, os.path.join(args.out_dir, 'preprocessor.joblib'))
    print(f"Saved preprocessor to {os.path.join(args.out_dir, 'preprocessor.joblib')}")
    if vt is not None:
        joblib.dump(vt, os.path.join(args.out_dir, 'variance_selector.joblib'))
        print(f"Saved variance selector to {os.path.join(args.out_dir, 'variance_selector.joblib')})")

    # Save a brief report JSON
    report = {
        'metrics': df_metrics.to_dict(orient='index'),
        'models_trained': list(fitted.keys()),
        'n_train': int(X_train_df.shape[0]),
        'n_test': int(X_test_df.shape[0])
    }
    with open(os.path.join(args.out_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {os.path.join(args.out_dir, 'report.json')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline de churn prediction')
    parser.add_argument('--data_path', type=str, required=True, help='Chemin vers le fichier CSV de données')
    parser.add_argument('--target', type=str, default='Churn', help='Nom de la colonne cible')
    parser.add_argument('--out_dir', type=str, default='results', help='Répertoire de sortie')
    parser.add_argument('--test_size', type=float, default=0.2, help='Taille du test set')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    parser.add_argument('--use_variance_threshold', action='store_true', help='Activer VarianceThreshold')
    parser.add_argument('--vt_threshold', type=float, default=0.0, help='Seuil de VarianceThreshold (par défaut 0.0)')

    args = parser.parse_args()
    main(args)
