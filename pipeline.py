# pipeline.py
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, recall_score, f1_score, roc_auc_score,
                             precision_recall_curve, roc_curve, auc, classification_report,
                             confusion_matrix, precision_score)
import matplotlib.pyplot as plt
import joblib
import os

RANDOM_STATE = 42

def load_data(path: str, target_col: str = "churn") -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    # Handle both 'churn' and 'Churn' column names
    actual_target_col = target_col
    if target_col.lower() == 'churn':
        # Try 'churn' first, then 'Churn'
        if target_col not in df.columns and 'Churn' in df.columns:
            actual_target_col = 'Churn'
        elif target_col not in df.columns:
            raise ValueError(f"Target column {target_col} or 'Churn' not found in data")
    elif target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in data")

    X = df.drop(columns=[actual_target_col])
    y = df[actual_target_col].apply(lambda v: 1 if str(v).lower() in ['1','yes','y','true','t'] else 0)
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=RANDOM_STATE):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def build_preprocessor(num_cols: List[str], cat_cols: List[str], use_variance_threshold: bool=False, var_thresh: float=0.0):
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    transformers = [('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)]
    preproc = ColumnTransformer(transformers=transformers, remainder='drop')
    if use_variance_threshold:
        return Pipeline([('preproc', preproc), ('vt', VarianceThreshold(threshold=var_thresh))])
    else:
        return Pipeline([('preproc', preproc)])

def train_models(X_train, y_train, preprocessor, models_to_train=None, cv=3):
    if models_to_train is None:
        models_to_train = {
            'logreg': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            'rf': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
            'svc': SVC(probability=True, random_state=RANDOM_STATE)
        }
    trained = {}
    for name, clf in models_to_train.items():
        pipe = Pipeline([('pre', preprocessor), ('clf', clf)])
        print(f"Training {name} ...")
        pipe.fit(X_train, y_train)
        trained[name] = pipe
    return trained

def evaluate_model(pipe, X_test, y_test) -> Dict[str, Any]:
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:,1] if hasattr(pipe, "predict_proba") else None
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    precision = precision_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return {"accuracy": acc, "recall": rec, "f1": f1, "roc_auc": roc_auc, "precision": precision, "confusion_matrix": cm, "y_pred": y_pred, "y_proba": y_proba}

def evaluate_models(trained_models: dict, X_test, y_test) -> Dict[str, dict]:
    results = {}
    for name, pipe in trained_models.items():
        results[name] = evaluate_model(pipe, X_test, y_test)
    return results

def plot_roc_pr(results: dict, X_test, y_test, outdir="output"):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(8,6))
    for name, res in results.items():
        y_proba = res["y_proba"]
        if y_proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curves"); plt.legend()
    plt.savefig(os.path.join(outdir,"roc_curves.png"))
    plt.close()

    # PR curves
    plt.figure(figsize=(8,6))
    for name, res in results.items():
        y_proba = res["y_proba"]
        if y_proba is None:
            continue
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{name} (AUPR={pr_auc:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall curves"); plt.legend()
    plt.savefig(os.path.join(outdir,"pr_curves.png"))
    plt.close()

def compare_bar(results: dict, outdir="output"):
    os.makedirs(outdir, exist_ok=True)
    metrics = ['accuracy','recall','f1','roc_auc']
    comps = {m: [] for m in metrics}
    names = []
    for name, res in results.items():
        names.append(name)
        for m in metrics:
            val = res.get(m)
            comps[m].append(val if val is not None else np.nan)
    # bar plot
    df_comp = pd.DataFrame(comps, index=names)
    ax = df_comp.plot(kind='bar', figsize=(10,6))
    ax.set_title("Comparaison des modèles")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,"models_comparison.png"))
    plt.close()
    return df_comp

def save_model(pipe, path):
    joblib.dump(pipe, path)
    print(f"Model saved to {path}")

# Example runner
def run_full_pipeline(data_path: str, target_col="churn", use_variance=False, var_thresh=0.0, test_size=0.2):
    X, y = load_data(data_path, target_col=target_col)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
    # identify feature types
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()
    preproc = build_preprocessor(num_cols, cat_cols, use_variance_threshold=use_variance, var_thresh=var_thresh)
    trained = train_models(X_train, y_train, preproc)
    results = evaluate_models(trained, X_test, y_test)
    plot_roc_pr(results, X_test, y_test)
    df_comp = compare_bar(results)
    # Save best model (choose by f1 or recall)
    best = max(results.items(), key=lambda kv: (kv[1]['f1'], kv[1]['recall']))[0]
    save_model(trained[best], f"output/best_model_{best}.joblib")
    return {"trained": trained, "results": results, "df_comp": df_comp, "best": best}
