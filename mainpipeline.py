"""
pipeline.py

Pipeline complet de préparation, entraînement et évaluation pour prédiction du churn client.
Fonctions réutilisables pour notebook et exécution CLI.
"""

import os
import json
import warnings
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
)

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available; skipping")


# -------------------------- Data Loading --------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# -------------------------- Preprocessing --------------------------
def infer_feature_types(df: pd.DataFrame, target_col: Optional[str] = "Churn", cat_threshold: int = 15) -> Tuple[List[str], List[str]]:
    cols = df.columns.tolist()
    if target_col in cols:
        cols.remove(target_col)

    numeric_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_candidates = df[cols].select_dtypes(exclude=[np.number]).columns.tolist()
    
    for c in df[cols].select_dtypes(include=[np.number]).columns:
        if df[c].nunique() <= cat_threshold:
            cat_candidates.append(c)
            if c in numeric_cols:
                numeric_cols.remove(c)

    categorical_cols = sorted(list(set(cat_candidates)))
    return numeric_cols, categorical_cols


def _binarize_target(y: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(int)
    mapping = {'yes':1,'y':1,'true':1,'1':1,'no':0,'n':0,'false':0,'0':0}
    def map_val(v):
        if pd.isna(v): return np.nan
        s = str(v).strip().lower()
        return mapping.get(s,np.nan)
    res = y.map(map_val)
    if res.isna().any():
        uniques = y.unique().tolist()
        if len(uniques)==2:
            res = y.map({uniques[0]:0, uniques[1]:1})
        else:
            raise ValueError(f"Target non-binaire. Valeurs détectées: {uniques}")
    return res.astype(int)


def train_test_split_data(df: pd.DataFrame, target_col: str = "Churn",
                          test_size: float = 0.2, random_state: int = 42,
                          stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[target_col])
    y = _binarize_target(df[target_col])
    strat = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)


def build_preprocessing_pipeline(numeric_cols: List[str], categorical_cols: List[str],
                                 num_impute_strategy: str = 'median', cat_impute_strategy: str = 'constant',
                                 cat_fill_value: str = 'Unknown', encoder: str = 'onehot',
                                 scaler: str = 'standard', variance_threshold: Optional[float] = None,
                                 onehot_sparse: bool = False) -> Pipeline:
    num_steps = [('imputer', SimpleImputer(strategy=num_impute_strategy))]
    if scaler == 'standard':
        num_steps.append(('scaler', StandardScaler()))
    elif scaler == 'minmax':
        num_steps.append(('scaler', MinMaxScaler()))
    numeric_transformer = Pipeline(steps=num_steps)

    cat_steps = [('imputer', SimpleImputer(strategy=cat_impute_strategy, fill_value=cat_fill_value))]
    if encoder == 'onehot':
        try:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=not onehot_sparse)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse=not onehot_sparse)
        cat_steps.append(('onehot', ohe))
    elif encoder == 'ordinal':
        try:
            ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        except TypeError:
            ord_enc = OrdinalEncoder()
        cat_steps.append(('ordinal', ord_enc))

    categorical_transformer = Pipeline(steps=cat_steps)
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_cols),
                                                   ('cat', categorical_transformer, categorical_cols)],
                                     remainder='drop', sparse_threshold=0)
    steps = [('preprocessor', preprocessor)]
    if variance_threshold is not None:
        steps.append(('var_thresh', VarianceThreshold(variance_threshold)))
    return Pipeline(steps=steps)


def fit_preprocessor(preprocessor: Pipeline, X_train: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    fitted = preprocessor.fit(X_train)
    X_trans = fitted.transform(X_train)
    feature_names = _get_feature_names_from_pipeline(fitted, X_train.columns)
    return X_trans, feature_names


def transform_df(preprocessor: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return preprocessor.transform(X)


def _get_feature_names_from_pipeline(pipeline_obj: Pipeline, input_features: Optional[List[str]] = None) -> List[str]:
    ct = None
    if isinstance(pipeline_obj, Pipeline):
        if 'preprocessor' in pipeline_obj.named_steps:
            ct = pipeline_obj.named_steps['preprocessor']
        else:
            for _, t in pipeline_obj.steps:
                if isinstance(t, ColumnTransformer):
                    ct = t
                    break
    elif isinstance(pipeline_obj, ColumnTransformer):
        ct = pipeline_obj
    if ct is None:
        return [f"feature_{i}" for i in range(0)]
    names = []
    for name, trans, cols in ct.transformers_:
        if name == 'remainder': continue
        cols_list = list(cols) if isinstance(cols,(list,tuple,np.ndarray)) else [cols]
        if isinstance(trans, Pipeline):
            last = list(trans.named_steps.items())[-1][1]
            trans = last
        if hasattr(trans,'get_feature_names_out'):
            try:
                out = trans.get_feature_names_out(cols_list)
                names.extend(out)
            except Exception:
                names.extend([f"{name}__{c}" for c in cols_list])
        else:
            names.extend([f"{name}__{c}" for c in cols_list])
    return names


def save_pipeline(obj, path: str):
    joblib.dump(obj, path)


def load_pipeline(path: str):
    return joblib.load(path)


def preprocess_and_split(df: pd.DataFrame, target_col: str = 'Churn', test_size: float = 0.2,
                         random_state: int = 42, stratify: bool = True,
                         cat_threshold: int = 15, variance_threshold: Optional[float] = None) -> Dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split_data(df, target_col, test_size, random_state, stratify)
    numeric_cols, categorical_cols = infer_feature_types(df, target_col, cat_threshold)
    pipeline = build_preprocessing_pipeline(numeric_cols, categorical_cols, variance_threshold=variance_threshold)
    X_train_proc, feature_names = fit_preprocessor(pipeline, X_train)
    X_test_proc = transform_df(pipeline, X_test)
    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
            'pipeline': pipeline, 'X_train_proc': X_train_proc, 'X_test_proc': X_test_proc,
            'feature_names': feature_names, 'numeric_cols': numeric_cols, 'categorical_cols': categorical_cols}


# -------------------------- Model Training & Evaluation --------------------------
def train_models(X_train: np.ndarray, y_train: np.ndarray, models: Dict[str, Any]) -> Dict[str, Any]:
    fitted = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        fitted[name] = model
        print(f"{name} trained")
    return fitted


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    y_prob = None
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        try:
            y_prob = model.decision_function(X_test)
            y_prob = (y_prob - y_prob.min())/(y_prob.max()-y_prob.min())
        except Exception:
            y_prob = None
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    if y_prob is not None:
        try: metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        except: metrics['roc_auc'] = np.nan
        try: metrics['average_precision'] = average_precision_score(y_test, y_prob)
        except: metrics['average_precision'] = np.nan
    else:
        metrics['roc_auc'] = np.nan
        metrics['average_precision'] = np.nan
    return metrics


def plot_roc_pr(model, X_test, y_test, out_prefix: str):
    y_prob = None
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        try:
            y_prob = model.decision_function(X_test)
            y_prob = (y_prob - y_prob.min())/(y_prob.max()-y_prob.min())
        except: return
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.grid(True)
    plt.savefig(out_prefix+'_roc.png'); plt.close()
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(); plt.plot(recall, precision); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR Curve'); plt.grid(True)
    plt.savefig(out_prefix+'_pr.png'); plt.close()


def compare_and_save(fitted_models: Dict[str, Any], X_test, y_test, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    metrics_list = []
    for name, model in fitted_models.items():
        metrics = evaluate_model(model, X_test, y_test)
        metrics['model'] = name
        metrics_list.append(metrics)
        joblib.dump(model, os.path.join(out_dir, f"{name}.joblib"))
        plot_roc_pr(model, X_test, y_test, os.path.join(out_dir, name))
    df_metrics = pd.DataFrame(metrics_list).set_index('model')
    df_metrics.to_csv(os.path.join(out_dir,'metrics_summary.csv'))
    return df_metrics


# -------------------------- CLI --------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Pipeline de churn prediction')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--target', type=str, default='Churn')
    parser.add_argument('--out_dir', type=str, default='results')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--use_variance_threshold', action='store_true')
    parser.add_argument('--vt_threshold', type=float, default=0.0)
    args = parser.parse_args()

    df = load_data(args.data_path)
    split_result = preprocess_and_split(df, target_col=args.target,
                                        test_size=args.test_size, random_state=args.random_state,
                                        variance_threshold=args.vt_threshold if args.use_variance_threshold else None)
    X_train_proc = split_result['X_train_proc']
    X_test_proc = split_result['X_test_proc']
    y_train = split_result['y_train'].values
    y_test = split_result['y_test'].values

    models = {
        'logistic': LogisticRegression(max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=args.random_state)
    }
    if XGBOOST_AVAILABLE:
        models['xgboost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=args.random_state)

    fitted = train_models(X_train_proc, y_train, models)
    compare_and_save(fitted, X_test_proc, y_test, args.out_dir)
    save_pipeline(split_result['pipeline'], os.path.join(args.out_dir,'preprocessor.joblib'))
    print(f"Pipeline and models saved to {args.out_dir}")
