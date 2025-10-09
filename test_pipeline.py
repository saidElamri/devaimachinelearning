# test_pipeline.py
import pytest
import pandas as pd
import numpy as np
from pipeline import load_data, split_data, build_preprocessor, train_models, evaluate_models, run_full_pipeline
import os

DATA_SAMPLE = "Data.csv"
TARGET_COL = "Churn"  # Use actual column name from dataset

@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory):
    # Si dataset trop gros, prendre un échantillon ; sinon utiliser le fichier.
    p = tmp_path_factory.mktemp("data") / "Data.csv"
    df = pd.read_csv(DATA_SAMPLE)
    df_sample = df.sample(n=min(1000, len(df)), random_state=42)
    df_sample.to_csv(p, index=False)
    return str(p)

def test_load_and_split(small_dataset):
    X, y = load_data(small_dataset, target_col=TARGET_COL)
    assert X.shape[0] == y.shape[0]
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
    # check stratification approximate
    prop_full = y.mean()
    prop_train = y_train.mean()
    assert abs(prop_full - prop_train) < 0.1  # tol 10%

def test_preprocessor_and_train(small_dataset):
    X, y = load_data(small_dataset, target_col=TARGET_COL)
    X_train, X_test, y_train, y_test = split_data(X, y)
    num_cols = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object','category','bool']).columns.tolist()
    preproc = build_preprocessor(num_cols, cat_cols, use_variance_threshold=True, var_thresh=0.0)
    models = {'logreg': None}
    trained = train_models(X_train, y_train, preproc, models_to_train={
        'logreg': None if False else __import__('sklearn.linear_model').linear_model.LogisticRegression(max_iter=1000)
    })
    # ensure keys present
    assert 'logreg' in trained

def test_run_full_pipeline_returns_expected_keys(small_dataset):
    out = run_full_pipeline(small_dataset, target_col=TARGET_COL, use_variance=False, var_thresh=0.0, test_size=0.2)
    assert 'trained' in out and 'results' in out and 'best' in out
    # results structure
    results = out['results']
    assert isinstance(results, dict)
    for name, res in results.items():
        assert 'accuracy' in res and 'f1' in res and 'recall' in res

def test_minimum_performance(small_dataset):
    # On échantillon, vérifier que F1 n'est pas zéro pour au moins un modèle
    out = run_full_pipeline(small_dataset, target_col=TARGET_COL, use_variance=False)
    results = out['results']
    f1s = [r['f1'] for r in results.values() if r['f1'] is not None]
    assert max(f1s) > 0.05  # seuil très bas mais aide à attraper pipeline cassé
