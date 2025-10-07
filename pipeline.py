"""
pipeline.py

Collection de fonctions réutilisables pour le préprocessing d'un projet de churn.
Contenu :
 - détection des colonnes numériques / catégorielles
 - encodage / imputation des colonnes catégorielles
 - imputation / normalisation des colonnes numériques
 - construction d'un pipeline sklearn (ColumnTransformer + Pipeline)
 - split train/test avec encodage du target binaire
 - fonctions utilitaires pour fitter, transformer, sauvegarder et récupérer les noms de features

Auteur: Data Scientist junior (template)
"""

from typing import List, Tuple, Optional, Dict
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

import joblib


def infer_feature_types(df: pd.DataFrame, target_col: Optional[str] = "Churn", cat_threshold: int = 15) -> Tuple[List[str], List[str]]:
    """Détecte automatiquement les colonnes numériques et catégorielles.

    - Exclut la colonne target si fournie.
    - cat_threshold : nombre unique maximum pour considérer une colonne d'objet comme catégorielle

    Retour : (numeric_cols, categorical_cols)
    """
    cols = df.columns.tolist()
    if target_col in cols:
        cols.remove(target_col)

    numeric_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    # Treat as categorical: object, bool, category OR numeric with few uniques
    cat_candidates = df[cols].select_dtypes(exclude=[np.number]).columns.tolist()
    for c in df[cols].select_dtypes(include=[np.number]).columns:
        if df[c].nunique() <= cat_threshold:
            cat_candidates.append(c)
            if c in numeric_cols:
                numeric_cols.remove(c)

    categorical_cols = sorted(list(set(cat_candidates)))

    return numeric_cols, categorical_cols


def _binarize_target(y: pd.Series) -> pd.Series:
    """Convertit la colonne target en 0/1 si nécessaire.
    Reconnaît des valeurs courantes ('Yes'/'No', 'Y'/'N', True/False, 1/0 ...)
    """
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(int)

    # mapping commun
    mapping = {
        'yes': 1, 'y': 1, 'true': 1, '1': 1,
        'no': 0, 'n': 0, 'false': 0, '0': 0
    }
    def map_val(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        return mapping.get(s, np.nan)

    res = y.map(map_val)
    if res.isna().any():
        # fallback : try sklearn LabelBinarizer-like behavior
        uniques = y.unique().tolist()
        if len(uniques) == 2:
            # map first->0 second->1
            m = {uniques[0]: 0, uniques[1]: 1}
            res = y.map(m)
        else:
            raise ValueError("Target non-binaire et conversion automatique impossible. Valeurs détectées: {}".format(uniques))
    return res.astype(int)


def train_test_split_data(df: pd.DataFrame,
                          target_col: str = "Churn",
                          test_size: float = 0.2,
                          random_state: int = 42,
                          stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataframe en X_train, X_test, y_train, y_test.

    - Convertit le target en binaire si nécessaire.
    - Garde les index d'origine.
    """
    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' pas trouvé dans le DataFrame")

    X = df.drop(columns=[target_col]).copy()
    y_raw = df[target_col].copy()
    y = _binarize_target(y_raw)

    strat = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    return X_train, X_test, y_train, y_test


def build_preprocessing_pipeline(numeric_cols: List[str],
                                 categorical_cols: List[str],
                                 num_impute_strategy: str = 'median',
                                 cat_impute_strategy: str = 'constant',
                                 cat_fill_value: str = 'Unknown',
                                 encoder: str = 'onehot',  # 'onehot' or 'ordinal'
                                 scaler: str = 'standard',  # 'standard' or 'minmax' or None
                                 variance_threshold: Optional[float] = None,
                                 onehot_sparse: bool = False
                                 ) -> Pipeline:
    """Construit un pipeline sklearn complet prêt à fitter.

    Retourne : sklearn.pipeline.Pipeline
    Étapes :
     - ColumnTransformer appliquant transformations num/ cat
     - (optionnel) VarianceThreshold
    """

    # Numeric pipeline
    num_steps = []
    num_steps.append(('imputer', SimpleImputer(strategy=num_impute_strategy)))
    if scaler is not None:
        if scaler == 'standard':
            num_steps.append(('scaler', StandardScaler()))
        elif scaler == 'minmax':
            num_steps.append(('scaler', MinMaxScaler()))
        else:
            raise ValueError('scaler doit être "standard" ou "minmax" ou None')

    numeric_transformer = Pipeline(steps=num_steps)

    # Categorical pipeline
    cat_steps = []
    cat_steps.append(('imputer', SimpleImputer(strategy=cat_impute_strategy, fill_value=cat_fill_value)))

    if encoder == 'onehot':
        # compatibility sklearn versions for sparse parameter
        try:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=not onehot_sparse)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse=not onehot_sparse)
        cat_steps.append(('onehot', ohe))
    elif encoder == 'ordinal':
        # Use -1 for unknown categories
        try:
            ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        except TypeError:
            # older sklearn may not support those args; use default (may error on unseen categories)
            ord_enc = OrdinalEncoder()
        cat_steps.append(('ordinal', ord_enc))
    else:
        raise ValueError("encoder must be 'onehot' or 'ordinal'")

    categorical_transformer = Pipeline(steps=cat_steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ], remainder='drop', sparse_threshold=0)

    steps = [('preprocessor', preprocessor)]
    if variance_threshold is not None:
        steps.append(('var_thresh', VarianceThreshold(variance_threshold)))

    full_pipeline = Pipeline(steps=steps)
    return full_pipeline


def fit_preprocessor(preprocessor: Pipeline, X_train: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Fit le pipeline (preprocessor) sur X_train et retourne le DataFrame transformé et la liste des noms de features.

    - preprocessor : objet Pipeline retourné par build_preprocessing_pipeline
    - Renvoie : (X_train_transformed (pd.DataFrame), feature_names)
    """
    fitted = preprocessor.fit(X_train)
    X_trans = fitted.transform(X_train)

    feature_names = _get_feature_names_from_pipeline(fitted, X_train.columns)

    try:
        df_out = pd.DataFrame(X_trans, columns=feature_names, index=X_train.index)
    except Exception:
        # fallback: index only, auto col names
        df_out = pd.DataFrame(X_trans, index=X_train.index)
        df_out.columns = feature_names if len(feature_names) == df_out.shape[1] else [f'feature_{i}' for i in range(df_out.shape[1])]

    return df_out, feature_names


def transform_df(preprocessor: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Transforme X avec un pipeline déjà fit et retourne un DataFrame avec noms de colonnes si possible."""
    X_t = preprocessor.transform(X)
    feature_names = _get_feature_names_from_pipeline(preprocessor, X.columns)
    try:
        df_out = pd.DataFrame(X_t, columns=feature_names, index=X.index)
    except Exception:
        df_out = pd.DataFrame(X_t, index=X.index)
    return df_out


def _get_feature_names_from_pipeline(pipeline_obj: Pipeline, input_features: Optional[List[str]] = None) -> List[str]:
    """Tentative robuste pour retrouver les noms des features après ColumnTransformer/OneHotEncoder.
    Retourne une liste de noms. Si impossible, renvoie des noms génériques.
    """
    # Si pipeline_obj est Pipeline avec étape 'preprocessor'
    ct = None
    if isinstance(pipeline_obj, Pipeline):
        if 'preprocessor' in pipeline_obj.named_steps:
            ct = pipeline_obj.named_steps['preprocessor']
        else:
            # maybe pipeline_obj is directly a ColumnTransformer
            possible = [s for s in pipeline_obj.steps if isinstance(s[1], ColumnTransformer)]
            if possible:
                ct = possible[0][1]
    elif isinstance(pipeline_obj, ColumnTransformer):
        ct = pipeline_obj

    if ct is None:
        # fallback: cannot identifier ColumnTransformer
        try:
            if hasattr(pipeline_obj, 'get_feature_names_out'):
                return list(pipeline_obj.get_feature_names_out(input_features))
        except Exception:
            pass
        # generic
        warnings.warn('Impossible de retrouver précisément les noms de features — noms génériques utilisés')
        return [f'feature_{i}' for i in range(_safe_n_columns_after_transform(pipeline_obj))]

    # maintenant ct est ColumnTransformer
    try:
        if input_features is not None:
            names = ct.get_feature_names_out(input_features)
        else:
            names = ct.get_feature_names_out()
        return list(names)
    except Exception:
        # fallback: essayons de construire manuellement
        names: List[str] = []
        for name, transformer, cols in ct.transformers_:
            if name == 'remainder':
                continue
            cols_list = list(cols) if isinstance(cols, (list, tuple, np.ndarray)) else [cols]
            # transformer peut être pipeline
            trans = transformer
            # si transformer est Pipeline, récupérons la dernière étape
            if isinstance(transformer, Pipeline):
                last = list(transformer.named_steps.items())[-1][1]
                trans = last
            if hasattr(trans, 'get_feature_names_out'):
                try:
                    out = trans.get_feature_names_out(cols_list)
                    names.extend(out)
                except Exception:
                    # fallback
                    for c in cols_list:
                        names.append(f"{name}__{c}")
            else:
                for c in cols_list:
                    names.append(f"{name}__{c}")
        return names


def _safe_n_columns_after_transform(preprocessor: Pipeline) -> int:
    """Tentative pour estimer le nombre de colonnes après transformation (utilisé pour noms génériques)."""
    # on essaie d'appliquer transform sur un petit DataFrame factice
    try:
        # build a dummy
        if isinstance(preprocessor, Pipeline) and 'preprocessor' in preprocessor.named_steps:
            ct = preprocessor.named_steps['preprocessor']
        elif isinstance(preprocessor, ColumnTransformer):
            ct = preprocessor
        else:
            ct = preprocessor
        # impossible de savoir la dimension sans input; renvoyer 0
        return 0
    except Exception:
        return 0


def save_pipeline(obj, path: str):
    """Sauvegarde un objet sklearn (pipeline, modèle...) avec joblib."""
    joblib.dump(obj, path)


def load_pipeline(path: str):
    """Charge un objet sauvegardé par joblib."""
    return joblib.load(path)


# Petite fonction utilitaire regroupant le flux complet de preprocessing et split
def preprocess_and_split(df: pd.DataFrame,
                         target_col: str = 'Churn',
                         test_size: float = 0.2,
                         random_state: int = 42,
                         stratify: bool = True,
                         cat_threshold: int = 15,
                         **pipeline_kwargs) -> Dict[str, object]:
    """Flux complet : detection colonnes -> split -> build pipeline -> fit pipeline -> transformer X_train/X_test.

    Retourne un dict contenant :
      X_train, X_test, y_train, y_test, pipeline (fit), X_train_proc, X_test_proc, feature_names
    """
    X_train, X_test, y_train, y_test = train_test_split_data(df, target_col=target_col,
                                                            test_size=test_size, random_state=random_state,
                                                            stratify=stratify)
    numeric_cols, categorical_cols = infer_feature_types(df, target_col=target_col, cat_threshold=cat_threshold)
    pipeline = build_preprocessing_pipeline(numeric_cols, categorical_cols, **pipeline_kwargs)
    X_train_proc, feature_names = fit_preprocessor(pipeline, X_train)
    X_test_proc = transform_df(pipeline, X_test)

    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'pipeline': pipeline,
        'X_train_proc': X_train_proc, 'X_test_proc': X_test_proc,
        'feature_names': feature_names,
        'numeric_cols': numeric_cols, 'categorical_cols': categorical_cols
    }


if __name__ == '__main__':
    print('Ce module contient des fonctions de preprocessing. Importez-les depuis votre notebook.')
