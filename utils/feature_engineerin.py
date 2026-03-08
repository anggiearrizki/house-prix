
import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer
from sklearn.neighbors import KNeighborsRegressor


# ══════════════════════════════════════════════════════════════
# Time Related Features  (from 'date', 'yr_built', 'yr_renovated')
# ══════════════════════════════════════════════════════════════

def engineer_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based signals.
    Insight: newer builds and recent renovations both boost price.
    """
    df = df.copy()

    # TODO: add features here 

    return df


# ══════════════════════════════════════════════════════════════
# Size & Space Features  (sqft_living, sqft_lot, sqft_above, sqft_basement)
# ══════════════════════════════════════════════════════════════

def engineer_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio and composite area features.
    Insight: relative proportions often matter as much as absolute size.
    """
    df = df.copy()

    # TODO: add features here
    return df


# ══════════════════════════════════════════════════════════════
# Quality Related Features  (view, condition, waterfront)
# ══════════════════════════════════════════════════════════════

def engineer_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag premium / atypical properties based on categorical quality columns.
    Insight: waterfront and high view scores carry large nonlinear premiums.
    """
    df = df.copy()

    # TODO: add features here

    return df


# ══════════════════════════════════════════════════════════════
# Interaction Features
# ══════════════════════════════════════════════════════════════

def engineer_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multiplicative interaction terms that a linear model cannot learn on its own.
    Only add interactions motivated by EDA — too many interactions → overfitting.
    """
    df = df.copy()

    # TODO: add features here
    return df



# ══════════════════════════════════════════════════════════════
# To generate the final features
# ══════════════════════════════════════════════════════════════

def run_feature_engineering(
    X_train_raw: pd.DataFrame,
    X_test_raw:  pd.DataFrame,
    y_train:     pd.Series,
    cfg:         dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply all enabled feature engineering steps.
    ALWAYS fits on X_train_raw, transforms both.

    Parameters
    ----------
    X_train_raw : raw training features (no target)
    X_test_raw  : raw test features
    y_train     : raw SalePrice (not log-transformed) for fitting encoders
    cfg         : FEATURE_CONFIG dict from the notebook

    Returns
    -------
    (X_train_fe, X_test_fe) ready to pass into the sklearn preprocessor
    """
    y_train_log = np.log1p(y_train)   # used by target encoders


    # ── Step 1: Time features ──────────────────────────────
    if cfg.get("temporal_features", True):
        Xtr = engineer_time(Xtr)
        Xte = engineer_time(Xte)

    # ── Step 2: Size & space features ─────────────────────────
    if cfg.get("size_features", True):
        Xtr = engineer_size(Xtr)
        Xte = engineer_size(Xte)

    # ── Step 3: Quality & condition features ───────────────────
    if cfg.get("quality_features", True):
        Xtr = engineer_quality(Xtr)
        Xte = engineer_quality(Xte)

    # ── Step 4: Interaction features ───────────────────────────
    if cfg.get("interaction_features", True):
        Xtr = engineer_interactions(Xtr)
        Xte = engineer_interactions(Xte)

    
    print(f"[FE] Train shape: {Xtr.shape} | Test shape: {Xte.shape}")
    return Xtr, Xte