
from sklearn.metrics         import make_scorer
import numpy as np
from sklearn.model_selection import  KFold
# ============================================================
# GLOBAL CONFIGURATION — edit here to control the whole run
# ============================================================

RANDOM_SEED = 42
TEST_SIZE    = 0.2   # fraction held out as final test set
CV_FOLDS     = 5     # cross-validation folds during model dev

# Toggle individual feature-engineering blocks on/off.
# Set to False to skip a block without deleting the code.
FEATURE_CONFIG = {
    "log_transform_target"     : True,   # model log(price) instead of price
    "basic_numeric_transforms" : True,   # log/sqrt of skewed numerics
    "polynomial_features"      : False,  # degree-2 poly on top features (slow)
    "spline_features"          : True,   # natural splines on key numerics
    "interaction_features"     : True,   # hand-crafted interaction terms
    "composite_features"       : True,   # domain-informed (age, ratios, etc.)
    "knn_neighborhood_features": False,  # KNN local-average features (slow)
    "target_encoding"          : True,   # target-encode high-cardinality cats
    "pca_features"             : False,  # PCA-derived features
    "segment_models"           : False,  # fit separate models per neighborhood
}

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error — primary evaluation metric."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mape_scorer():
    """Sklearn-compatible MAPE scorer (negative, for GridSearch)."""
    def _mape(y_true, y_pred):
        return -mape(y_true, y_pred)
    return make_scorer(_mape)

def cv_mape(model, X, y, folds=CV_FOLDS):
    """Cross-validated MAPE on *price* space (handles log target)."""
    kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        # If we modelled log(price), convert back before MAPE
        if FEATURE_CONFIG["log_transform_target"]:
            preds  = np.expm1(preds)
            y_val  = np.expm1(y_val)
        scores.append(mape(y_val, preds))
    return np.mean(scores), np.std(scores)