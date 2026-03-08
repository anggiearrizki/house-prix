import numpy as np
import pandas as pd

from sklearn.pipeline    import Pipeline
from sklearn.compose     import ColumnTransformer
from sklearn.impute      import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_preprocessor(X_fe: pd.DataFrame, ohe_location: bool = False, BINARY_COLS=[], ORDINAL_COLS=[], CONTINUOUS_COLS=[]) -> ColumnTransformer:
    """
    Build a ColumnTransformer fitted to the actual columns in X_fe.

    Parameters
    ----------
    X_fe         : output of run_feature_engineering() — used to detect
                   which columns actually exist (some are conditional)
    ohe_location : set True if you turned OFF target encoding and want
                   to OHE city/statezip instead. Default False.

    Returns
    -------
    ColumnTransformer (unfitted) — pass into a sklearn Pipeline.
    """
    present = set(X_fe.columns)

    # ── Numeric columns to scale ───────────────────────────────
    # Combine all expected numeric groups, keep only what's present,
    # and also catch any auto-generated spline_* columns.
    expected_numeric = BINARY_COLS + ORDINAL_COLS + CONTINUOUS_COLS
    spline_cols      = [c for c in X_fe.columns if c.startswith("sp_")]

    numeric_cols = (
        [c for c in expected_numeric if c in present]
        + spline_cols
    )
    # De-duplicate preserving order
    seen = set()
    numeric_cols = [c for c in numeric_cols if not (c in seen or seen.add(c))]

    # ── Categorical columns to OHE ─────────────────────────────
    if ohe_location:
        cat_cols = [c for c in OHE_COLS_FALLBACK if c in present]
    else:
        cat_cols = []   # target encoding already handled location

    # ── Sanity check — warn about any columns that will be dropped ─
    accounted_for = set(numeric_cols + cat_cols)
    dropped        = present - accounted_for
    if dropped:
        print(f"[Preprocessor] WARNING — these columns are NOT in any "
              f"transformer and will be dropped by remainder='drop':\n  {sorted(dropped)}")

    # ── Build pipelines ────────────────────────────────────────
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])

    transformers = [("num", numeric_pipe, numeric_cols)]

    if cat_cols:
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",     # drop anything not explicitly listed
        verbose_feature_names_out=False,
    )

    print(f"[Preprocessor] Numeric cols : {len(numeric_cols)}")
    print(f"[Preprocessor] Categorical  : {len(cat_cols)}")
    print(f"[Preprocessor] Total input  : {len(numeric_cols) + len(cat_cols)}")

    return preprocessor


