"""The model pipeline, shared by every script. Preprocessing lives INSIDE the pipeline."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_data import (
    CATEGORICAL_FEATURES,
    FORBIDDEN_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)


def build_model(numeric=None, categorical=None) -> Pipeline:
    """Default features come from ml_data, so the forbidden-feature guard covers them."""
    numeric = list(NUMERIC_FEATURES if numeric is None else numeric)
    categorical = list(CATEGORICAL_FEATURES if categorical is None else categorical)

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def assert_no_forbidden_features(features) -> None:
    """Raise if a post-call feature sneaks into the feature list."""
    leaked = FORBIDDEN_FEATURES.intersection(features)
    if leaked:
        raise ValueError(f"Forbidden post-call features: {sorted(leaked)}")
