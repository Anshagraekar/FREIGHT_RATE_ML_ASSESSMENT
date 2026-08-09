from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import create_features


# ============================================================
# Configuration
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT_DIR / "data" / "train-test.csv"

OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load data
# ============================================================

def load_data():
    """Load development dataset."""

    df = pd.read_csv(TRAIN_PATH)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    return df


# ============================================================
# Prepare features
# ============================================================

def prepare_features(df):
    """
    Apply shared feature engineering and prepare
    data for CatBoost.
    """

    data = create_features(df)

    # load_id is an identifier, not a predictive feature.
    if "load_id" in data.columns:
        data = data.drop(columns=["load_id"])

    # Target is separated before training.
    if "posted_rate" in data.columns:
        y = data["posted_rate"].copy()
        X = data.drop(columns=["posted_rate"])
    else:
        y = None
        X = data

    return X, y


# ============================================================
# CatBoost categorical columns
# ============================================================

def get_categorical_columns(X):
    """Return categorical feature names."""

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    return categorical_columns


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(y_true, predictions):
    """Calculate regression metrics."""

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions
        )
    )

    r2 = r2_score(
        y_true,
        predictions
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# Create time-based folds
# ============================================================

def create_time_folds(df):
    """
    Create forward-looking validation folds.

    Each validation month is strictly after
    the corresponding training period.
    """

    folds = [
        {
            "name": "July",
            "train_end": "2025-06-30",
            "valid_start": "2025-07-01",
            "valid_end": "2025-07-31",
        },
        {
            "name": "August",
            "train_end": "2025-07-31",
            "valid_start": "2025-08-01",
            "valid_end": "2025-08-31",
        },
        {
            "name": "September",
            "train_end": "2025-08-31",
            "valid_start": "2025-09-01",
            "valid_end": "2025-09-30",
        },
        {
            "name": "October",
            "train_end": "2025-09-30",
            "valid_start": "2025-10-01",
            "valid_end": "2025-10-31",
        },
    ]

    return folds


# ============================================================
# Train one CatBoost model
# ============================================================

def train_model(X_train, y_train, X_valid, y_valid):
    """Train CatBoost regression model."""

    categorical_columns = get_categorical_columns(
        X_train
    )

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=1200,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=200,
        allow_writing_files=False
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_columns,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=100
    )

    return model


# ============================================================
# Walk-forward validation
# ============================================================

def run_walk_forward_validation(df):
    """Run all time-based validation folds."""

    results = []

    folds = create_time_folds(df)

    for fold in folds:

        print("\n" + "=" * 80)
        print(f"VALIDATION FOLD: {fold['name']}")
        print("=" * 80)

        train_mask = (
            df["date"] <= pd.Timestamp(
                fold["train_end"]
            )
        )

        valid_mask = (
            (df["date"] >= pd.Timestamp(
                fold["valid_start"]
            ))
            &
            (df["date"] <= pd.Timestamp(
                fold["valid_end"]
            ))
        )

        train_df = df.loc[train_mask].copy()
        valid_df = df.loc[valid_mask].copy()

        print(
            f"Training rows: {len(train_df):,}"
        )

        print(
            f"Validation rows: {len(valid_df):,}"
        )

        # ----------------------------------------------------
        # Feature engineering
        # ----------------------------------------------------

        X_train, y_train = prepare_features(
            train_df
        )

        X_valid, y_valid = prepare_features(
            valid_df
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model = train_model(
            X_train,
            y_train,
            X_valid,
            y_valid
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        predictions = model.predict(
            X_valid
        )

        predictions = np.maximum(
            predictions,
            1e-6
        )

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        metrics = calculate_metrics(
            y_valid,
            predictions
        )

        print("\nFold results:")

        print(
            f"MAE  : ${metrics['MAE']:,.2f}"
        )

        print(
            f"RMSE : ${metrics['RMSE']:,.2f}"
        )

        print(
            f"R²   : {metrics['R2']:.4f}"
        )

        results.append(
            {
                "fold": fold["name"],
                "train_rows": len(train_df),
                "validation_rows": len(valid_df),
                **metrics
            }
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "walk_forward_results.csv",
        index=False
    )

    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 80)

    print(results_df.to_string(index=False))

    print("\nAverage metrics:")

    print(
        f"MAE  : ${results_df['MAE'].mean():,.2f}"
    )

    print(
        f"RMSE : ${results_df['RMSE'].mean():,.2f}"
    )

    print(
        f"R²   : {results_df['R2'].mean():.4f}"
    )

    return results_df


# ============================================================
# Train final development model
# ============================================================

def train_final_model(df):

    print("\n" + "=" * 80)
    print("TRAINING FINAL DEVELOPMENT MODEL")
    print("=" * 80)

    X, y = prepare_features(df)

    categorical_columns = get_categorical_columns(X)

    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=1200,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=200,
        allow_writing_files=False
    )

    model.fit(
        X,
        y,
        cat_features=categorical_columns
    )

    model_path = MODEL_DIR / "catboost_baseline.cbm"

    model.save_model(model_path)

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.get_feature_importance()
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    importance.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False
    )

    print("\nTop 20 features:")
    print(
        importance.head(20).to_string(
            index=False
        )
    )

    print(
        f"\nFinal model saved to: {model_path}"
    )

    return model

# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("FREIGHT RATE PREDICTION")
    print("=" * 80)

    df = load_data()

    print(
        f"\nDevelopment dataset: {df.shape}"
    )

    # --------------------------------------------------------
    # Walk-forward validation
    # --------------------------------------------------------

    run_walk_forward_validation(df)

    # --------------------------------------------------------
    # Train final model
    # --------------------------------------------------------

    train_final_model(df)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()