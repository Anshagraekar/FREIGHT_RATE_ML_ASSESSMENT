from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# Configuration
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT_DIR / "data" / "train-test.csv"

OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Original feature set
# ============================================================

ORIGINAL_FEATURES = [
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "market_index",
    "quote_signal",
]


# ============================================================
# Time-based validation folds
# ============================================================

FOLDS = [
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


# ============================================================
# Load data
# ============================================================

def load_data():

    df = pd.read_csv(TRAIN_PATH)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    return df


# ============================================================
# Prepare original features
# ============================================================

def prepare_features(df):

    data = df.copy()

    # Convert date into a numeric time representation.
    #
    # We do not use raw date strings because CatBoost would
    # otherwise treat them as categorical values.

    data["date_numeric"] = (
        data["date"]
        - pd.Timestamp("2025-01-01")
    ).dt.days

    X = data[
        ORIGINAL_FEATURES
        + ["date_numeric"]
    ].copy()

    y = data["posted_rate"].copy()

    return X, y


# ============================================================
# Get categorical columns
# ============================================================

def get_categorical_columns(X):

    return X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true,
    predictions
):

    predictions = np.maximum(
        predictions,
        1e-6
    )

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

    return mae, rmse, r2


# ============================================================
# Train CatBoost
# ============================================================

def train_catboost(
    X_train,
    y_train,
    X_valid,
    y_valid,
    depth,
    learning_rate,
    iterations,
    use_log_target=False,
    loss_function="RMSE"
):

    categorical_columns = (
        get_categorical_columns(X_train)
    )

    # --------------------------------------------------------
    # Target transformation
    # --------------------------------------------------------

    if use_log_target:

        y_train_model = np.log1p(
            y_train
        )

        y_valid_model = np.log1p(
            y_valid
        )

    else:

        y_train_model = y_train
        y_valid_model = y_valid

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = CatBoostRegressor(
        loss_function=loss_function,
        eval_metric="MAE",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=False,
        allow_writing_files=False
    )

    model.fit(
        X_train,
        y_train_model,
        cat_features=categorical_columns,
        eval_set=(
            X_valid,
            y_valid_model
        ),
        early_stopping_rounds=80
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = model.predict(
        X_valid
    )

    if use_log_target:

        predictions = np.expm1(
            predictions
        )

    return model, predictions


# ============================================================
# Run one experiment
# ============================================================

def run_experiment(
    df,
    experiment_name,
    depth,
    learning_rate,
    iterations,
    use_log_target=False,
    loss_function="RMSE"
):

    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 80)

    fold_results = []

    for fold in FOLDS:

        print(
            f"\nRunning fold: {fold['name']}"
        )

        train_mask = (
            df["date"]
            <= pd.Timestamp(
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

        train_df = df.loc[
            train_mask
        ].copy()

        valid_df = df.loc[
            valid_mask
        ].copy()

        X_train, y_train = (
            prepare_features(
                train_df
            )
        )

        X_valid, y_valid = (
            prepare_features(
                valid_df
            )
        )

        model, predictions = train_catboost(
            X_train,
            y_train,
            X_valid,
            y_valid,
            depth=depth,
            learning_rate=learning_rate,
            iterations=iterations,
            use_log_target=use_log_target,
            loss_function=loss_function
        )

        mae, rmse, r2 = calculate_metrics(
            y_valid,
            predictions
        )

        best_iteration = (
            model.get_best_iteration()
        )

        print(
            f"MAE: ${mae:,.2f} | "
            f"RMSE: ${rmse:,.2f} | "
            f"R²: {r2:.4f} | "
            f"Best iteration: {best_iteration}"
        )

        fold_results.append({
            "experiment": experiment_name,
            "fold": fold["name"],
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "best_iteration": best_iteration
        })

    return fold_results


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("CATBOOST MODEL EXPERIMENTS")
    print("=" * 80)

    df = load_data()

    # --------------------------------------------------------
    # Model configurations
    # --------------------------------------------------------

    configurations = [
        {
            "name": "CatBoost_A",
            "depth": 6,
            "learning_rate": 0.05,
            "iterations": 500,
        },
        {
            "name": "CatBoost_B",
            "depth": 7,
            "learning_rate": 0.05,
            "iterations": 500,
        },
        {
            "name": "CatBoost_C",
            "depth": 8,
            "learning_rate": 0.05,
            "iterations": 500,
        },
        {
            "name": "CatBoost_D",
            "depth": 6,
            "learning_rate": 0.03,
            "iterations": 800,
        },
        {
            "name": "CatBoost_E",
            "depth": 8,
            "learning_rate": 0.03,
            "iterations": 800,
        },
    ]

    all_results = []

    # --------------------------------------------------------
    # Normal target experiments
    # --------------------------------------------------------

    for config in configurations:

        results = run_experiment(
            df=df,
            experiment_name=config["name"],
            depth=config["depth"],
            learning_rate=config["learning_rate"],
            iterations=config["iterations"],
            use_log_target=False
        )

        all_results.extend(
            results
        )

    # --------------------------------------------------------
    # Summary of normal target models
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    summary = (
        results_df
        .groupby("experiment")
        [["MAE", "RMSE", "R2", "best_iteration"]]
        .mean()
        .reset_index()
    )

    summary = summary.sort_values(
        "MAE"
    )

    print("\n" + "=" * 80)
    print("NORMAL TARGET MODEL SUMMARY")
    print("=" * 80)

    print(
        summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df.to_csv(
        OUTPUT_DIR / "model_experiment_results.csv",
        index=False
    )

    summary.to_csv(
        OUTPUT_DIR / "model_experiment_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # Log-target experiment using the best configuration
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("LOG-TARGET EXPERIMENT")
    print("=" * 80)

    log_results = run_experiment(
        df=df,
        experiment_name="CatBoost_A_LogTarget",
        depth=6,
        learning_rate=0.05,
        iterations=500,
        use_log_target=True
    )

    all_results.extend(
        log_results
    )

    mae_loss_results = run_experiment(
    df=df,
    experiment_name="CatBoost_A_LogTarget_MAELoss",
    depth=6,
    learning_rate=0.05,
    iterations=500,
    use_log_target=True,
    loss_function="MAE"
    )

    log_results_df = pd.DataFrame(
        log_results
    )
    
    mae_loss_df = pd.DataFrame(
        mae_loss_results
    )

    print("\n" + "=" * 80)
    print("LOG TARGET: RMSE LOSS VS MAE LOSS")
    print("=" * 80)

    comparison = pd.concat(
        [
            log_results_df,
            mae_loss_df
        ],
        ignore_index=True
    )

    comparison_summary = (
        comparison
        .groupby("experiment")
        [["MAE", "RMSE", "R2", "best_iteration"]]
        .mean()
        .reset_index()
    )

    print(
        comparison_summary.to_string(
            index=False
        )
    )

    comparison_summary.to_csv(
        OUTPUT_DIR /
        "loss_function_comparison.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save updated results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    log_summary = (
        results_df[
            results_df["experiment"].isin(
                [
                    "CatBoost_A",
                    "CatBoost_A_LogTarget"
                ]
            )
        ]
        .groupby("experiment")
        [["MAE", "RMSE", "R2", "best_iteration"]]
        .mean()
        .reset_index()
    )

    print("\n" + "=" * 80)
    print("NORMAL TARGET VS LOG TARGET")
    print("=" * 80)

    print(
        log_summary.to_string(
            index=False
        )
    )

    log_summary.to_csv(
        OUTPUT_DIR / "log_target_comparison.csv",
        index=False
    )


if __name__ == "__main__":
    main()