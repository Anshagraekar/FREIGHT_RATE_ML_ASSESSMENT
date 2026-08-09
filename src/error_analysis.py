from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor


# ============================================================
# Configuration
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT_DIR / "data" / "train-test.csv"

OUTPUT_DIR = ROOT_DIR / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Features
# ============================================================

FEATURES = [
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
# Load data
# ============================================================

def load_data():

    df = pd.read_csv(
        TRAIN_PATH
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    return df


# ============================================================
# Prepare features
# ============================================================

def prepare_features(df):

    data = df.copy()

    data["date_numeric"] = (
        data["date"]
        - pd.Timestamp("2025-01-01")
    ).dt.days

    X = data[
        FEATURES + ["date_numeric"]
    ].copy()

    y = data["posted_rate"].copy()

    return X, y


# ============================================================
# Train final-like model
# ============================================================

def train_model(
    X_train,
    y_train,
    X_valid,
    y_valid
):

    categorical_columns = (
        X_train
        .select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()
    )

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=False,
        allow_writing_files=False
    )

    model.fit(
        X_train,
        np.log1p(y_train),
        cat_features=categorical_columns,
        eval_set=(
            X_valid,
            np.log1p(y_valid)
        ),
        early_stopping_rounds=80
    )

    return model


# ============================================================
# Main error analysis
# ============================================================

def main():

    df = load_data()

    # --------------------------------------------------------
    # Use October as the most recent internal validation month
    # --------------------------------------------------------

    train_mask = (
        df["date"]
        <= pd.Timestamp("2025-09-30")
    )

    valid_mask = (
        (df["date"] >= pd.Timestamp("2025-10-01"))
        &
        (df["date"] <= pd.Timestamp("2025-10-31"))
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

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train,
        X_valid,
        y_valid
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions_log = model.predict(
        X_valid
    )

    predictions = np.expm1(
        predictions_log
    )

    # --------------------------------------------------------
    # Build error dataframe
    # --------------------------------------------------------

    analysis = valid_df.copy()

    analysis["predicted_rate"] = (
        predictions
    )

    analysis["error"] = (
        analysis["posted_rate"]
        - analysis["predicted_rate"]
    )

    analysis["absolute_error"] = (
        analysis["error"]
        .abs()
    )

    analysis["percentage_error"] = (
        analysis["absolute_error"]
        / analysis["posted_rate"]
        * 100
    )

    # --------------------------------------------------------
    # Sort by largest absolute error
    # --------------------------------------------------------

    worst_predictions = (
        analysis
        .sort_values(
            "absolute_error",
            ascending=False
        )
    )

    print("\n" + "=" * 80)
    print("TOP 20 WORST PREDICTIONS")
    print("=" * 80)

    columns = [
        "load_id",
        "pickup",
        "delivery",
        "equipment",
        "distance",
        "weight",
        "market_index",
        "quote_signal",
        "posted_rate",
        "predicted_rate",
        "error",
        "absolute_error",
        "percentage_error",
    ]

    print(
        worst_predictions[
            columns
        ]
        .head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Equipment-level errors
    # --------------------------------------------------------

    equipment_errors = (
        analysis
        .groupby("equipment")
        .agg(
            count=("posted_rate", "size"),
            MAE=(
                "absolute_error",
                "mean"
            ),
            median_absolute_error=(
                "absolute_error",
                "median"
            )
        )
        .reset_index()
    )

    print("\n" + "=" * 80)
    print("ERROR BY EQUIPMENT")
    print("=" * 80)

    print(
        equipment_errors.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Distance bucket errors
    # --------------------------------------------------------

    analysis["distance_bucket"] = pd.cut(
        analysis["distance"],
        bins=[
            0,
            500,
            1000,
            1500,
            2000,
            3000,
            np.inf
        ],
        labels=[
            "<500",
            "500-1000",
            "1000-1500",
            "1500-2000",
            "2000-3000",
            ">3000"
        ]
    )

    distance_errors = (
        analysis
        .groupby(
            "distance_bucket",
            observed=True
        )
        .agg(
            count=("posted_rate", "size"),
            MAE=(
                "absolute_error",
                "mean"
            ),
            median_absolute_error=(
                "absolute_error",
                "median"
            )
        )
        .reset_index()
    )

    print("\n" + "=" * 80)
    print("ERROR BY DISTANCE")
    print("=" * 80)

    print(
        distance_errors.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    analysis.to_csv(
        OUTPUT_DIR /
        "error_analysis_october.csv",
        index=False
    )

    equipment_errors.to_csv(
        OUTPUT_DIR /
        "error_by_equipment.csv",
        index=False
    )

    distance_errors.to_csv(
        OUTPUT_DIR /
        "error_by_distance.csv",
        index=False
    )

    print(
        "\nError analysis saved to outputs/"
    )


if __name__ == "__main__":
    main()