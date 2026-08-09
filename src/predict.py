from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor


# ============================================================
# Configuration
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT_DIR / "data" / "train-test.csv"
VALIDATION_PATH = ROOT_DIR / "data" / "validation.csv"

OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
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
    "date_numeric",
    "market_index",
    "quote_signal",
]


# ============================================================
# Load data
# ============================================================

def load_data(path):

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    return df


# ============================================================
# Feature preparation
# ============================================================

def prepare_features(df):

    data = df.copy()

    data["date_numeric"] = (
        data["date"]
        - pd.Timestamp("2025-01-01")
    ).dt.days

    return data[FEATURES].copy()


# ============================================================
# Train final model
# ============================================================

def train_final_model(train_df):

    X_train = prepare_features(
        train_df
    )

    y_train = train_df[
        "posted_rate"
    ].copy()

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
        iterations=240,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=100,
        allow_writing_files=False
    )

    # Train on log-transformed target
    model.fit(
        X_train,
        np.log1p(y_train),
        cat_features=categorical_columns
    )

    model_path = (
        MODEL_DIR /
        "catboost_final_log_target.cbm"
    )

    model.save_model(
        model_path
    )

    print(
        f"\nModel saved to:\n{model_path}"
    )

    return model


# ============================================================
# Generate predictions
# ============================================================

def generate_predictions(
    model,
    df
):

    X = prepare_features(
        df
    )

    predictions_log = (
        model.predict(X)
    )

    predictions = np.expm1(
        predictions_log
    )

    predictions = np.maximum(
        predictions,
        0
    )

    return predictions


# ============================================================
# Validation predictions
# ============================================================

def create_validation_predictions(
    model,
    validation_df
):

    predictions = generate_predictions(
        model,
        validation_df
    )

    output = pd.DataFrame({
        "load_id": validation_df[
            "load_id"
        ],
        "posted_rate": predictions
    })

    output.to_csv(
        OUTPUT_DIR /
        "validation_predictions.csv",
        index=False
    )

    print(
        "\nValidation predictions saved:"
    )

    print(
        OUTPUT_DIR /
        "validation_predictions.csv"
    )

    print(
        f"Rows: {len(output):,}"
    )

    print(
        output.head()
    )

    return output


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("FINAL FREIGHT RATE MODEL")
    print("=" * 80)

    # --------------------------------------------------------
    # Load development data
    # --------------------------------------------------------

    train_df = load_data(
        TRAIN_PATH
    )

    print(
        f"\nDevelopment data: "
        f"{train_df.shape}"
    )

    # --------------------------------------------------------
    # Train final model
    # --------------------------------------------------------

    model = train_final_model(
        train_df
    )

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    validation_df = load_data(
        VALIDATION_PATH
    )

    print(
        f"\nValidation data: "
        f"{validation_df.shape}"
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    create_validation_predictions(
        model,
        validation_df
    )

    print(
        "\nPrediction pipeline complete."
    )


if __name__ == "__main__":
    main()