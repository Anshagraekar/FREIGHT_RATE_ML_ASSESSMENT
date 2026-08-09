from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor



# ============================================================
# Configuration
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DEVELOPMENT_PATH = (
    ROOT_DIR / "data" / "train-test.csv"
)

VALIDATION_PATH = (
    ROOT_DIR / "data" / "validation.csv"
)

DECEMBER_PATH = (
    ROOT_DIR / "data" / "december-chart-inputs.csv"
)

MODEL_PATH = (
    ROOT_DIR
    / "outputs"
    / "models"
    / "catboost_final_log_target.cbm"
)

OUTPUT_PATH = (
    ROOT_DIR
    / "data"
    / "december-chart-inputs.csv"
)

# ============================================================
# City coordinate lookup
# ============================================================

def build_city_coordinates(development):

    coordinates = {}

    # Pickup coordinates
    pickup_coords = (
        development[
            [
                "pickup",
                "pickup_lat",
                "pickup_lon"
            ]
        ]
        .dropna()
        .drop_duplicates(
            subset=["pickup"]
        )
    )

    for _, row in pickup_coords.iterrows():

        coordinates[row["pickup"]] = (
            row["pickup_lat"],
            row["pickup_lon"]
        )

    # Delivery coordinates
    delivery_coords = (
        development[
            [
                "delivery",
                "delivery_lat",
                "delivery_lon"
            ]
        ]
        .dropna()
        .drop_duplicates(
            subset=["delivery"]
        )
    )

    for _, row in delivery_coords.iterrows():

        city = row["delivery"]

        if city not in coordinates:

            coordinates[city] = (
                row["delivery_lat"],
                row["delivery_lon"]
            )

    return coordinates


# ============================================================
# Add coordinates
# ============================================================

def add_coordinates(df, coordinates):

    data = df.copy()

    data["pickup_lat"] = (
        data["pickup"]
        .map(
            lambda x: coordinates[x][0]
            if x in coordinates
            else np.nan
        )
    )

    data["pickup_lon"] = (
        data["pickup"]
        .map(
            lambda x: coordinates[x][1]
            if x in coordinates
            else np.nan
        )
    )

    data["delivery_lat"] = (
        data["delivery"]
        .map(
            lambda x: coordinates[x][0]
            if x in coordinates
            else np.nan
        )
    )

    data["delivery_lon"] = (
        data["delivery"]
        .map(
            lambda x: coordinates[x][1]
            if x in coordinates
            else np.nan
        )
    )

    return data


# ============================================================
# Add December market features
# ============================================================

def add_market_features(
    december,
    validation
):

    data = december.copy()

    validation = validation.copy()

    validation["date"] = pd.to_datetime(
        validation["date"],
        errors="coerce"
    )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    # Only use December observations
    december_market = validation[
        validation["date"].dt.month == 12
    ].copy()

    # Daily market conditions
    daily_market = (
        december_market
        .groupby("date")
        .agg(
            market_index=(
                "market_index",
                "mean"
            ),
            quote_signal=(
                "quote_signal",
                "mean"
            )
        )
        .reset_index()
    )

    data = data.merge(
        daily_market,
        on="date",
        how="left"
    )

    return data


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("DECEMBER FREIGHT RATE PREDICTION")
    print("=" * 80)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    development = pd.read_csv(
        DEVELOPMENT_PATH
    )

    validation = pd.read_csv(
        VALIDATION_PATH
    )

    december = pd.read_csv(
        DECEMBER_PATH
    )

    print("\nDevelopment data:")
    print(development.shape)

    print("\nValidation data:")
    print(validation.shape)

    print("\nDecember scenario:")
    print(december.shape)

    # --------------------------------------------------------
    # Build coordinate lookup
    # --------------------------------------------------------

    coordinates = build_city_coordinates(
        development
    )

    # --------------------------------------------------------
    # Add coordinates
    # --------------------------------------------------------

    december = add_coordinates(
        december,
        coordinates
    )

    # --------------------------------------------------------
    # Check coordinates
    # --------------------------------------------------------

    coordinate_columns = [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon"
    ]

    print("\nCoordinate check:")

    print(
        december[
            coordinate_columns
        ].isna().sum()
    )

    # --------------------------------------------------------
    # Add market features
    # --------------------------------------------------------

    december = add_market_features(
        december,
        validation
    )

    print("\nMarket feature check:")

    print(
        december[
            [
                "market_index",
                "quote_signal"
            ]
        ]
        .isna()
        .sum()
    )

    december["date"] = pd.to_datetime(
    december["date"],
    errors="coerce"
    )

    december["date_numeric"] = (
        december["date"]
        - pd.Timestamp("2025-01-01")
    ).dt.days

    # --------------------------------------------------------
    # Prepare exact model features
    # --------------------------------------------------------

    december["date"] = pd.to_datetime(
        december["date"],
        errors="coerce"
    )

    december["date_numeric"] = (
        december["date"]
        - pd.Timestamp("2025-01-01")
    ).dt.days

    model_features = [
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

    X_december = december[
        model_features
    ].copy()

    print("\nModel feature shape:")
    print(X_december.shape)

    print("\nModel features:")
    print(X_december.columns.tolist())

    # --------------------------------------------------------
    # Load final model
    # --------------------------------------------------------

    print("\nLoading final model:")

    model = CatBoostRegressor()

    model.load_model(
        MODEL_PATH
    )

    print(
        "Model loaded:",
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Match training feature order
    # --------------------------------------------------------


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    log_predictions = model.predict(
        X_december
    )

    predictions = np.expm1(
        log_predictions
    )

    # Prevent tiny numerical negative values
    predictions = np.maximum(
        predictions,
        0
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    result = december[
        [
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "date"
        ]
    ].copy()

    result["date"] = pd.to_datetime(
        result["date"]
    ).dt.strftime(
        "%Y-%m-%d"
    )

    result["predicted_rate"] = predictions

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("DECEMBER PREDICTIONS")
    print("=" * 80)

    print(
        result.to_string(
            index=False
        )
    )

    print("\nPrediction statistics:")

    print(
        result["predicted_rate"]
        .describe()
        .to_string()
    )

    print(
        "\nPredictions saved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()