from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]


# ============================================================
# Geographic feature
# ============================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate great-circle distance between two coordinates.

    Returns distance in miles.
    """

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    earth_radius_miles = 3958.7613

    return earth_radius_miles * c


# ============================================================
# Feature engineering
# ============================================================

def create_features(df):
    """
    Create features for the freight-rate model.

    This function does not perform model-specific encoding
    or imputation. Those steps belong in the modeling pipeline.
    """

    data = df.copy()

    # --------------------------------------------------------
    # Date features
    # --------------------------------------------------------

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day_of_month"] = data["date"].dt.day
    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_year"] = data["date"].dt.dayofyear
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)

    # Days since beginning of dataset
    data["days_since_start"] = (
        data["date"] - pd.Timestamp("2025-01-01")
    ).dt.days

    # --------------------------------------------------------
    # Cyclic date features
    # --------------------------------------------------------

    data["month_sin"] = np.sin(
        2 * np.pi * data["month"] / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * data["month"] / 12
    )

    data["day_of_week_sin"] = np.sin(
        2 * np.pi * data["day_of_week"] / 7
    )

    data["day_of_week_cos"] = np.cos(
        2 * np.pi * data["day_of_week"] / 7
    )

    # --------------------------------------------------------
    # Route feature
    # --------------------------------------------------------

    data["route"] = (
        data["pickup"].astype(str)
        + "__"
        + data["delivery"].astype(str)
    )

    # --------------------------------------------------------
    # Geographic features
    # --------------------------------------------------------

    data["haversine_distance"] = haversine_distance(
        data["pickup_lat"],
        data["pickup_lon"],
        data["delivery_lat"],
        data["delivery_lon"]
    )

    # Difference in coordinates
    data["lat_difference"] = (
        data["delivery_lat"]
        - data["pickup_lat"]
    )

    data["lon_difference"] = (
        data["delivery_lon"]
        - data["pickup_lon"]
    )

    # --------------------------------------------------------
    # Weight quality features
    # --------------------------------------------------------

    data["weight_missing"] = (
        data["weight"].isna()
        | (data["weight"] <= 0)
    ).astype(int)

    # Convert invalid weights to missing
    data.loc[
        data["weight"] <= 0,
        "weight"
    ] = np.nan

    # Weight per mile
    data["weight_per_mile"] = (
        data["weight"]
        / data["distance"].replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Market data quality
    # --------------------------------------------------------

    data["market_index_missing"] = (
        data["market_index"].isna()
    ).astype(int)

    # --------------------------------------------------------
    # Remove raw date
    # --------------------------------------------------------

    data.drop(
        columns=["date"],
        inplace=True
    )

    return data