from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT_DIR / "data" / "train-test.csv"
VALIDATION_PATH = ROOT_DIR / "data" / "validation.csv"

OUTPUT_DIR = ROOT_DIR / "outputs" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utility functions
# ============================================================

def load_data():
    """Load development and validation datasets."""

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    return train_df, validation_df


def print_dataset_overview(df, name):
    """Print basic dataset information."""

    print("\n" + "=" * 70)
    print(f"{name} DATASET OVERVIEW")
    print("=" * 70)

    print(f"Shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())


def missing_value_report(df, name):
    """Generate missing-value report."""

    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    report = pd.DataFrame({
        "missing_count": missing,
        "missing_percentage": missing_pct
    })

    report = report[report["missing_count"] > 0]
    report = report.sort_values("missing_count", ascending=False)

    print("\n" + "=" * 70)
    print(f"{name} MISSING VALUES")
    print("=" * 70)

    if report.empty:
        print("No missing values found.")
    else:
        print(report)

    report.to_csv(
        OUTPUT_DIR / f"{name.lower()}_missing_values.csv"
    )


def duplicate_report(df, name):
    """Check duplicate rows and IDs."""

    print("\n" + "=" * 70)
    print(f"{name} DUPLICATE CHECK")
    print("=" * 70)

    print(f"Duplicate rows: {df.duplicated().sum()}")

    if "load_id" in df.columns:
        print(f"Duplicate load_id: {df['load_id'].duplicated().sum()}")


def invalid_value_report(df, name):
    """Check potentially invalid numerical values."""

    print("\n" + "=" * 70)
    print(f"{name} DATA QUALITY CHECK")
    print("=" * 70)

    if "weight" in df.columns:
        negative_weight = (df["weight"] < 0).sum()
        zero_weight = (df["weight"] == 0).sum()

        print(f"Negative weights: {negative_weight}")
        print(f"Zero weights: {zero_weight}")

    if "distance" in df.columns:
        negative_distance = (df["distance"] < 0).sum()
        zero_distance = (df["distance"] == 0).sum()

        print(f"Negative distances: {negative_distance}")
        print(f"Zero distances: {zero_distance}")

    if "posted_rate" in df.columns:
        negative_rate = (df["posted_rate"] < 0).sum()
        zero_rate = (df["posted_rate"] == 0).sum()

        print(f"Negative posted rates: {negative_rate}")
        print(f"Zero posted rates: {zero_rate}")


# ============================================================
# Target analysis
# ============================================================

def analyze_target(df):
    """Analyze posted_rate distribution."""

    if "posted_rate" not in df.columns:
        return

    target = df["posted_rate"].dropna()

    print("\n" + "=" * 70)
    print("TARGET ANALYSIS")
    print("=" * 70)

    print(target.describe())

    print("\nTarget quantiles:")
    print(
        target.quantile(
            [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        )
    )

    # Histogram
    plt.figure(figsize=(10, 6))

    plt.hist(target, bins=60)

    plt.xlabel("Posted Rate")
    plt.ylabel("Frequency")
    plt.title("Distribution of Posted Freight Rate")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "target_distribution.png",
        dpi=150
    )

    plt.close()


# ============================================================
# Numerical analysis
# ============================================================

def numerical_analysis(df):
    """Analyze numerical variables and correlations."""

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    print("\n" + "=" * 70)
    print("NUMERICAL SUMMARY")
    print("=" * 70)

    print(df[numeric_columns].describe().T)

    if "posted_rate" in df.columns:

        correlations = (
            df[numeric_columns]
            .corr()["posted_rate"]
            .sort_values(ascending=False)
        )

        print("\nCorrelation with posted_rate:")
        print(correlations)

        correlations.to_csv(
            OUTPUT_DIR / "target_correlations.csv"
        )


# ============================================================
# Distance analysis
# ============================================================

def distance_analysis(df):
    """Analyze relationship between distance and posted rate."""

    if "distance" not in df.columns:
        return

    if "posted_rate" not in df.columns:
        return

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df["distance"],
        df["posted_rate"],
        alpha=0.15,
        s=10
    )

    plt.xlabel("Distance")
    plt.ylabel("Posted Rate")
    plt.title("Distance vs Posted Rate")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "distance_vs_rate.png",
        dpi=150
    )

    plt.close()


# ============================================================
# Equipment analysis
# ============================================================

def equipment_analysis(df):
    """Analyze rates by equipment type."""

    if "equipment" not in df.columns:
        return

    if "posted_rate" not in df.columns:
        return

    summary = (
        df.groupby("equipment")["posted_rate"]
        .agg(["count", "mean", "median", "std"])
        .sort_values("mean", ascending=False)
    )

    print("\n" + "=" * 70)
    print("EQUIPMENT ANALYSIS")
    print("=" * 70)

    print(summary)

    summary.to_csv(
        OUTPUT_DIR / "equipment_summary.csv"
    )

    plt.figure(figsize=(8, 5))

    df.boxplot(
        column="posted_rate",
        by="equipment"
    )

    plt.title("Posted Rate by Equipment")
    plt.suptitle("")
    plt.xlabel("Equipment")
    plt.ylabel("Posted Rate")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "rate_by_equipment.png",
        dpi=150
    )

    plt.close()


# ============================================================
# Date analysis
# ============================================================

def date_analysis(df):
    """Analyze temporal patterns."""

    if "date" not in df.columns:
        return

    if "posted_rate" not in df.columns:
        return

    temp = df.copy()

    temp["date"] = pd.to_datetime(
        temp["date"],
        errors="coerce"
    )

    temp["month"] = temp["date"].dt.to_period("M").astype(str)

    monthly = (
        temp.groupby("month")["posted_rate"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )

    print("\n" + "=" * 70)
    print("MONTHLY RATE ANALYSIS")
    print("=" * 70)

    print(monthly)

    monthly.to_csv(
        OUTPUT_DIR / "monthly_rate_summary.csv",
        index=False
    )

    plt.figure(figsize=(11, 6))

    plt.plot(
        monthly["month"],
        monthly["mean"],
        marker="o"
    )

    plt.xlabel("Month")
    plt.ylabel("Average Posted Rate")
    plt.title("Average Posted Rate by Month")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "monthly_rate.png",
        dpi=150
    )

    plt.close()


# ============================================================
# Route analysis
# ============================================================

def route_analysis(df):
    """Analyze pickup-delivery routes."""

    if "pickup" not in df.columns:
        return

    if "delivery" not in df.columns:
        return

    if "posted_rate" not in df.columns:
        return

    temp = df.copy()

    temp["route"] = (
        temp["pickup"].astype(str)
        + " -> "
        + temp["delivery"].astype(str)
    )

    route_summary = (
        temp.groupby("route")["posted_rate"]
        .agg(["count", "mean", "median"])
        .sort_values("count", ascending=False)
    )

    print("\n" + "=" * 70)
    print("ROUTE ANALYSIS")
    print("=" * 70)

    print("\nTop 20 routes by number of loads:")
    print(route_summary.head(20))

    route_summary.to_csv(
        OUTPUT_DIR / "route_summary.csv"
    )

def geographic_distance_analysis(df):
    """Analyze supplied distance vs Haversine distance."""

    required_columns = [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "distance"
    ]

    if not all(
        column in df.columns
        for column in required_columns
    ):
        return

    from features import haversine_distance

    temp = df.copy()

    temp["haversine_distance"] = haversine_distance(
        temp["pickup_lat"],
        temp["pickup_lon"],
        temp["delivery_lat"],
        temp["delivery_lon"]
    )

    temp["distance_ratio"] = (
        temp["distance"]
        / temp["haversine_distance"]
    )

    print("\n" + "=" * 70)
    print("GEOGRAPHIC DISTANCE ANALYSIS")
    print("=" * 70)

    print("\nProvided distance:")
    print(temp["distance"].describe())

    print("\nHaversine distance:")
    print(temp["haversine_distance"].describe())

    print("\nDistance ratio:")
    print(temp["distance_ratio"].describe())

    print("\nCorrelation:")
    print(
        temp[
            [
                "distance",
                "haversine_distance",
                "distance_ratio"
            ]
        ].corr()
    )

    temp[
        [
            "distance",
            "haversine_distance",
            "distance_ratio"
        ]
    ].to_csv(
        OUTPUT_DIR / "geographic_distance_analysis.csv",
        index=False
    )

    # Plot
    plt.figure(figsize=(9, 6))

    plt.scatter(
        temp["haversine_distance"],
        temp["distance"],
        alpha=0.15,
        s=10
    )

    plt.xlabel("Haversine Distance (miles)")
    plt.ylabel("Provided Distance (miles)")
    plt.title(
        "Provided Distance vs Haversine Distance"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "distance_comparison.png",
        dpi=150
    )

    plt.close()

# ============================================================
# Main
# ============================================================

def main():

    train_df, validation_df = load_data()

    print_dataset_overview(
        train_df,
        "TRAIN / DEVELOPMENT"
    )

    print_dataset_overview(
        validation_df,
        "VALIDATION"
    )

    missing_value_report(
        train_df,
        "train"
    )

    missing_value_report(
        validation_df,
        "validation"
    )

    duplicate_report(
        train_df,
        "TRAIN"
    )

    duplicate_report(
        validation_df,
        "VALIDATION"
    )

    invalid_value_report(
        train_df,
        "TRAIN"
    )

    invalid_value_report(
        validation_df,
        "VALIDATION"
    )

    analyze_target(train_df)

    numerical_analysis(train_df)

    distance_analysis(train_df)

    geographic_distance_analysis(train_df)
    
    equipment_analysis(train_df)

    date_analysis(train_df)

    route_analysis(train_df)

    print("\n" + "=" * 70)
    print("EDA COMPLETE")
    print("=" * 70)

    print(f"EDA outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()