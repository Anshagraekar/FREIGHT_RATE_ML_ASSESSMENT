from pathlib import Path
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

DECEMBER_PATH = (
    ROOT_DIR / "data" / "december-chart-inputs.csv"
)


def main():

    print("=" * 80)
    print("DECEMBER DATA INSPECTION")
    print("=" * 80)

    df = pd.read_csv(
        DECEMBER_PATH
    )

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))

    print("\nMissing values:")
    print(
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    if "load_id" in df.columns:

        print("\nDuplicate load IDs:")
        print(
            df["load_id"].duplicated().sum()
        )

    if "date" in df.columns:

        print("\nDate range:")

        dates = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        print("Minimum:", dates.min())
        print("Maximum:", dates.max())


if __name__ == "__main__":
    main()