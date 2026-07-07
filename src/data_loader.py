import pandas as pd


def load_data():
    file_path = "data/v1/Telco_customer_churn.xlsx"

    df = pd.read_excel(file_path)

    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    return df


if __name__ == "__main__":
    load_data()