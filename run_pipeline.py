from src.data_loader import load_data
from src.preprocessing import preprocess
from src.features import feature_engineering


def main():

    df = load_data()

    df = preprocess(df)

    df = feature_engineering(df)

    print("\nFeature Engineering finished!")
    print(df.head())

    df.to_excel(
        "data/v3/Telco_customer_churn.xlsx",
        index=False
    )


if __name__ == "__main__":
    main()