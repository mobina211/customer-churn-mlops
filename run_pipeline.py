from src.data_loader import load_data
from src.preprocessing import preprocess


def main():
    # بارگذاری داده
    df = load_data()

    # پیش پردازش
    df = preprocess(df)

    print("\nPreprocessing finished!")
    print(df.head())

    # ذخیره نسخه دوم دیتاست
    df.to_excel(
        "data/v2/Telco_customer_churn.xlsx",
        index=False
    )


if __name__ == "__main__":
    main()