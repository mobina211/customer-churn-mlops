import pandas as pd
import os


def preprocess(df):

    # حذف شناسه مشتری
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])

    # حذف ستون Churn Reason قبل از حذف Missing Values
    if "Churn Reason" in df.columns:
        df = df.drop(columns=["Churn Reason"])

    # تبدیل Total Charges به عدد
    df["Total Charges"] = pd.to_numeric(
        df["Total Charges"],
        errors="coerce"
    )

    # فقط ردیف‌هایی را حذف کن که Total Charges ندارند
    df = df.dropna(subset=["Total Charges"])

    # Save preprocessed dataset (v2)

    os.makedirs("data/v2", exist_ok=True)

    df.to_excel(
    "data/v2/Telco_customer_churn.xlsx",
    index=False
     )

    print("Preprocessed dataset saved to data/v2/")

    return df