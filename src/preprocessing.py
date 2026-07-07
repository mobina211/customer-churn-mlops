import pandas as pd


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

    return df