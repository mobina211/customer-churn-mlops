import pandas as pd


def preprocess(df):
    # حذف ستون شناسه مشتری
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])

    # تبدیل TotalCharges به عدد
    df["Total Charges"] = pd.to_numeric(
        df["Total Charges"],
        errors="coerce"
    )

    # حذف مقادیر گمشده
    df = df.dropna()

    return df