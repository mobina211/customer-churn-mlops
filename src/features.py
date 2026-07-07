import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def feature_engineering(df):
    df = df.copy()

    # حذف ستون‌های غیرضروری
    columns_to_drop = [
        "Churn Label",
        "Churn Reason",
        "Lat Long",
        "Country",
        "State",
        "City",
        "Zip Code"
    ]

    df = df.drop(columns=columns_to_drop, errors="ignore")

    # ستون‌های متنی را به عدد تبدیل کن
    categorical_columns = df.select_dtypes(include=["object"]).columns
    categorical_columns = [c for c in categorical_columns if c != "Churn Value"]

    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # تبدیل True/False به 0 و 1
    df = df.astype(int)

    # نرمال‌سازی ستون‌های عددی (به جز ستون هدف)
    scaler = MinMaxScaler()

    numeric_cols = df.columns.drop("Churn Value")

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df