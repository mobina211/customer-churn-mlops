import pandas as pd
import os
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
        "Zip Code",
        "Churn Score",
        "CLTV",
        "Count", 
        "Latitude",
        "Longitude"
    ]

    df = df.drop(columns=columns_to_drop, errors="ignore")

    # ستون‌های متنی را به عدد تبدیل کن
    categorical_columns = df.select_dtypes(include=["object"]).columns
    categorical_columns = [c for c in categorical_columns if c != "Churn Value"]

    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # تبدیل True/False به 0 و 1
    # تبدیل فقط ستون‌های بولی به عدد
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    # نرمال‌سازی ستون‌های عددی (به جز ستون هدف)
    scaler = MinMaxScaler()

    numeric_cols = df.columns.drop("Churn Value")

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    

    os.makedirs("data/v3", exist_ok=True)

    df.to_excel(
    "data/v3/Telco_customer_churn.xlsx",
    index=False
    )

    print("Feature engineered dataset saved to data/v3/")

    return df