from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


SEED = 42


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=SEED
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=SEED
        ),

        "XGBoost": XGBClassifier(
            random_state=SEED,
            eval_metric="logloss"
        ),

        "CatBoost": CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            verbose=0,
            random_state=SEED
        )
    }


def train_models(df):

    # Target
    y = df["Churn Value"]

    # Features
    X = df.drop(columns=["Churn Value"])

    # Train / Validation / Test Split

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.20,
        random_state=SEED,
        stratify=y_train_val
    )

    # Scaling (فقط روی Train)

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)

    X_valid = scaler.transform(X_valid)

    X_test = scaler.transform(X_test)

    # Cross Validation

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED
    )

    models = get_models()

    trained_models = {}

    cv_scores = {}

    print("\n========== TRAINING ==========\n")

    for name, model in models.items():

        print(f"Training {name}")

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy"
        )

        cv_scores[name] = scores.mean()

        model.fit(X_train, y_train)

        trained_models[name] = model

        print(f"CV Accuracy = {scores.mean():.4f}")

    return (
        trained_models,
        cv_scores,
        scaler,
        X_valid,
        y_valid,
        X_test,
        y_test
    )