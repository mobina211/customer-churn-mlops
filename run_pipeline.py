from src.preprocessing import preprocess
from src.save_model import save_best_model
from src.data_loader import load_data
from src.features import feature_engineering
from src.train import train_models
from src.evaluate import evaluate_models
from src.mlflow_utils import log_models


def main():

    df = load_data()
    df = preprocess(df)
    df = feature_engineering(df)

    (
        models,
        cv_scores,
        X_valid,
        y_valid,
        X_test,
        y_test
    ) = train_models(df)

    print("\n========== VALIDATION ==========\n")

    validation_results = evaluate_models(
        models,
        X_valid,
        y_valid
    )

    best_model = save_best_model(
        models,
        validation_results
    )

    print("\n========== FINAL TEST ==========\n")

    test_results = evaluate_models(
        models,
        X_test,
        y_test
    )

    log_models(
        models,
        test_results,
        cv_scores
    )

    print(df.columns.tolist())


if __name__ == "__main__":
    main()