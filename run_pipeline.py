from src.data_loader import load_data
from src.preprocessing import preprocess
from src.features import feature_engineering
from src.train import train_models
from src.evaluate import evaluate_models


def main():

    df = load_data()

    df = preprocess(df)

    df = feature_engineering(df)

    models, X_test, y_test = train_models(df)

    results = evaluate_models(models, X_test, y_test)

    print("\n========== RESULTS ==========\n")

    for model_name, metrics in results.items():

        print(f"\n{model_name}")

        for key, value in metrics.items():
            print(f"{key}:")
            print(value)


if __name__ == "__main__":
    main()