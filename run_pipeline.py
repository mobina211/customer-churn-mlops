from src.data_loader import load_data
from src.preprocessing import preprocess
from src.features import feature_engineering
from src.train import train_models


def main():

    df = load_data()

    df = preprocess(df)

    df = feature_engineering(df)

    models, X_test, y_test = train_models(df)

    print("\nTraining Finished!")

    print(models.keys())


if __name__ == "__main__":
    main()