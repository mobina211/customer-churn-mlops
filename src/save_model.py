import os
import joblib


def save_best_model(models, results, scaler):

    # انتخاب بهترین مدل بر اساس F1 Score
    best_model_name = max(
        results,
        key=lambda x: results[x]["F1 Score"]
    )

    best_model = models[best_model_name]

    os.makedirs("models", exist_ok=True)

    model_path = os.path.join(
        "models",
        "best_model.pkl"
    )

    scaler_path = os.path.join(
        "models",
        "scaler.pkl"
    )

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\nBest model saved successfully!")
    print(f"Model: {best_model_name}")
    print(f"Model Location : {model_path}")
    print(f"Scaler Location: {scaler_path}")

    return best_model_name