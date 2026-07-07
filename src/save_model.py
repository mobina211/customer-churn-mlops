import os
import joblib


def save_best_model(models, results):

    # انتخاب بهترین مدل بر اساس F1 Score
    best_model_name = max(
        results,
        key=lambda x: results[x]["F1 Score"]
    )

    best_model = models[best_model_name]

    os.makedirs("models", exist_ok=True)

    save_path = os.path.join("models", "best_model.pkl")

    joblib.dump(best_model, save_path)

    print(f"\nBest model saved successfully!")
    print(f"Model: {best_model_name}")
    print(f"Location: {save_path}")

    return best_model_name