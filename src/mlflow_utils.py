import mlflow
import mlflow.sklearn


def log_models(models, results, cv_scores):

    mlflow.set_experiment("Customer Churn Prediction")

    DATASET_VERSION = "v3"
    RANDOM_SEED = 42

    for model_name, model in models.items():

        with mlflow.start_run(run_name=model_name):

            # پارامترها
            mlflow.log_param("Model", model_name)
            mlflow.log_param("Dataset Version", DATASET_VERSION)
            mlflow.log_param("Random Seed", RANDOM_SEED)

            # هایپرپارامترها
            params = model.get_params()

            for key, value in params.items():
                try:
                    mlflow.log_param(key, value)
                except:
                    pass

            # Cross Validation
            mlflow.log_metric("CV Accuracy", cv_scores[model_name])

            # سایر متریک‌ها
            for metric_name, metric_value in results[model_name].items():

                if metric_name != "Confusion Matrix":
                    mlflow.log_metric(metric_name, metric_value)

            # فقط مدل‌های sklearn ذخیره شوند
            if model_name in ["Logistic Regression", "Random Forest"]:

                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model"
                )

    print("\nMLflow logging completed.")