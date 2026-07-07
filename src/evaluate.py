from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_models(models, X_test, y_test):

    results = {}

    print("\n========== MODEL EVALUATION ==========\n")

    best_model = None
    best_f1 = 0

    for name, model in models.items():

        y_pred = model.predict(X_test)

        # احتمال کلاس مثبت
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        if y_prob is not None:
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            roc_auc = 0

        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": roc_auc,
            "Confusion Matrix": cm
        }

        print(f"\n{name}")
        print("-" * 40)
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC AUC  : {roc_auc:.4f}")
        print("Confusion Matrix:")
        print(cm)

        if f1 > best_f1:
            best_f1 = f1
            best_model = name

    print("\n===================================")
    print(f"Best Model: {best_model}")
    print(f"Best F1 Score: {best_f1:.4f}")
    print("===================================\n")

    return results