import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt

MODEL_PATH = "model.pkl"
FEATURES_PATH = "features.txt"


print("Loading model...")
model, scaler = joblib.load(MODEL_PATH)


if os.path.exists(FEATURES_PATH):
    with open(FEATURES_PATH, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
else:
    feature_names = None


print("Model type:", type(model))


if hasattr(model, "model"):
    print("\nBest Estimator:", model.model.estimator)


if hasattr(model, "best_config"):
    print("\nBest Hyperparameters:")
    for key, val in model.best_config.items():
        print(f"  {key}: {val}")


if hasattr(model, "best_loss"):
    print(f"\nBest Loss: {model.best_loss:.4f}")


if hasattr(model, "best_estimator"):
    print(f"\nBest Learner: {model.best_estimator}")


if hasattr(model, "training_history"):
    print(f"\nTotal Training Trials: {len(model.training_history)}")
    print("Sample Trial:")
    print(model.training_history[0] if len(model.training_history) > 0 else "None")


if hasattr(model.model, "feature_importances_"):
    print("\nFeature Importance:")
    importance = model.model.feature_importances_

    if feature_names and len(feature_names) == len(importance):
        feat_imp = pd.Series(importance, index=feature_names)
    else:
        feat_imp = pd.Series(importance)

    feat_imp = feat_imp.sort_values(ascending=False)
    print(feat_imp)

    feat_imp.head(15).plot(kind="barh", title="Top 15 Feature Importances", figsize=(10, 6))
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

