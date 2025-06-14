import pandas as pd
import numpy as np
import os
import joblib
from flaml import AutoML
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "Lap_Time_Seconds"
MODEL_DIR = "flaml_motogp_model"
TRAIN_PATH = "train.csv"
VAL_PATH = "val.csv"
TEST_PATH = "test.csv"
DROP_COLS = ["Unique ID", "Rider_name", "Team_name", "Bike_name", "Shortname", "Circuit_name"]

def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {path} with shape {df.shape}")
    return df

def preprocess(df, target_col=None):
    df = df.copy()
    df.columns = df.columns.str.strip()

    
    for col in DROP_COLS:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    
    if target_col and target_col in df.columns:
        target = df[target_col]
    else:
        target = None

    
    for col in df.select_dtypes(include='object').columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            df[col] = df[col].astype('category').cat.codes

    
    df.fillna(df.median(numeric_only=True), inplace=True)

    if target is not None:
        df[target_col] = target

    return df

def train_flaml(X, y):
    automl = AutoML()

    settings = {
        "time_budget": 6600,  
        "metric": 'rmse',
        "task": 'regression',
        "estimator_list": ['lgbm'],
        "log_file_name": "motogp_flaml.log",
        "starting_points": {
            "lgbm": {
                "n_estimators": 50,
                "num_leaves": 11736,
                "min_child_samples": 3,
                "learning_rate": 0.29899735606397315,
                "log_max_bin": 7,
                "colsample_bytree": 0.345753331470006,
                "reg_alpha": 0.0009765625,
                "reg_lambda": 0.03998672339022226,
            }
        }
    }

    automl.fit(X_train=X, y_train=y, **settings)
    print("Best Estimator:", automl.model.estimator)
    print("Best Hyperparameters:", automl.best_config)
    return automl

def save_model(automl, scaler, feature_order):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump((automl, scaler), os.path.join(MODEL_DIR, "model.pkl"))
    with open(os.path.join(MODEL_DIR, "features.txt"), 'w') as f:
        for feat in feature_order:
            f.write(f"{feat}\n")

def load_model():
    model, scaler = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    with open(os.path.join(MODEL_DIR, "features.txt"), 'r') as f:
        feature_order = [line.strip() for line in f.readlines()]
    return model, scaler, feature_order

def evaluate(model, X, y):
    preds = model.predict(X)
    rmse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"📊 RMSE: {rmse:.4f}")
    print(f"📈 R² Score: {r2:.4f}")

def main():
    print("🔹 Loading training data...")
    train_df = load_data(TRAIN_PATH)
    train_df = preprocess(train_df, TARGET_COLUMN)

    y_train = train_df[TARGET_COLUMN]
    X_train = train_df.drop(columns=[TARGET_COLUMN])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("Training FLAML model...")
    model = train_flaml(X_train_scaled, y_train)
    save_model(model, scaler, X_train.columns)

    print("Model trained and saved.")

    print("Validating on validation data...")
    val_df = load_data(VAL_PATH)
    val_df = preprocess(val_df, TARGET_COLUMN)
    y_val = val_df[TARGET_COLUMN]
    X_val = val_df.drop(columns=[TARGET_COLUMN])
    X_val_scaled = scaler.transform(X_val)
    evaluate(model, X_val_scaled, y_val)

    print("Predicting on test data...")
    test_df = load_data(TEST_PATH)
    test_processed = preprocess(test_df)
    for col in X_train.columns:
        if col not in test_processed.columns:
            test_processed[col] = 0
    test_processed = test_processed[X_train.columns]
    X_test_scaled = scaler.transform(test_processed)

    preds = model.predict(X_test_scaled)
    result_df = test_df.copy()
    result_df["Lap_Time_Seconds_Predicted"] = preds
    result_df.to_csv("test_predictions.csv", index=False)

    print("Predictions saved to 'test_predictions.csv'")

if __name__ == "__main__":
    main()
