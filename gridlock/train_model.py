import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ==========================
# LOAD DATA
# ==========================

train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)

# ==========================
# HANDLE MISSING VALUES
# ==========================

for col in ["RoadType", "Weather"]:
    train[col] = train[col].fillna("Unknown")
    test[col] = test[col].fillna("Unknown")

temp_median = train["Temperature"].median()

train["Temperature"] = train["Temperature"].fillna(temp_median)
test["Temperature"] = test["Temperature"].fillna(temp_median)

# ==========================
# TIMESTAMP FEATURES
# ==========================

train["timestamp"] = pd.to_datetime(
    train["timestamp"],
    format="%H:%M"
)

test["timestamp"] = pd.to_datetime(
    test["timestamp"],
    format="%H:%M"
)

for df in [train, test]:

    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["road_weather"] = (
        df["RoadType"].astype(str)
        + "_"
        + df["Weather"].astype(str)
    )

    df["temp_lane"] = (
        df["Temperature"]
        * df["NumberofLanes"]
    )

# ==========================
# FEATURES
# ==========================

X = train.drop(
    columns=[
        "Index",
        "timestamp",
        "demand"
    ]
)

y = train["demand"]

X_test = test.drop(
    columns=[
        "Index",
        "timestamp"
    ]
)

# ==========================
# CATEGORICAL FEATURES
# ==========================

cat_features = [
    "geohash",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather",
    "road_weather"
]

# ==========================
# TRAIN / VALIDATION SPLIT
# ==========================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# MODEL
# ==========================

model = CatBoostRegressor(
    iterations=1200,
    depth=8,
    learning_rate=0.05,
    loss_function="RMSE",
    eval_metric="R2",
    random_seed=42,
    verbose=100
)

# ==========================
# TRAIN
# ==========================

model.fit(
    X_train,
    y_train,
    eval_set=(X_valid, y_valid),
    cat_features=cat_features,
    use_best_model=True
)

# ==========================
# VALIDATION SCORE
# ==========================

val_preds = model.predict(X_valid)

score = r2_score(
    y_valid,
    val_preds
)

print("\nValidation R2:", score)

# ==========================
# FEATURE IMPORTANCE
# ==========================

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

print("\nTop Features:")
print(
    importance.sort_values(
        "importance",
        ascending=False
    ).head(10)
)

# ==========================
# TEST PREDICTIONS
# ==========================

preds = model.predict(X_test)

submission = pd.DataFrame({
    "Index": test["Index"],
    "demand": preds
})

submission.to_csv(
    "submission_v2.csv",
    index=False
)

print("\nsubmission_v2.csv created successfully")
print("Saved in:", "submission_v2.csv")