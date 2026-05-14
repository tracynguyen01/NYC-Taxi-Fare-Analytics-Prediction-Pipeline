# Databricks notebook source
# MAGIC %md
# MAGIC 1. Load dataset

# COMMAND ----------

import pandas as pd
X_train = pd.read_parquet("/Volumes/workspace/default/AT2/X_train.parquet")
X_test  = pd.read_parquet("/Volumes/workspace/default/AT2/X_test.parquet")
y_train = pd.read_parquet("/Volumes/workspace/default/AT2/y_train.parquet")["total_amount"]
y_test  = pd.read_parquet("/Volumes/workspace/default/AT2/y_test.parquet")["total_amount"]

# COMMAND ----------

# MAGIC %md
# MAGIC 2. Log Transformation for numerical variable

# COMMAND ----------

import numpy as np

skewed_features = ["trip_distance"]

for col in skewed_features:
    X_train[col] = np.log1p(X_train[col])
    X_test[col]  = np.log1p(X_test[col])

# COMMAND ----------

# MAGIC %md
# MAGIC 3. One-hot encoding for categorical variable

# COMMAND ----------

import pandas as pd

# One-hot encode taxi_color
X_train= pd.get_dummies(X_train, columns=["taxi_color"], drop_first=True)
X_test= pd.get_dummies(X_test,  columns=["taxi_color"], drop_first=True)
# Check new columns
print(X_train.head())

# COMMAND ----------

# MAGIC %md
# MAGIC 4. Build ML model: XGBoost Regressor tuned manually

# COMMAND ----------

# MAGIC %pip install xgboost

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# ==============================
# 1. Sample training data (10%)
# ==============================
X_sample, _, y_sample, _ = train_test_split(
    X_train, y_train, train_size=0.3, random_state=42
)

# Split sample into train/val to tune
X_tr, X_val, y_tr, y_val = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=42
)

print("Train shape:", X_tr.shape, "Val shape:", X_val.shape)

# ==============================
# 2. Baseline params
# ==============================
base_params = {
    "objective": "reg:squarederror",
    "n_estimators": 500,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": -1,
    "random_state": 42
}

# ==============================
# 3. Step 1 – Tune max_depth & min_child_weight
# ==============================
best_rmse = float("inf")
best_depth, best_child = None, None

for max_depth in [4, 6, 8]:
    for min_child_weight in [1, 3, 5]:
        params = {**base_params, "max_depth": max_depth, "min_child_weight": min_child_weight}
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        print(f"max_depth={max_depth}, min_child_weight={min_child_weight} -> RMSE={rmse:.4f}")
        if rmse < best_rmse:
            best_rmse, best_depth, best_child = rmse, max_depth, min_child_weight

print(f"==> Best Step1: max_depth={best_depth}, min_child_weight={best_child}, RMSE={best_rmse:.4f}")

# ==============================
# 4. Step 2 – Tune subsample & colsample_bytree
# ==============================
best_rmse = float("inf")
best_subsample, best_colsample = None, None

for subsample in [0.6, 0.8, 1.0]:
    for colsample in [0.6, 0.8, 1.0]:
        params = {**base_params,
                  "max_depth": best_depth, "min_child_weight": best_child,
                  "subsample": subsample, "colsample_bytree": colsample}
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        print(f"subsample={subsample}, colsample={colsample} -> RMSE={rmse:.4f}")
        if rmse < best_rmse:
            best_rmse, best_subsample, best_colsample = rmse, subsample, colsample

print(f"==> Best Step2: subsample={best_subsample}, colsample={best_colsample}, RMSE={best_rmse:.4f}")

# ==============================
# 5. Step 3 – Tune learning_rate
# ==============================
best_rmse = float("inf")
best_lr = None

for lr in [0.05, 0.1, 0.2]:
    params = {**base_params,
              "max_depth": best_depth, "min_child_weight": best_child,
              "subsample": best_subsample, "colsample_bytree": best_colsample,
              "learning_rate": lr, "n_estimators": 700}
    model = XGBRegressor(**params)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"learning_rate={lr} -> RMSE={rmse:.4f}")
    if rmse < best_rmse:
        best_rmse, best_lr = rmse, lr

print(f"==> Best Step3: learning_rate={best_lr}, RMSE={best_rmse:.4f}")

# ==============================
# 6. Final model (train trên toàn bộ X_train, y_train)
# ==============================
final_params = {**base_params,
                "max_depth": best_depth, "min_child_weight": best_child,
                "subsample": best_subsample, "colsample_bytree": best_colsample,
                "learning_rate": best_lr, "n_estimators": 700}

final_model = XGBRegressor(**final_params)
final_model.fit(X_train, y_train)

# Evaluate on test set
y_test_pred = final_model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
print("==> Final Test RMSE:", rmse_test)
