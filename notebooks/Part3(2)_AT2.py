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
# MAGIC 2. Handle skewness of numerical variables

# COMMAND ----------

import numpy as np

skewed_features = ["trip_distance"]

for col in skewed_features:
    X_train[col] = np.log1p(X_train[col])
    X_test[col]  = np.log1p(X_test[col])

# COMMAND ----------

# MAGIC %md
# MAGIC 3. Import libraries

# COMMAND ----------

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC 4. Build ML model: ElasticNet

# COMMAND ----------

# Take 10% training sample (avoid OOM) ====
X_train_sample = X_train.sample(frac=0.1, random_state=42)
y_train_sample = y_train.loc[X_train_sample.index]

print("Train sample size:", X_train_sample.shape, y_train_sample.shape)

# COMMAND ----------

# Define feature types ====
num_cols = ["trip_distance", "passenger_count"]
cat_cols = ["taxi_color", "payment_type", "RatecodeID",
            "PULocationID", "DOLocationID", "dow", "hour", "month"]
# Preprocessing: scale numeric + one-hot categorical
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols)
    ]
)

# COMMAND ----------

# ElasticNet model (alpha & l1_ratio can be tuned)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=10000)

# Pipeline 
pipe_elastic = Pipeline([
    ("preprocessor", preprocessor),
    ("model", elastic)
])

# Cross-validation (5-fold) on sample 
scores = cross_val_score(
    pipe_elastic,
    X_train_sample, y_train_sample,
    scoring="neg_root_mean_squared_error",
    cv=5,
    n_jobs=-1
)
rmse_cv = -scores.mean()
print(f"[ElasticNet] CV RMSE (5-fold, 10% sample): {rmse_cv:.4f}")

# Train final model (fit trên sample)
pipe_elastic.fit(X_train_sample, y_train_sample)

# Predict + evaluate 
# Train RMSE
y_train_pred = pipe_elastic.predict(X_train_sample)
rmse_train = np.sqrt(mean_squared_error(y_train_sample, y_train_pred))

# Test RMSE (use full test set: Oct–Dec 2024)
y_test_pred = pipe_elastic.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"[ElasticNet] Train RMSE (10% sample): {rmse_train:.4f}")
print(f"[ElasticNet] Test RMSE (full test set): {rmse_test:.4f}")