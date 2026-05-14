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
# MAGIC 2. Explore dataset

# COMMAND ----------

# MAGIC %md
# MAGIC 2.1. Numerical variables

# COMMAND ----------

X_train.describe()

# COMMAND ----------

#Check null
num_cols = ["trip_distance","passenger_count","month","hour","dow", "payment_type","RatecodeID","PULocationID","DOLocationID"]
null_counts = X_train[num_cols].isna().sum()
print(null_counts[null_counts > 0])  # show only columns with missing values

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

discrete_features = ["passenger_count", "payment_type"]
for col in discrete_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=X_train[col], y=y_train)
    plt.title(f"{col} vs total_amount")
    plt.xlabel(col)
    plt.ylabel("total_amount")
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 2.2. Categorical variable

# COMMAND ----------

print(X_train["taxi_color"].value_counts())

# COMMAND ----------

print(X_train["taxi_color"].isna().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC 3. Data preparation for Modeling

# COMMAND ----------

# MAGIC %md
# MAGIC 3.1. Check and handle skewness (numerical variables)

# COMMAND ----------

num_cols = ["trip_distance","passenger_count","month","hour","dow", "payment_type","RatecodeID","PULocationID","DOLocationID"]
skewness = X_train[num_cols].skew().sort_values(ascending=False)
print(skewness)

# COMMAND ----------

import numpy as np

skewed_features = ["trip_distance"]

for col in skewed_features:
    X_train[col] = np.log1p(X_train[col])
    X_test[col]  = np.log1p(X_test[col])

# COMMAND ----------

# MAGIC %md
# MAGIC 3.2. One-hot encoding (categorical variables)

# COMMAND ----------

import pandas as pd

# One-hot encode taxi_color
X_train= pd.get_dummies(X_train, columns=["taxi_color"], drop_first=True)
X_test= pd.get_dummies(X_test,  columns=["taxi_color"], drop_first=True)
# Check new columns
print(X_train.head())

# COMMAND ----------

# MAGIC %md
# MAGIC 4. Baseline model Performance

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Baseline evaluation on clipped dataset (correct grouping)
# MAGIC CREATE OR REPLACE TABLE baseline_eval_clipped AS
# MAGIC WITH baseline AS (
# MAGIC   SELECT
# MAGIC     taxi_color,
# MAGIC     PULocationID,
# MAGIC     DOLocationID,
# MAGIC     month,
# MAGIC     dow,
# MAGIC     hour,
# MAGIC     ROUND(AVG(total_amount), 2) AS avg_amount_per_trip
# MAGIC   FROM workspace.default.trips_clipped_ml
# MAGIC   GROUP BY taxi_color, PULocationID, DOLocationID, month, dow, hour
# MAGIC )
# MAGIC SELECT
# MAGIC   t.total_amount AS actual,
# MAGIC   b.avg_amount_per_trip AS predicted
# MAGIC FROM workspace.default.trips_clipped_ml t
# MAGIC LEFT JOIN baseline b
# MAGIC   ON t.taxi_color   = b.taxi_color
# MAGIC  AND t.PULocationID = b.PULocationID
# MAGIC  AND t.DOLocationID = b.DOLocationID
# MAGIC  AND t.month        = b.month
# MAGIC  AND t.dow          = b.dow
# MAGIC  AND t.hour         = b.hour
# MAGIC WHERE t.year = 2024 AND t.month IN (10, 11, 12);  -- Oct–Dec 2024

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
baseline_eval_df = spark.table("baseline_eval_clipped").toPandas()
rmse_baseline = np.sqrt(mean_squared_error(
    baseline_eval_df["actual"],
    baseline_eval_df["predicted"]
))
print("Baseline RMSE:", rmse_baseline)

# COMMAND ----------

# MAGIC %md
# MAGIC 5. Build ML model: XGBoost

# COMMAND ----------

# MAGIC %pip install xgboost

# COMMAND ----------

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import numpy as np
from xgboost import XGBRegressor
# Take 30% sample of training data 
X_train_sample = X_train.sample(frac=0.3, random_state=42)
y_train_sample = y_train.loc[X_train_sample.index]

print("Train sample size:", X_train_sample.shape, y_train_sample.shape)

# Define the XGBoost model
xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
# Build pipeline (no preprocessing if already numeric)
pipe_xgb = Pipeline([
    ("model", xgb)
])
# ---- Cross-validation RMSE ----
scores = cross_val_score(
    pipe_xgb,
    X_train_sample, y_train_sample,
    scoring="neg_root_mean_squared_error",
    cv=5,
    n_jobs=-1
)
rmse_cv = -scores.mean()

# ---- Fit final model on sample ----
pipe_xgb.fit(X_train_sample, y_train_sample)

# Train RMSE
y_train_pred = pipe_xgb.predict(X_train_sample)
rmse_train = np.sqrt(mean_squared_error(y_train_sample, y_train_pred))

# Test RMSE (still full test set for fairness)
y_test_pred = pipe_xgb.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

# ---- Print results ----
print(f"[XGB baseline] Train RMSE (30% sample): {rmse_train:.4f}")
print(f"[XGB baseline] CV RMSE (5-fold, 30% sample): {rmse_cv:.4f}")
print(f"[XGB baseline] Test RMSE: {rmse_test:.4f}")