# Databricks notebook source
# MAGIC %md
# MAGIC 1. Check top 10 trips in both year (2023 and 2024)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   taxi_color,
# MAGIC   pu_borough,
# MAGIC   do_borough,
# MAGIC   COALESCE(tpep_pickup_datetime,  lpep_pickup_datetime)  AS pickup_ts,
# MAGIC   COALESCE(tpep_dropoff_datetime, lpep_dropoff_datetime) AS dropoff_ts,
# MAGIC   ROUND(total_amount, 2)                                 AS total_amount,
# MAGIC   trip_distance                                          AS distance_miles,
# MAGIC   ROUND(trip_distance * 1.60934, 2)                      AS distance_km
# MAGIC FROM final_trips
# MAGIC WHERE total_amount IS NOT NULL
# MAGIC   AND YEAR(COALESCE(tpep_pickup_datetime, lpep_pickup_datetime)) IN (2023, 2024)
# MAGIC ORDER BY total_amount DESC, pickup_ts DESC
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC 2. Check average total amount of each year and both year (2023 and 2024)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Averages for 2023, 2024, and both years
# MAGIC WITH base AS (
# MAGIC   SELECT
# MAGIC     YEAR(COALESCE(tpep_pickup_datetime, lpep_pickup_datetime)) AS yr,
# MAGIC     COALESCE(tpep_pickup_datetime,  lpep_pickup_datetime)  AS pu_ts,
# MAGIC     COALESCE(tpep_dropoff_datetime, lpep_dropoff_datetime) AS do_ts,
# MAGIC     total_amount,
# MAGIC     trip_distance * 1.60934 AS distance_km
# MAGIC   FROM final_trips
# MAGIC   WHERE YEAR(COALESCE(tpep_pickup_datetime, lpep_pickup_datetime)) IN (2023, 2024)
# MAGIC ),
# MAGIC calc AS (
# MAGIC   SELECT
# MAGIC     yr,
# MAGIC     total_amount,
# MAGIC     distance_km,
# MAGIC     (CAST(do_ts AS BIGINT) - CAST(pu_ts AS BIGINT)) / 60.0           AS duration_min,
# MAGIC     CASE
# MAGIC       WHEN (CAST(do_ts AS BIGINT) - CAST(pu_ts AS BIGINT)) > 0
# MAGIC         THEN distance_km / ((CAST(do_ts AS BIGINT) - CAST(pu_ts AS BIGINT)) / 3600.0)
# MAGIC     END AS speed_kmh
# MAGIC   FROM base
# MAGIC )
# MAGIC SELECT
# MAGIC   CASE WHEN GROUPING(yr) = 1 THEN '2023-2024' ELSE CAST(yr AS STRING) END AS year_group,
# MAGIC   COUNT(*)                                           AS trips,
# MAGIC   ROUND(AVG(total_amount), 2)                        AS avg_total_amount,
# MAGIC   ROUND(AVG(distance_km), 2)                         AS avg_distance_km,
# MAGIC   ROUND(AVG(duration_min), 2)                        AS avg_duration_min,
# MAGIC   ROUND(AVG(speed_kmh), 2)                           AS avg_speed_kmh
# MAGIC FROM calc
# MAGIC GROUP BY GROUPING SETS ((yr), ())
# MAGIC ORDER BY CASE year_group WHEN '2023' THEN 1 WHEN '2024' THEN 2 ELSE 3 END;

# COMMAND ----------

# MAGIC %md
# MAGIC 3. Create clipped dataset

# COMMAND ----------

# MAGIC %sql
# MAGIC with calc as (
# MAGIC   SELECT
# MAGIC     trip_distance,
# MAGIC     (CAST(COALESCE(tpep_dropoff_datetime,lpep_dropoff_datetime) AS BIGINT) -
# MAGIC      CAST(COALESCE(tpep_pickup_datetime, lpep_pickup_datetime)  AS BIGINT)) AS dur_sec,
# MAGIC     total_amount
# MAGIC   FROM final_trips
# MAGIC   WHERE trip_distance IS NOT NULL AND total_amount IS NOT NULL
# MAGIC )
# MAGIC SELECT
# MAGIC   percentile_approx(trip_distance, 0.01) AS d_p01,
# MAGIC   percentile_approx(trip_distance, 0.99) AS d_p99,
# MAGIC   percentile_approx(dur_sec/60.0, 0.99)  AS t_p99,
# MAGIC   percentile_approx(total_amount, 0.99)  AS amt_p99
# MAGIC FROM calc
# MAGIC WHERE trip_distance >= 0.1 AND dur_sec > 0;

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH base AS (
# MAGIC   SELECT
# MAGIC     *,
# MAGIC     (CAST(COALESCE(tpep_dropoff_datetime,lpep_dropoff_datetime) AS BIGINT) -
# MAGIC      CAST(COALESCE(tpep_pickup_datetime, lpep_pickup_datetime)  AS BIGINT)) AS dur_sec
# MAGIC   FROM final_trips
# MAGIC   WHERE trip_distance >= 0.1            -- my basic quality filter
# MAGIC     AND (CAST(COALESCE(tpep_dropoff_datetime,lpep_dropoff_datetime) AS BIGINT) -
# MAGIC          CAST(COALESCE(tpep_pickup_datetime, lpep_pickup_datetime)  AS BIGINT)) > 0
# MAGIC ),
# MAGIC calc AS (
# MAGIC   SELECT
# MAGIC     *,
# MAGIC     dur_sec/60.0 AS duration_min,
# MAGIC     trip_distance * 1.60934 AS distance_km
# MAGIC   FROM base
# MAGIC )
# MAGIC SELECT
# MAGIC   taxi_color, pu_borough, do_borough,
# MAGIC   -- clip distance in miles to [0.30, 18.90]
# MAGIC   LEAST(GREATEST(trip_distance, 0.30), 18.90) AS trip_distance_mi_clip,
# MAGIC   -- clip duration in minutes to [1, 57.98] (set 1 to avoid ultra-small durations)
# MAGIC   LEAST(GREATEST(duration_min, 1.0), 57.98)   AS duration_min_clip,
# MAGIC   -- clip total_amount to [0, 75.41]
# MAGIC   LEAST(GREATEST(total_amount, 0.0), 75.41)   AS total_amount_clip,
# MAGIC   *
# MAGIC FROM calc;

# COMMAND ----------

# MAGIC %md
# MAGIC 4. Define train set and test set. Convert them to dataframe

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE workspace.default.trips_clipped_ml AS
# MAGIC WITH base AS (
# MAGIC   SELECT
# MAGIC     -- raw fields
# MAGIC     trip_distance,
# MAGIC     total_amount,
# MAGIC     passenger_count,
# MAGIC     taxi_color,
# MAGIC     payment_type,
# MAGIC     RatecodeID,
# MAGIC     PULocationID,
# MAGIC     DOLocationID,
# MAGIC     tpep_pickup_datetime,
# MAGIC     lpep_pickup_datetime,
# MAGIC     tpep_dropoff_datetime,
# MAGIC     lpep_dropoff_datetime,
# MAGIC     -- duration in seconds (coalesce yellow/green schemas)
# MAGIC     (CAST(COALESCE(tpep_dropoff_datetime, lpep_dropoff_datetime) AS BIGINT) -
# MAGIC      CAST(COALESCE(tpep_pickup_datetime,  lpep_pickup_datetime)  AS BIGINT)) AS dur_sec
# MAGIC   FROM workspace.default.final_trips
# MAGIC   WHERE trip_distance IS NOT NULL AND total_amount IS NOT NULL
# MAGIC ),
# MAGIC calc AS (
# MAGIC   SELECT
# MAGIC     *,
# MAGIC     dur_sec/60.0 AS duration_min,
# MAGIC     -- apply clipping using your thresholds
# MAGIC     LEAST(GREATEST(trip_distance, 0.30), 18.90)     AS trip_distance_mi_clip,
# MAGIC     LEAST(GREATEST(dur_sec/60.0,   1.00), 57.983333) AS duration_min_clip,
# MAGIC     LEAST(GREATEST(total_amount,   0.00), 75.41)     AS total_amount_clip
# MAGIC   FROM base
# MAGIC   WHERE trip_distance >= 0.1 AND dur_sec > 0
# MAGIC )
# MAGIC SELECT
# MAGIC   -- target (clipped)
# MAGIC   total_amount_clip            AS total_amount,
# MAGIC   -- numeric features (clipped)
# MAGIC   trip_distance_mi_clip        AS trip_distance,
# MAGIC   duration_min_clip            AS duration_min,
# MAGIC   passenger_count,
# MAGIC   -- categorical features
# MAGIC   taxi_color, payment_type, RatecodeID, PULocationID, DOLocationID,
# MAGIC   -- time features from pickup
# MAGIC   YEAR(COALESCE(tpep_pickup_datetime, lpep_pickup_datetime))   AS year,
# MAGIC   MONTH(COALESCE(tpep_pickup_datetime, lpep_pickup_datetime))  AS month,
# MAGIC   HOUR(COALESCE(tpep_pickup_datetime, lpep_pickup_datetime))   AS hour,
# MAGIC   dayofweek(COALESCE(tpep_pickup_datetime, lpep_pickup_datetime)) AS dow
# MAGIC FROM calc;

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

from pyspark.sql import functions as F

# Load the clipped ML table
df_clip = spark.table("workspace.default.trips_clipped_ml")

# Train set = Jan 2023 → Sep 2024
trips_clipped_train = df_clip.filter(
    ((F.col("year") == 2023) & (F.col("month").between(1, 12))) |
    ((F.col("year") == 2024) & (F.col("month") <= 9))
)

# Test set = Oct–Dec 2024
trips_clipped_test = df_clip.filter(
    (F.col("year") == 2024) & (F.col("month").isin([10, 11, 12]))
)

# ---- Sanity checks ----
display(trips_clipped_train.groupBy("year").count().orderBy("year"))
display(trips_clipped_test.groupBy("year","month").count().orderBy("year","month"))

display("Train rows:", trips_clipped_train.count())
display("Test rows:",  trips_clipped_test.count())

# COMMAND ----------

# MAGIC %md
# MAGIC 5. 

# COMMAND ----------

import pandas as pd

num_cols = ["trip_distance","passenger_count","month","hour","dow"]
cat_cols = ["taxi_color","payment_type","RatecodeID","PULocationID","DOLocationID"]
target   = "total_amount"

# ---- Numeric + target (convert once, then split in Pandas) ----
train_num_pd = trips_clipped_train.select(num_cols + [target]).toPandas()
test_num_pd  = trips_clipped_test.select(num_cols + [target]).toPandas()

# Pop target out so we don't keep two copies
y_train = train_num_pd.pop(target)
y_test  = test_num_pd.pop(target)

X_train_num = train_num_pd
X_test_num  = test_num_pd

# ---- Categorical only (as category dtype to save RAM) ----
X_train_cat = trips_clipped_train.select(cat_cols).toPandas()
X_test_cat  = trips_clipped_test.select(cat_cols).toPandas()
for c in cat_cols:
    X_train_cat[c] = X_train_cat[c].astype("category")
    X_test_cat[c]  = X_test_cat[c].astype("category")

# ---- Align indices and combine (row-wise concat) ----
X_train_num = X_train_num.reset_index(drop=True)
X_test_num  = X_test_num.reset_index(drop=True)
X_train_cat = X_train_cat.reset_index(drop=True)
X_test_cat  = X_test_cat.reset_index(drop=True)

X_train = pd.concat([X_train_num, X_train_cat], axis=1)
X_test  = pd.concat([X_test_num,  X_test_cat],  axis=1)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# COMMAND ----------

display(X_train.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC 6.

# COMMAND ----------

X_train.info()

# COMMAND ----------

num_convert = ["payment_type", "RatecodeID", "PULocationID", "DOLocationID"]

for col in num_convert:
    X_train[col] = X_train[col].astype("int32")
    X_test[col]  = X_test[col].astype("int32")

# Double check
print(X_train[num_convert].dtypes)

# COMMAND ----------

X_train["passenger_count"] = X_train["passenger_count"].astype("int32")
X_test["passenger_count"]  = X_test["passenger_count"].astype("int32")

# COMMAND ----------

y_test.info()

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME AT2
# MAGIC COMMENT 'Volume for AT2 data';

# COMMAND ----------

X_train.to_parquet("/Volumes/workspace/default/AT2/X_train.parquet")
X_test.to_parquet("/Volumes/workspace/default/AT2/X_test.parquet")
y_train.to_frame("total_amount").to_parquet("/Volumes/workspace/default/AT2/y_train.parquet")
y_test.to_frame("total_amount").to_parquet("/Volumes/workspace/default/AT2/y_test.parquet")
