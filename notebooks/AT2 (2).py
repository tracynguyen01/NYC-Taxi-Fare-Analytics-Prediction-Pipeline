# Databricks notebook source
# MAGIC %md
# MAGIC **Part 1. Data Ingestion and Preparation**

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Import dataset

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("AT2(2)").getOrCreate()

# COMMAND ----------

df = spark.table("workspace.default.taxi_zone_lookup")
display(df)

# COMMAND ----------

# Paths
GREEN_DST = "/Volumes/workspace/bde/assignment2/green/green_taxi.parquet"
YELLOW_DST = "/Volumes/workspace/bde/assignment2/yellow/yellow_taxi.parquet"

# Load into DataFrames
green_df = spark.read.parquet(GREEN_DST)
yellow_df = spark.read.parquet(YELLOW_DST)

# Peek at data
display(green_df.limit(10))
display(yellow_df.limit(10))

# COMMAND ----------

#Copy the original dataset into a new DataFrame, so we can keep the original data for later use
green_raw  = green_df
yellow_raw = yellow_df

# COMMAND ----------

# MAGIC %md
# MAGIC 2. Clean dataset

# COMMAND ----------

# MAGIC %md
# MAGIC 2.1. Trips finishing before the starting time

# COMMAND ----------

from pyspark.sql.functions import unix_timestamp

# Green taxis
green_duration = green_raw.withColumn(
    "trip_duration_sec",
    unix_timestamp("lpep_dropoff_datetime") - unix_timestamp("lpep_pickup_datetime")
)

# COMMAND ----------

#Trips finishing before the starting time 
green_negative = green_duration.filter(green_duration.trip_duration_sec <= 0)

# COMMAND ----------

count = green_negative.count()
print(count)
print(f"{count/green_duration.count():.2%}")

# COMMAND ----------

#Trips finishing before the starting time 
#Yellow taxis
yellow_duration = yellow_raw.withColumn(
    "trip_duration_sec",
    unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime")
)

yellow_negative = yellow_duration.filter(yellow_duration.trip_duration_sec <= 0)

# COMMAND ----------

count_yel = yellow_negative.count()
print(count_yel)
print(f"{count_yel/yellow_duration.count():.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC 2.2. Trips where the pickup/dropoff datetime is outside of the range

# COMMAND ----------

from pyspark.sql.functions import col, min as smin, max as smax, lit

#Extract the min, max datatime of each dataset
# Green min/max
green_stats = (
    green_raw
    .select(
        smin(col("lpep_pickup_datetime")).alias("min_pickup"),
        smax(col("lpep_pickup_datetime")).alias("max_pickup"),
        smin(col("lpep_dropoff_datetime")).alias("min_dropoff"),
        smax(col("lpep_dropoff_datetime")).alias("max_dropoff"),
    )
    .withColumn("dataset", lit("green"))
)

# Yellow min/max
yellow_stats = (
    yellow_raw
    .select(
        smin(col("tpep_pickup_datetime")).alias("min_pickup"),
        smax(col("tpep_pickup_datetime")).alias("max_pickup"),
        smin(col("tpep_dropoff_datetime")).alias("min_dropoff"),
        smax(col("tpep_dropoff_datetime")).alias("max_dropoff"),
    )
    .withColumn("dataset", lit("yellow"))
)

# Combine and display
time_ranges = green_stats.unionByName(yellow_stats).select(
    "dataset", "min_pickup", "max_pickup", "min_dropoff", "max_dropoff"
)

display(time_ranges)

# COMMAND ----------

from pyspark.sql.functions import col

# Define calendar for min/max datatime of each dataset
YELLOW_MIN = "2009-01-01 00:00:00"
YELLOW_MAX = "2024-12-31 23:59:59"
GREEN_MIN  = "2013-08-01 00:00:00"
GREEN_MAX  = "2024-12-31 23:59:59"

def filter_datatime(green_raw, pickup_col, drop_col, start_ts, end_ts):
  kept = green_raw.filter( (col(pickup_col) >= start_ts) & (col(drop_col) <= end_ts) )
  removed = green_raw.filter( (col(pickup_col) <  start_ts) | (col(drop_col) >  end_ts) )
  return kept, removed

# Yellow
yellow_inside, yellow_outside = filter_datatime(
    yellow_raw, "tpep_pickup_datetime", "tpep_dropoff_datetime",
    YELLOW_MIN, YELLOW_MAX)

#Green
green_inside, green_outside = filter_datatime(
    green_raw, "lpep_pickup_datetime", "lpep_dropoff_datetime",
    GREEN_MIN, GREEN_MAX)

# Quick counts so we can verify removals are small (<10%)
print("yellow outside rows:", yellow_outside.count())
print("green outside rows:", green_outside.count())
print(f"Green  outside: {green_outside.count()/ green_raw.count():%}")
print(f"Yellow outside: {yellow_outside.count()/ yellow_raw.count():%}")

# COMMAND ----------

# MAGIC %md
# MAGIC 2.3. Trips with negative speed

# COMMAND ----------

#Trips with negative speed
from pyspark.sql import functions as F

def add_speed(dist_col = 'trip_distance', time_col = 'trip_duration_sec'):
    return (F.col(dist_col) * F.lit(3600.0))/ F.col(time_col)

#Green
green_neg_speed = (
  green_duration
  .withColumn("avg_speed", F.when(F.col("trip_duration_sec") > 0, add_speed()).otherwise(F.lit(None))).filter( (F.col("trip_duration_sec") > 0) & (F.col("avg_speed") <= 0))
)

#Yellow
yellow_neg_speed = (
  yellow_duration
  .withColumn("avg_speed", F.when(F.col("trip_duration_sec") > 0, add_speed()).otherwise(F.lit(None))).filter( (F.col("trip_duration_sec") > 0) & (F.col("avg_speed") <= 0))
)

print(f"Green negative speed: {green_neg_speed.count()} rows, {green_neg_speed.count()/green_raw.count():%}")
print(f"Yellow negative speed: {yellow_neg_speed.count()} rows, {yellow_neg_speed.count()/yellow_raw.count():%}")

# COMMAND ----------

# MAGIC %md
# MAGIC 2.4. Trips with very high speed (look for NYC and outside of NYC speed limit )

# COMMAND ----------

from pyspark.sql import functions as F

# --- speed caps (mph) ---
CITY_CAP_MPH    = 50.0   # inside NYC
OUTSIDE_CAP_MPH = 65.0   # outside NYC / EWR

NYC_BOROUGHS = ("Manhattan","Brooklyn","Queens","Bronx","Staten Island")

#Zone lookup
zones = (spark.table("workspace.default.taxi_zone_lookup").select("LocationID","Borough","service_zone"))

def add_inside_outside(df, pu_loc_col, do_loc_col):
    pu = zones.select(
        F.col("LocationID").alias("PULocationID_flag"),
        F.col("Borough").alias("pu_borough"),
        F.col("service_zone").alias("pu_service_zone"),
    )
    do = zones.select(
        F.col("LocationID").alias("DOLocationID_flag"),
        F.col("Borough").alias("do_borough"),
        F.col("service_zone").alias("do_service_zone"),
    )
    j = (df.join(pu, df[pu_loc_col] == pu["PULocationID_flag"], "left").drop("PULocationID_flag")
           .join(do, df[do_loc_col] == do["DOLocationID_flag"], "left").drop("DOLocationID_flag"))

    in_nyc  = j["pu_borough"].isin(*NYC_BOROUGHS) & j["do_borough"].isin(*NYC_BOROUGHS)
    outside = (~in_nyc) | (F.upper(F.coalesce(j["pu_service_zone"], F.lit(""))) == "EWR") | \
                           (F.upper(F.coalesce(j["do_service_zone"], F.lit(""))) == "EWR")
    return j.withColumn("trip_in_nyc", in_nyc).withColumn("trip_outside", outside)

def flag_high_speed(df_with_sec, pu_loc_col, do_loc_col):
    base = df_with_sec.withColumn("avg_mph", add_speed())

    j = add_inside_outside(base, pu_loc_col, do_loc_col)

    high_in_city  = (F.col("trip_in_nyc"))  & (F.col("trip_duration_sec") > 0) & (F.col("avg_mph") > CITY_CAP_MPH)
    high_out_city = (F.col("trip_outside")) & (F.col("trip_duration_sec") > 0) & (F.col("avg_mph") > OUTSIDE_CAP_MPH)

    return j.filter(high_in_city), j.filter(high_out_city), j

# Apply to the DFs that ALREADY have trip_duration_sec
g_high_city, g_high_out, g_all = flag_high_speed(green_duration,  "PULocationID", "DOLocationID")
y_high_city, y_high_out, y_all = flag_high_speed(yellow_duration, "PULocationID", "DOLocationID")

print(f"GREEN high-speed (inside): {g_high_city.count()} ({g_high_city.count()/green_raw.count():%})")
print(f"GREEN high-speed (outside): {g_high_out.count()} ({g_high_out.count()/green_raw.count():%})")
print(f"YELLOW high-speed (inside): {y_high_city.count()} ({y_high_city.count()/yellow_raw.count():%})")
print(f"YELLOW high-speed (outside): {y_high_out.count()} ({y_high_out.count()/yellow_raw.count():%})")

# COMMAND ----------

# MAGIC %md
# MAGIC 2.5. Trips that are travelling too short or too long (duration wise)

# COMMAND ----------

from pyspark.sql import functions as F

# thresholds in seconds
MIN_SEC = 60              # >= 1 minute
MAX_SEC = 10 * 3600        # <= 10 hours

def flag_duration_outliers(df):
    too_short = (F.col("trip_duration_sec") > 0) & (F.col("trip_duration_sec") < MIN_SEC)
    too_long  = (F.col("trip_duration_sec") > 0) & (F.col("trip_duration_sec") > MAX_SEC)

    flagged = (df
               .withColumn("too_short_duration", too_short)
               .withColumn("too_long_duration",  too_long)
               .withColumn("duration_minutes", F.round(F.col("trip_duration_sec")/60.0, 1))
              )

    outliers = flagged.filter(F.col("too_short_duration") | F.col("too_long_duration"))
    cleaned  = flagged.filter(~(F.col("too_short_duration") | F.col("too_long_duration")))

    return flagged, outliers, cleaned

# Apply to DataFrames that ALREADY have trip_duration_sec
g_flagged, g_duration_bad, g_duration_ok = flag_duration_outliers(green_duration)
y_flagged, y_duration_bad, y_duration_ok = flag_duration_outliers(yellow_duration)

# Counts & percentages
g_bad = g_duration_bad.count()
y_bad = y_duration_bad.count()

print(f"GREEN duration outliers: {g_bad} rows ({g_bad/green_raw.count():%})")
print(f"YELLOW duration outliers: {y_bad} rows ({y_bad/yellow_raw.count():%})")

# COMMAND ----------

# MAGIC %md
# MAGIC 2.6. Trips that are travelling too short or too long (distance wise)

# COMMAND ----------

from pyspark.sql import functions as F

MIN_MILES = 0.1
MAX_MILES = 100.0

def flag_distance_outliers(df, dist_col="trip_distance", duration_col="trip_duration_sec"):
    pos = F.col(dist_col) > 0

    flagged = (
        df
        .withColumn(
            "too_short_distance",
            pos & (F.col(dist_col) < F.lit(MIN_MILES))
        )
        .withColumn(
            "too_long_distance",  pos & (F.col(dist_col) > F.lit(MAX_MILES))
        )
    )
    outliers = flagged.filter(F.col("too_short_distance") | F.col("too_long_distance"))
    cleaned  = flagged.filter(~(F.col("too_short_distance") | F.col("too_long_distance")))
    return flagged, outliers, cleaned

# Apply to the DFs we already have (with trip_duration_sec)
g_flag_dist, g_dist_bad, g_dist_ok = flag_distance_outliers(green_duration)
y_flag_dist, y_dist_bad, y_dist_ok = flag_distance_outliers(yellow_duration)

# Counts & percentages
# Use the ORIGINAL DataFrames as denominators (before any filtering)
g_total = green_raw.count()    # original green
y_total = yellow_raw.count()   # original yellow

# g_dist_bad, y_dist_bad  (distance-wise too short/too long)
g_bad = g_dist_bad.count()
y_bad = y_dist_bad.count()

print(f"GREEN distance outliers:  {g_bad} rows  ({g_bad/g_total:%})")
print(f"YELLOW distance outliers: {y_bad} rows  ({y_bad/y_total:%})")


# COMMAND ----------

# MAGIC %md
# MAGIC 2.7. Check invalid passengers

# COMMAND ----------

# Check invalid passengers
invalid_passengers_green = green_duration.filter(F.col("passenger_count") <= 0)
invalid_passengers_yellow = yellow_duration.filter(F.col("passenger_count") <= 0)

print(f"Green invalid passenger rows: {invalid_passengers_green.count()} ({invalid_passengers_green.count()/green_raw.count():%})")
print(f"Yellow invalid passenger rows: {invalid_passengers_yellow.count()} ({invalid_passengers_yellow.count()/yellow_raw.count():%})")

# COMMAND ----------

# MAGIC %md
# MAGIC 2.8. Trips with total amount <= 0 

# COMMAND ----------

# Green: total_amount <= 0
invalid_total_green = green_duration.filter(F.col("total_amount") <= 0)
count_invalid_green = invalid_total_green.count()

# Yellow: total_amount <= 0
invalid_total_yellow = yellow_duration.filter(F.col("total_amount") <= 0)
count_invalid_yellow = invalid_total_yellow.count()

print(f"Green invalid total_amount rows: {count_invalid_green} ({count_invalid_green/green_raw.count():%})")
print(f"Yellow invalid total_amount rows: {count_invalid_yellow} ({count_invalid_yellow/yellow_raw.count():%})")


# COMMAND ----------

# MAGIC %md
# MAGIC 2.9. Trips out of RatecodeID

# COMMAND ----------

from pyspark.sql import functions as F

VALID_RATECODES = [1, 2, 3, 4, 5, 6, 99]

# Green
invalid_rate_green = green_duration.filter(~F.col("RatecodeID").isin(VALID_RATECODES) | F.col("RatecodeID").isNull())
print(f"Green invalid RatecodeID rows: {invalid_rate_green.count()} ({invalid_rate_green.count()/green_raw.count():%})")

# Yellow
invalid_rate_yellow = yellow_duration.filter(~F.col("RatecodeID").isin(VALID_RATECODES) | F.col("RatecodeID").isNull())
print(f"Yellow invalid RatecodeID rows: {invalid_rate_yellow.count()} ({invalid_rate_yellow.count()/yellow_raw.count():%})")

# COMMAND ----------

# MAGIC %md
# MAGIC 2.12. Drop all invalid trips

# COMMAND ----------

# === Collect invalid rows → union by trip_key → drop them (no cache) ===
from pyspark.sql import functions as F
from functools import reduce

def with_trip_key(df, pickup_col, drop_col, round_dist=4, round_money=2):
    fp_cols = F.concat_ws(
        "||",
        F.date_format(F.col(pickup_col), "yyyy-MM-dd HH:mm:ss"),
        F.date_format(F.col(drop_col),  "yyyy-MM-dd HH:mm:ss"),
        F.coalesce(F.col("PULocationID").cast("string"), F.lit("")),
        F.coalesce(F.col("DOLocationID").cast("string"), F.lit("")),
        F.coalesce(F.round(F.col("trip_distance"), round_dist).cast("string"), F.lit("")),
        F.coalesce(F.col("passenger_count").cast("string"), F.lit("")),
        F.coalesce(F.round(F.col("fare_amount"),  2).cast("string"), F.lit("")),
        F.coalesce(F.round(F.col("total_amount"), 2).cast("string"), F.lit(""))
    )
    return (
        df.withColumn("trip_fp", fp_cols)
          .withColumn("trip_key", F.sha2(F.col("trip_fp"), 256))
    )

def build_invalid_keys(invalid_dfs, pickup_col, drop_col):
    if not invalid_dfs:
        return None
    keyed = [with_trip_key(x, pickup_col, drop_col).select("trip_key") for x in invalid_dfs]
    return reduce(lambda a, b: a.unionByName(b), keyed).distinct()

def drop_invalid(base_df, invalid_keys):
    if invalid_keys is None:
        return base_df.drop(*[c for c in ("trip_fp","trip_key") if c in base_df.columns])
    # broadcast without caching
    out = (base_df.join(F.broadcast(invalid_keys), on="trip_key", how="left_anti")
                 .drop("trip_fp","trip_key"))
    return out

# 1) Keyed bases ---
g_base = with_trip_key(green_duration,  "lpep_pickup_datetime", "lpep_dropoff_datetime")
y_base = with_trip_key(yellow_duration, "tpep_pickup_datetime", "tpep_dropoff_datetime")

# 2) Gather ALL invalid subsets (as defined earlier in your notebook) ---
g_invalid_sets = [
    green_negative, green_outside, green_neg_speed,
    g_duration_bad, g_dist_bad, g_high_city, g_high_out,
    invalid_passengers_green, invalid_total_green, invalid_rate_green
]
y_invalid_sets = [
    yellow_negative, yellow_outside, yellow_neg_speed,
    y_duration_bad, y_dist_bad, y_high_city, y_high_out,
    invalid_passengers_yellow, invalid_total_yellow, invalid_rate_yellow
]

# 3) Build distinct key sets per taxi ---
g_invalid_keys = build_invalid_keys(g_invalid_sets, "lpep_pickup_datetime", "lpep_dropoff_datetime")
y_invalid_keys = build_invalid_keys(y_invalid_sets, "tpep_pickup_datetime", "tpep_dropoff_datetime")

# 4) Anti-join by key to get final valid datasets 
green_valid  = drop_invalid(g_base, g_invalid_keys)
yellow_valid = drop_invalid(y_base, y_invalid_keys)


# COMMAND ----------

# MAGIC %md
# MAGIC 2.13. Check duplicate rows

# COMMAND ----------

from pyspark.sql import Window as W

def show_duplicate_rows(df, pickup_col, drop_col, name="Taxi"):
    # Build a signature for grouping
    sig = F.concat_ws(
        "||",
        F.date_format(F.col(pickup_col), "yyyy-MM-dd HH:mm:ss"),
        F.date_format(F.col(drop_col),  "yyyy-MM-dd HH:mm:ss"),
        F.coalesce(F.col("PULocationID").cast("string"), F.lit("")),
        F.coalesce(F.col("DOLocationID").cast("string"), F.lit("")),
        F.round(F.col("trip_distance"), 4).cast("string"),
        F.coalesce(F.col("passenger_count").cast("string"), F.lit("")),
        F.round(F.col("fare_amount"), 2).cast("string"),
        F.round(F.col("total_amount"), 2).cast("string")
    )
    df_sig = df.withColumn("trip_sig", sig)

    # Rank within each signature → keep only rows after the 1st occurrence
    w = W.partitionBy("trip_sig").orderBy(F.lit(1))
    dup_rows = (
        df_sig.withColumn("rn", F.row_number().over(w))
              .filter(F.col("rn") > 1)
              .drop("rn", "trip_sig")
    )

    dup_count = dup_rows.count()
    print(f"{name}: {dup_count:,} duplicate rows to drop")

    # Show the duplicate rows
    return dup_rows

# Run for Green & Yellow 
green_dup_rows  = show_duplicate_rows(green_valid,  "lpep_pickup_datetime", "lpep_dropoff_datetime",  "Green Taxi")
yellow_dup_rows = show_duplicate_rows(yellow_valid, "tpep_pickup_datetime", "tpep_dropoff_datetime", "Yellow Taxi")

# COMMAND ----------

# Drop duplicates already identified 
green_cleaned  = green_valid.subtract(green_dup_rows)
yellow_cleaned = yellow_valid.subtract(yellow_dup_rows)

# COMMAND ----------

# MAGIC %md
# MAGIC 3. Count the total number of rows for both green and yellow taxis.

# COMMAND ----------

# MAGIC %md
# MAGIC 3.1. The green taxi

# COMMAND ----------

print(f"Cleaned Green taxi: {green_cleaned.count()} rows ({green_cleaned.count()/green_duration.count():%}))")

# COMMAND ----------

# MAGIC %md
# MAGIC 3.2. The yellow taxi

# COMMAND ----------

yellow_cleaned.createOrReplaceTempView("yellow_cleaned")

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) as Cleaned_yellow_taxi from yellow_cleaned;

# COMMAND ----------

# MAGIC %md
# MAGIC 4. Combine the yellow and green taxi dataset together

# COMMAND ----------

# Step 1: Get all columns
green_cols = set(green_cleaned.columns)
yellow_cols = set(yellow_cleaned.columns)
all_cols = list(green_cols.union(yellow_cols))

# Step 2: Add missing columns
def align_columns(df, all_cols):
    for col in all_cols:
        if col not in df.columns:
            df = df.withColumn(col, F.lit(None))
    return df.select(all_cols)  # reorder

green_aligned = align_columns(green_cleaned, all_cols).withColumn("taxi_color", F.lit("green"))
yellow_aligned = align_columns(yellow_cleaned, all_cols).withColumn("taxi_color", F.lit("yellow"))

# Step 3: Union both
combined_taxi = green_aligned.unionByName(yellow_aligned)

# COMMAND ----------

# MAGIC %md
# MAGIC 5. Combine the new dataframe/table with the location data

# COMMAND ----------

from pyspark.sql import functions as F

# Use the zones table
ZONES_TBL = "workspace.default.taxi_zone_lookup"
zones_raw = spark.table(ZONES_TBL)

# Standardise likely column names to the TLC schema:
# Expecting: LocationID, Borough, Zone, service_zone
zones_df = (
    zones_raw
    .withColumnRenamed("locationid", "LocationID")
    .withColumnRenamed("location_id", "LocationID")
    .withColumnRenamed("borough", "Borough")
    .withColumnRenamed("zone", "Zone")
    .withColumnRenamed("servicezone", "service_zone")
    .withColumnRenamed("service_zone", "service_zone")
)

# If LocationID is string, cast to int
if dict(zones_df.dtypes).get("LocationID") != "int":
    zones_df = zones_df.withColumn("LocationID", F.col("LocationID").cast("int"))

# Resolve trip ID columns 
def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of {candidates} found in {df}")

PU_COL = first_existing(combined_taxi, ["PULocationID","pu_location_id","pickup_location_id","pickup_zone_id"])
DO_COL = first_existing(combined_taxi, ["DOLocationID","do_location_id","dropoff_location_id","dropoff_zone_id"])

# Build pickup/dropoff lookup views 
zones_pu = zones_df.select(
    F.col("LocationID").alias("pu_location_id"),
    F.col("Borough").alias("pu_borough"),
    F.col("Zone").alias("pu_zone"),
    F.col("service_zone").alias("pu_service_zone"),
)
zones_do = zones_df.select(
    F.col("LocationID").alias("do_location_id"),
    F.col("Borough").alias("do_borough"),
    F.col("Zone").alias("do_zone"),
    F.col("service_zone").alias("do_service_zone"),
)

# Join twice 
trips_with_loc = (
    combined_taxi
    .join(zones_pu, F.col(PU_COL) == F.col("pu_location_id"), "left")
    .join(zones_do, F.col(DO_COL) == F.col("do_location_id"), "left")
)

# Nice column order
front_cols = [c for c in ["trip_id","taxi_color","pickup_datetime","dropoff_datetime"] if c in trips_with_loc.columns]
pu_cols    = ["pu_location_id","pu_borough","pu_zone","pu_service_zone"]
do_cols    = ["do_location_id","do_borough","do_zone","do_service_zone"]
other_cols = [c for c in trips_with_loc.columns if c not in set(front_cols+pu_cols+do_cols)]
trips_with_loc = trips_with_loc.select(front_cols + pu_cols + do_cols + other_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC 6. Save the final cleaned and joined dataset as a table

# COMMAND ----------

# Define target table (catalog.schema.table_name)
CATALOG = "workspace"  
SCHEMA  = "default"         
TABLE   = "final_trips"

GOLD = f"{CATALOG}.{SCHEMA}.{TABLE}"

# Save as Delta table (overwrite old if exists)
(
    trips_with_loc
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(GOLD)
)

# COMMAND ----------

# MAGIC %md
# MAGIC 7. Count the total number of rows of the final table

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) AS total_rows from final_trips;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from final_trips;