# Databricks notebook source
# MAGIC %sql
# MAGIC DESCRIBE TABLE final_trips;

# COMMAND ----------

# MAGIC %md
# MAGIC 1. For each year and month:
# MAGIC - a. What was the total number of trips?
# MAGIC - b. Which day of week (e.g. monday, tuesday, etc..) had the most trips?
# MAGIC - c. Which hour of the day had the most trips?
# MAGIC - d. What was the average number of passengers?
# MAGIC - e. What was the average amount paid per trip (using total_amount)?
# MAGIC - f. What was the average amount paid per passenger (using total_amount)?

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH base AS (
# MAGIC   SELECT
# MAGIC     DATE_FORMAT(COALESCE(tpep_pickup_datetime, lpep_pickup_datetime), 'yyyy-MM') AS year_month,
# MAGIC     COALESCE(tpep_pickup_datetime, lpep_pickup_datetime) AS pickup_ts,
# MAGIC     passenger_count,
# MAGIC     total_amount
# MAGIC   FROM final_trips
# MAGIC ),
# MAGIC
# MAGIC -- a, d, e, f (monthly aggregates)
# MAGIC monthly AS (
# MAGIC   SELECT
# MAGIC     year_month,
# MAGIC     COUNT(*)                                        AS total_trips,               -- (a)
# MAGIC     ROUND(AVG(passenger_count), 2)                  AS avg_passengers,            -- (d)
# MAGIC     ROUND(AVG(total_amount), 2)                     AS avg_amount_per_trip,       -- (e)
# MAGIC     ROUND(AVG(CASE WHEN passenger_count > 0
# MAGIC                    THEN total_amount / passenger_count END), 2)
# MAGIC                                                     AS avg_amount_per_passenger   -- (f)
# MAGIC   FROM base
# MAGIC   GROUP BY year_month
# MAGIC ),
# MAGIC
# MAGIC -- b) weekday with most trips
# MAGIC dow_counts AS (
# MAGIC   SELECT
# MAGIC     year_month,
# MAGIC     DATE_FORMAT(pickup_ts, 'EEEE') AS day_of_week,
# MAGIC     COUNT(*) AS trips
# MAGIC   FROM base
# MAGIC   GROUP BY year_month, day_of_week
# MAGIC ),
# MAGIC dow_top AS (
# MAGIC   SELECT
# MAGIC     year_month, day_of_week, trips,
# MAGIC     ROW_NUMBER() OVER (PARTITION BY year_month ORDER BY trips DESC) AS rn
# MAGIC   FROM dow_counts
# MAGIC ),
# MAGIC
# MAGIC -- c) hour with most trips
# MAGIC hour_counts AS (
# MAGIC   SELECT
# MAGIC     year_month,
# MAGIC     HOUR(pickup_ts) AS hour_of_day,
# MAGIC     COUNT(*) AS trips
# MAGIC   FROM base
# MAGIC   GROUP BY year_month, hour_of_day
# MAGIC ),
# MAGIC hour_top AS (
# MAGIC   SELECT
# MAGIC     year_month, hour_of_day, trips,
# MAGIC     ROW_NUMBER() OVER (PARTITION BY year_month ORDER BY trips DESC) AS rn
# MAGIC   FROM hour_counts
# MAGIC )
# MAGIC
# MAGIC SELECT
# MAGIC   m.year_month,
# MAGIC   m.total_trips,                                      -- (a)
# MAGIC   dt.day_of_week       AS most_trips_day,             -- (b)
# MAGIC   ht.hour_of_day       AS most_trips_hour,            -- (c)
# MAGIC   m.avg_passengers,                                   -- (d)
# MAGIC   m.avg_amount_per_trip,                              -- (e)
# MAGIC   m.avg_amount_per_passenger                          -- (f)
# MAGIC FROM monthly m
# MAGIC LEFT JOIN dow_top  dt ON m.year_month = dt.year_month AND dt.rn = 1
# MAGIC LEFT JOIN hour_top ht ON m.year_month = ht.year_month AND ht.rn = 1
# MAGIC ORDER BY m.year_month;

# COMMAND ----------

# MAGIC %md
# MAGIC 2. For each taxi color (yellow and green)

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH base AS (
# MAGIC   SELECT
# MAGIC     taxi_color,
# MAGIC     COALESCE(tpep_pickup_datetime,  lpep_pickup_datetime)  AS pu_ts,
# MAGIC     COALESCE(tpep_dropoff_datetime, lpep_dropoff_datetime) AS do_ts,
# MAGIC     trip_distance * 1.60934 AS distance_km                  -- miles -> km
# MAGIC   FROM final_trips
# MAGIC ),
# MAGIC calc AS (
# MAGIC   SELECT
# MAGIC     taxi_color,
# MAGIC     (CAST(do_ts AS BIGINT) - CAST(pu_ts AS BIGINT))              AS dur_sec,
# MAGIC    (CAST(do_ts AS BIGINT) - CAST(pu_ts AS BIGINT)) / 60.0        AS duration_min,
# MAGIC     distance_km,
# MAGIC     CASE
# MAGIC       WHEN (CAST(do_ts AS BIGINT) - CAST(pu_ts AS BIGINT)) > 0 AND distance_km >= 0
# MAGIC         THEN distance_km / ((CAST(do_ts AS BIGINT) - CAST(pu_ts AS BIGINT)) / 3600.0)
# MAGIC     END AS speed_kmh
# MAGIC   FROM base
# MAGIC )
# MAGIC SELECT
# MAGIC   taxi_color,
# MAGIC
# MAGIC   -- (a) trip duration in minutes
# MAGIC   ROUND(AVG(duration_min), 2)                              AS avg_duration_min,
# MAGIC   ROUND(percentile_approx(duration_min, 0.5), 2)           AS median_duration_min,
# MAGIC   ROUND(MIN(duration_min), 2)                              AS min_duration_min,
# MAGIC   ROUND(MAX(duration_min), 2)                              AS max_duration_min,
# MAGIC
# MAGIC   -- (b) trip distance in km
# MAGIC   ROUND(AVG(distance_km), 2)                               AS avg_distance_km,
# MAGIC   ROUND(percentile_approx(distance_km, 0.5), 2)            AS median_distance_km,
# MAGIC   ROUND(MIN(distance_km), 2)                               AS min_distance_km,
# MAGIC   ROUND(MAX(distance_km), 2)                               AS max_distance_km,
# MAGIC
# MAGIC   -- (c) speed in km/h
# MAGIC   ROUND(AVG(speed_kmh), 2)                                 AS avg_speed_kmh,
# MAGIC   ROUND(percentile_approx(speed_kmh, 0.5), 2)              AS median_speed_kmh,
# MAGIC   ROUND(MIN(speed_kmh), 2)                                 AS min_speed_kmh,
# MAGIC   ROUND(MAX(speed_kmh), 2)                                 AS max_speed_kmh
# MAGIC
# MAGIC FROM calc
# MAGIC GROUP BY taxi_color
# MAGIC ORDER BY taxi_color;

# COMMAND ----------

# MAGIC %md
# MAGIC 3. For each taxi colour (yellow and green), each pair of pick up and drop off locations (use boroughs not the id), each month, each day of week and each hours:
# MAGIC - a. The total number of trips
# MAGIC - b. The average distance
# MAGIC - c. The average amount paid per trip (using total_amount)
# MAGIC - d. The total amount paid (using total_amount)

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH base AS (
# MAGIC   SELECT
# MAGIC     taxi_color,
# MAGIC     pu_borough,
# MAGIC     do_borough,
# MAGIC     COALESCE(tpep_pickup_datetime,  lpep_pickup_datetime)  AS pickup_ts,
# MAGIC     trip_distance * 1.60934 AS distance_km,   -- miles → km
# MAGIC     total_amount
# MAGIC   FROM final_trips
# MAGIC ),
# MAGIC dim AS (
# MAGIC   SELECT
# MAGIC     taxi_color,
# MAGIC     pu_borough,
# MAGIC     do_borough,
# MAGIC     DATE_FORMAT(pickup_ts, 'yyyy-MM') AS year_month,
# MAGIC     DATE_FORMAT(pickup_ts, 'EEEE')    AS day_of_week,
# MAGIC     HOUR(pickup_ts)                   AS hour_of_day,
# MAGIC     distance_km,
# MAGIC     total_amount
# MAGIC   FROM base
# MAGIC )
# MAGIC SELECT
# MAGIC   taxi_color,
# MAGIC   pu_borough,
# MAGIC   do_borough,
# MAGIC   year_month,
# MAGIC   day_of_week,
# MAGIC   hour_of_day,
# MAGIC
# MAGIC   COUNT(*)                           AS total_trips,          -- (a)
# MAGIC   ROUND(AVG(distance_km), 2)         AS avg_distance_km,      -- (b)
# MAGIC   ROUND(AVG(total_amount), 2)        AS avg_amount_per_trip,  -- (c)
# MAGIC   ROUND(SUM(total_amount), 2)        AS total_amount_paid     -- (d)
# MAGIC
# MAGIC FROM dim
# MAGIC GROUP BY
# MAGIC   taxi_color, pu_borough, do_borough, year_month, day_of_week, hour_of_day
# MAGIC ORDER BY
# MAGIC   taxi_color, pu_borough, do_borough, year_month, day_of_week, hour_of_day;

# COMMAND ----------

# MAGIC %md
# MAGIC 4. For 2024, compute the share of total revenue contributed by the Top 10 pickup→dropoff borough pairs (ranked by total_amount)

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH filtered AS (
# MAGIC   SELECT
# MAGIC     COALESCE(pu_borough, 'Unknown') AS pu_borough,
# MAGIC     COALESCE(do_borough, 'Unknown') AS do_borough,
# MAGIC     total_amount
# MAGIC   FROM final_trips
# MAGIC   WHERE YEAR(COALESCE(tpep_pickup_datetime, lpep_pickup_datetime)) = 2024
# MAGIC ),
# MAGIC pair_rev AS (
# MAGIC   SELECT
# MAGIC     pu_borough,
# MAGIC     do_borough,
# MAGIC     SUM(total_amount) AS revenue
# MAGIC   FROM filtered
# MAGIC   GROUP BY pu_borough, do_borough
# MAGIC ),
# MAGIC ranked AS (
# MAGIC   SELECT
# MAGIC     pu_borough,
# MAGIC     do_borough,
# MAGIC     revenue,
# MAGIC     SUM(revenue) OVER () AS total_revenue,
# MAGIC     DENSE_RANK() OVER (ORDER BY revenue DESC) AS rnk
# MAGIC   FROM pair_rev
# MAGIC )
# MAGIC SELECT
# MAGIC   pu_borough,
# MAGIC   do_borough,
# MAGIC   ROUND(revenue, 2) AS revenue_2024,
# MAGIC   ROUND(100.0 * revenue / total_revenue, 2) AS share_pct,
# MAGIC   ROUND(100.0 *
# MAGIC         (SUM(revenue) OVER (ORDER BY revenue DESC
# MAGIC                             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)) / total_revenue, 2)
# MAGIC         AS cumulative_share_pct
# MAGIC FROM ranked
# MAGIC WHERE rnk <= 10
# MAGIC ORDER BY revenue_2024 DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC 5. The percentage of trips where drivers received tips

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   ROUND(100.0 * SUM(CASE WHEN tip_amount > 0 THEN 1 ELSE 0 END) / COUNT(*), 2)
# MAGIC     AS pct_trips_with_tips
# MAGIC FROM final_trips;

# COMMAND ----------

# MAGIC %md
# MAGIC 6. For trips where the driver received tips, what was the percentage where the driver received tips of at least $15

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   ROUND(
# MAGIC     100.0 * AVG(CASE
# MAGIC                   WHEN tip_amount > 0 THEN        -- restrict to tipped trips
# MAGIC                     CASE WHEN tip_amount >= 15 THEN 1.0 ELSE 0.0 END
# MAGIC                 END),
# MAGIC     2
# MAGIC   ) AS pct_tips_ge_15_among_tipped
# MAGIC FROM final_trips;

# COMMAND ----------

# MAGIC %md
# MAGIC 7. Classify each trip into bins of durations:

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH base AS (
# MAGIC   SELECT
# MAGIC     -- unify timestamps
# MAGIC     COALESCE(tpep_pickup_datetime,  lpep_pickup_datetime)  AS pu_ts,
# MAGIC     COALESCE(tpep_dropoff_datetime, lpep_dropoff_datetime) AS do_ts,
# MAGIC     trip_distance * 1.60934 AS distance_km,
# MAGIC     total_amount
# MAGIC   FROM final_trips
# MAGIC ),
# MAGIC calc AS (
# MAGIC   SELECT
# MAGIC     distance_km,
# MAGIC     total_amount,
# MAGIC     -- duration in seconds and minutes
# MAGIC     (CAST(do_ts AS BIGINT) - CAST(pu_ts AS BIGINT)) AS dur_sec,
# MAGIC     (CAST(do_ts AS BIGINT) - CAST(pu_ts AS BIGINT)) / 60.0 AS duration_min
# MAGIC   FROM base
# MAGIC ),
# MAGIC binned AS (
# MAGIC   SELECT
# MAGIC     CASE
# MAGIC       WHEN duration_min < 5  THEN 'Under 5 mins'
# MAGIC       WHEN duration_min < 10 THEN '5–10 mins'
# MAGIC       WHEN duration_min < 20 THEN '10–20 mins'
# MAGIC       WHEN duration_min < 30 THEN '20–30 mins'
# MAGIC       WHEN duration_min < 60 THEN '30–60 mins'
# MAGIC       ELSE '≥ 60 mins'
# MAGIC     END AS duration_bin,
# MAGIC     -- speed only if duration > 0 and distance > 0
# MAGIC     CASE
# MAGIC       WHEN dur_sec > 0 AND distance_km > 0
# MAGIC         THEN distance_km / (dur_sec / 3600.0)  -- km / (hours)
# MAGIC     END AS speed_kmh,
# MAGIC     -- km per $ only if charge > 0
# MAGIC     CASE
# MAGIC       WHEN total_amount > 0 THEN distance_km / total_amount
# MAGIC     END AS km_per_dollar
# MAGIC   FROM calc
# MAGIC )
# MAGIC SELECT
# MAGIC   duration_bin,
# MAGIC   COUNT(*) AS trips,
# MAGIC   ROUND(AVG(speed_kmh), 2)      AS avg_speed_kmh,
# MAGIC   ROUND(AVG(km_per_dollar), 3)  AS avg_km_per_dollar
# MAGIC FROM binned
# MAGIC GROUP BY duration_bin
# MAGIC ORDER BY
# MAGIC   CASE duration_bin
# MAGIC     WHEN 'Under 5 mins' THEN 1
# MAGIC     WHEN '5–10 mins'    THEN 2
# MAGIC     WHEN '10–20 mins'   THEN 3
# MAGIC     WHEN '20–30 mins'   THEN 4
# MAGIC     WHEN '30–60 mins'   THEN 5
# MAGIC     WHEN '≥ 60 mins'    THEN 6
# MAGIC   END;

# COMMAND ----------

# MAGIC %md
# MAGIC 8. Which duration bin will you advise a taxi driver to target to maximise his income?
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC --Use $/hour.
# MAGIC WITH base AS (
# MAGIC   SELECT
# MAGIC     COALESCE(tpep_pickup_datetime,  lpep_pickup_datetime)  AS pu_ts,
# MAGIC     COALESCE(tpep_dropoff_datetime, lpep_dropoff_datetime) AS do_ts,
# MAGIC     total_amount,
# MAGIC     trip_distance
# MAGIC   FROM final_trips
# MAGIC ),
# MAGIC calc AS (
# MAGIC   SELECT
# MAGIC     -- duration
# MAGIC     (CAST(do_ts AS BIGINT) - CAST(pu_ts AS BIGINT)) AS dur_sec,
# MAGIC     (CAST(do_ts AS BIGINT) - CAST(pu_ts AS BIGINT)) / 60.0 AS dur_min,
# MAGIC     total_amount,
# MAGIC     trip_distance
# MAGIC   FROM base
# MAGIC ),
# MAGIC binned AS (
# MAGIC   SELECT
# MAGIC     CASE
# MAGIC       WHEN dur_min < 5  THEN 'Under 5 mins'
# MAGIC       WHEN dur_min < 10 THEN '5–10 mins'
# MAGIC       WHEN dur_min < 20 THEN '10–20 mins'
# MAGIC       WHEN dur_min < 30 THEN '20–30 mins'
# MAGIC       WHEN dur_min < 60 THEN '30–60 mins'
# MAGIC       ELSE '≥ 60 mins'
# MAGIC     END AS duration_bin,
# MAGIC     dur_sec,
# MAGIC     dur_min,
# MAGIC     total_amount
# MAGIC   FROM calc
# MAGIC   -- optional cleaning (helps avoid zeros/infinite rates)
# MAGIC   WHERE dur_sec > 0 AND total_amount > 0 AND trip_distance >= 0.1
# MAGIC )
# MAGIC SELECT
# MAGIC   duration_bin,
# MAGIC   COUNT(*) AS trips,
# MAGIC   ROUND(AVG(dur_min), 2) AS avg_duration_min,
# MAGIC   -- $ per hour for each trip = total_amount / (dur_sec/3600)
# MAGIC   ROUND(AVG(total_amount / (dur_sec / 3600.0)), 2)                    AS avg_dollars_per_hour,
# MAGIC   ROUND(percentile_approx(total_amount / (dur_sec / 3600.0), 0.5), 2) AS median_dollars_per_hour,
# MAGIC   -- add trip share to ensure the bin has enough volume to be actionable
# MAGIC   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS trip_share_pct
# MAGIC FROM binned
# MAGIC GROUP BY duration_bin
# MAGIC ORDER BY
# MAGIC   CASE duration_bin
# MAGIC     WHEN 'Under 5 mins' THEN 1
# MAGIC     WHEN '5–10 mins'    THEN 2
# MAGIC     WHEN '10–20 mins'   THEN 3
# MAGIC     WHEN '20–30 mins'   THEN 4
# MAGIC     WHEN '30–60 mins'   THEN 5
# MAGIC     WHEN '≥ 60 mins'    THEN 6
# MAGIC   END;