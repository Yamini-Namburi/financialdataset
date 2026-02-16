# Import SparkSession, the entry point to any Spark application
from pyspark.sql import SparkSession

# Import Spark SQL aggregation function `sum`
# Aliased as spark_sum to avoid conflict with Python's built-in sum()
# Import broadcast to explicitly force a Broadcast Hash Join
from pyspark.sql.functions import sum as spark_sum, broadcast

# Import time module to measure execution time
import time


# ============================================================
# SPARK SESSION
# ============================================================

# Create a SparkSession
# - appName identifies this job in Spark UI
# - spark.sql.adaptive.enabled = false disables AQE so Spark
#   does NOT automatically switch join strategies at runtime
#   This ensures deterministic, comparable join behaviour
spark = (
    SparkSession.builder
    .appName("SparkJoinOptimization_Part3_JoinStrategies")
    .config("spark.sql.adaptive.enabled", "false")
    .getOrCreate()
)

# Reduce log verbosity so output is readable
spark.sparkContext.setLogLevel("WARN")


# ============================================================
# PATHS
# ============================================================

# Input path for transactions (fact table)
TRANSACTIONS_PATH = "data/transactions_data.csv"

# Input path for users (dimension table)
USERS_PATH = "data/users_data.csv"

# Output path for default (Spark-chosen) join
OUTPUT_DEFAULT = "output/local/part3_default_join"

# Output path for broadcast join
OUTPUT_BROADCAST = "output/local/part3_broadcast_join"

# Print experiment header
print("=== Part 3: Exploring Spark Join Strategies ===")


# ============================================================
# LOAD DATA
# ============================================================

# Load transactions CSV into a Spark DataFrame
# - header=True: first row contains column names
# - inferSchema=True: Spark infers column data types
transactions = spark.read.csv(
    TRANSACTIONS_PATH,
    header=True,
    inferSchema=True
)

# Load users CSV into a Spark DataFrame
users = spark.read.csv(
    USERS_PATH,
    header=True,
    inferSchema=True
)


# ============================================================
# CLEAN DATA (same logic as Part 1)
# ============================================================

# Clean transactions data
transactions_clean = (
    transactions
    # selectExpr allows SQL-style expressions
    .selectExpr(
        # Keep client_id unchanged
        "client_id",

        # Remove currency symbols ($ and commas) from amount
        # Safely cast to double using try_cast to avoid job failure
        "try_cast(regexp_replace(amount, '[$,]', '') as double) as amount"
    )
    # Remove invalid records
    .where("client_id IS NOT NULL AND amount IS NOT NULL")

    # Persist the cleaned DataFrame in memory
    # Prevents re-reading CSV and stabilizes execution plans
    .persist()
)

# Trigger an action to materialize the persisted DataFrame
transactions_clean.count()

# Clean users data by removing null IDs
users_clean = users.where("id IS NOT NULL")

# Print schema for verification
print("Cleaned schema:")
transactions_clean.printSchema()


# ============================================================
# 1️⃣ DEFAULT JOIN (Spark chooses strategy)
# ============================================================

print("\n=== Default Join Strategy (Spark Decides) ===")

# Record start time for default join
start_default = time.time()

# Perform an inner join without hints
# Spark will decide the join strategy:
# - Sort-Merge Join (most likely)
# - Shuffle Hash Join (if conditions allow)
joined_default = (
    transactions_clean
    .join(
        users_clean,
        transactions_clean.client_id == users_clean.id,
        "inner"
    )
)

# Print detailed physical and logical execution plan
# This reveals which join strategy Spark selected
print("\nExplain plan (Default Join):")
joined_default.explain(True)

# Aggregate results by gender
result_default = (
    joined_default
    .groupBy("gender")
    .agg(
        # Compute total transaction amount per gender
        spark_sum("amount").alias("total_transaction_amount")
    )
)

# Write results to Parquet
# Writing is an action and triggers the full Spark DAG
result_default.write.mode("overwrite").parquet(OUTPUT_DEFAULT)

# Record end time for default join
end_default = time.time()

# Print execution time for default join
print(f"Default join execution time: {end_default - start_default:.2f} seconds")


# ============================================================
# 2️⃣ BROADCAST HASH JOIN (Forced)
# ============================================================

print("\n=== Broadcast Hash Join (Forced) ===")

# Record start time for broadcast join
start_broadcast = time.time()

# Perform join with explicit broadcast hint
# broadcast(users_clean):
# - Sends the entire users DataFrame to every executor
# - Eliminates shuffle on the large transactions dataset
# - Forces a Broadcast Hash Join
joined_broadcast = (
    transactions_clean
    .join(
        broadcast(users_clean),
        transactions_clean.client_id == users_clean.id,
        "inner"
    )
)

# Print execution plan to confirm BroadcastHashJoin
print("\nExplain plan (Broadcast Join):")
joined_broadcast.explain(True)

# Aggregate results by gender
result_broadcast = (
    joined_broadcast
    .groupBy("gender")
    .agg(
        spark_sum("amount").alias("total_transaction_amount")
    )
)

# Write broadcast join results to Parquet
result_broadcast.write.mode("overwrite").parquet(OUTPUT_BROADCAST)

# Record end time for broadcast join
end_broadcast = time.time()

# Print execution time for broadcast join
print(f"Broadcast join execution time: {end_broadcast - start_broadcast:.2f} seconds")


# ============================================================
# SUMMARY
# ============================================================

# Print comparison summary of join strategies
print("\n=== Join Strategy Comparison Summary ===")
print(f"Default Join Time   : {end_default - start_default:.2f} seconds")
print(f"Broadcast Join Time : {end_broadcast - start_broadcast:.2f} seconds")


# ============================================================
# STOP SPARK
# ============================================================

# Stop Spark session and release resources
spark.stop()