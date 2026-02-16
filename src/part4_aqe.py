# Import SparkSession, the entry point to Spark SQL and DataFrame APIs
from pyspark.sql import SparkSession

# Import Spark SQL aggregation function `sum`
# Aliased as spark_sum to avoid conflict with Python's built-in sum()
from pyspark.sql.functions import sum as spark_sum

# Import time module to measure execution duration
import time


# =========================
# SPARK SESSION (AQE ENABLED)
# =========================

# Create a SparkSession with Adaptive Query Execution (AQE) enabled
spark = (
    SparkSession.builder
    # Name of the Spark application (visible in Spark UI)
    .appName("SparkJoinOptimization_Part4_AQE")

    # Enable Adaptive Query Execution
    # AQE allows Spark to optimize the query *at runtime*
    .config("spark.sql.adaptive.enabled", "true")

    # Enable dynamic coalescing of shuffle partitions
    # Spark can merge small partitions into larger ones after shuffle
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")

    # Allow Spark to switch to Broadcast Join at runtime
    # If one side of the join is <= 10 MB (after runtime statistics)
    .config("spark.sql.adaptive.autoBroadcastJoinThreshold", "10MB")

    # Create or reuse SparkSession
    .getOrCreate()
)

# Reduce log verbosity for readability
spark.sparkContext.setLogLevel("WARN")


# =========================
# PATHS
# =========================

# Path to transactions dataset (fact table)
TRANSACTIONS_PATH = "data/transactions_data.csv"

# Path to users dataset (dimension table)
USERS_PATH = "data/users_data.csv"

# Output path for AQE results
OUTPUT_PATH = "output/local/part4_aqe"

# Print experiment header
print("=== Part 4: Adaptive Query Execution (AQE) ===")


# =========================
# LOAD DATA
# =========================

# Read transactions CSV into a Spark DataFrame
# - header=True: first row contains column names
# - inferSchema=True: Spark infers data types automatically
transactions = spark.read.csv(
    TRANSACTIONS_PATH,
    header=True,
    inferSchema=True
)

# Read users CSV into a Spark DataFrame
users = spark.read.csv(
    USERS_PATH,
    header=True,
    inferSchema=True
)


# =========================
# CLEAN DATA (same as Part 1)
# =========================

# Clean the transactions data
transactions_clean = (
    transactions
    # Use SQL-style expressions for transformation
    .selectExpr(
        # Select client_id as-is
        "client_id",

        # Remove currency symbols ($, commas) from amount
        # Safely cast to double using try_cast
        # try_cast avoids job failure due to malformed values
        "try_cast(regexp_replace(amount, '[$,]', '') as double) as amount"
    )
    # Filter out invalid rows
    .where("client_id IS NOT NULL AND amount IS NOT NULL")
)

# Persist cleaned DataFrame in memory
# This stabilizes input data and avoids repeated CSV reads
transactions_clean = transactions_clean.persist()

# Trigger an action to materialize the persisted DataFrame
transactions_clean.count()

# Print schema to verify correct data types
print("Cleaned schema:")
transactions_clean.printSchema()


# =========================
# JOIN (NO HINTS, NO REPARTITION)
# =========================

# Record start time before join execution
start_time = time.time()

# Perform an inner join between transactions and users
# - No broadcast hint
# - No repartition
# Spark initially plans a shuffle-based join
# AQE may change the join strategy at runtime
joined = (
    transactions_clean
    .join(
        users,
        transactions_clean.client_id == users.id,
        "inner"
    )
)


# =========================
# AGGREGATION
# =========================

# Group joined data by gender
# Aggregate total transaction amount per gender
result = (
    joined
    .groupBy("gender")
    .agg(
        # Distributed sum aggregation
        spark_sum("amount").alias("total_transaction_amount")
    )
)


# =========================
# EXPLAIN PLAN (IMPORTANT FOR REPORT)
# =========================

# Print the extended physical and logical plan
# This shows:
# - Initial plan (before AQE)
# - Final plan (after AQE optimizations)
# Look for AdaptiveSparkPlan, BroadcastHashJoin,
# and coalesced shuffle partitions
print("\n=== AQE Physical Plan ===")
joined.explain(extended=True)


# =========================
# WRITE OUTPUT
# =========================

# Write the final result to Parquet format
# Writing is an action that triggers full execution of the query
result.write.mode("overwrite").parquet(OUTPUT_PATH)

# Record end time after execution completes
end_time = time.time()

# Print total execution time
print(f"AQE execution time: {end_time - start_time:.2f} seconds")

# Display result in console for validation
result.show(truncate=False)


# =========================
# STOP SPARK SESSION
# =========================

# Stop Spark session and release resources
spark.stop()