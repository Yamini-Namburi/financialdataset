# Import SparkSession – the entry point to any Spark application
from pyspark.sql import SparkSession

# Import Spark SQL aggregation function `sum`
# Renamed to spark_sum to avoid clashing with Python's built-in sum()
from pyspark.sql.functions import sum as spark_sum

# Import time module to measure execution time
import time


# =========================
# SPARK SESSION SETUP
# =========================

# Create a SparkSession
# - appName: Helps identify the job in Spark UI
# - spark.sql.adaptive.enabled = false:
#   Explicitly disables Adaptive Query Execution (AQE)
#   so this script represents a *naive baseline* for comparison
spark = (
    SparkSession.builder
    .appName("SparkJoinOptimization_Part1_Naive")
    .config("spark.sql.adaptive.enabled", "false")
    .getOrCreate()
)

# Reduce Spark log noise so output is readable
spark.sparkContext.setLogLevel("WARN")


# =========================
# FILE PATH DEFINITIONS
# =========================

# Path to transactions dataset (raw transactional data)
TRANSACTIONS_PATH = "data/transactions_data.csv"

# Path to users dataset (dimension table with user attributes)
USERS_PATH = "data/users_data.csv"

# Output path for results of naive join
OUTPUT_PATH = "output/local/part1_naive_join"

print("=== Part 1: Naive Join (Baseline) ===")


# =========================
# LOAD DATA
# =========================

# Read transactions CSV into a Spark DataFrame
# - header=True: First row contains column names
# - inferSchema=True: Spark infers column data types
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
# DATA CLEANING
# =========================

# Clean and prepare transaction data
# This transformation intentionally:
# - Breaks Catalyst optimizer lineage from the raw CSV
# - Forces Spark to materialize this step separately
transactions_clean = (
    transactions
    # selectExpr allows SQL-like expressions
    .selectExpr(
        # Keep client_id as-is
        "client_id",

        # Remove currency symbols ($, commas) from amount
        # Convert cleaned string to double using try_cast
        # try_cast prevents job failure on malformed values
        "try_cast(regexp_replace(amount, '[$,]', '') as double) as amount"
    )
    # Filter out invalid records
    .where("client_id IS NOT NULL AND amount IS NOT NULL")
)

# Persist the cleaned DataFrame in memory
# This:
# - Forces Spark to materialize the dataset
# - Prevents Catalyst from re-optimizing earlier stages
# - Makes this a true baseline for later optimizations
transactions_clean = transactions_clean.persist()

# Trigger an action to force evaluation of the persist()
transactions_clean.count()

# Print schema to verify cleaning and data types
print("Cleaned schema:")
transactions_clean.printSchema()


# =========================
# JOIN OPERATION
# =========================

# Capture start time for performance measurement
start_time = time.time()

# Perform an inner join between transactions and users
# Join condition:
#   transactions_clean.client_id == users.id
# Join type:
#   "inner" ensures only matching records are retained
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
# Calculate total transaction amount per gender
result = (
    joined
    .groupBy("gender")
    .agg(
        # Sum transaction amounts using Spark aggregation
        spark_sum("amount").alias("total_transaction_amount")
    )
)


# =========================
# WRITE OUTPUT
# =========================

# Write the aggregation result to Parquet format
# - overwrite mode ensures idempotent runs
# - Parquet is columnar and efficient for analytics
result.write.mode("overwrite").parquet(OUTPUT_PATH)

# Capture end time after job completion
end_time = time.time()
   
# Print total execution time
print(f"Naive execution time: {end_time - start_time:.2f} seconds")

# Display result in console (for validation)
result.show(truncate=False)


# =========================
# SHUTDOWN
# =========================

# Stop Spark session and release resources
spark.stop()