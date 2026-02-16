# Import SparkSession, which is the entry point for creating DataFrames and running Spark jobs
from pyspark.sql import SparkSession

# Import Spark SQL aggregation function `sum`
# Aliased as spark_sum to avoid conflict with Python's built-in sum()
from pyspark.sql.functions import sum as spark_sum

# Import time module to measure job execution duration
import time


# ============================================================
# SPARK SESSION CONFIGURATION
# ============================================================

# Create a SparkSession
# - appName: identifies this job in Spark UI
# - spark.sql.adaptive.enabled = false:
#   Disables Adaptive Query Execution so Spark does NOT
#   automatically change partition counts at runtime.
#   This is critical for controlled partition experiments.
spark = (
    SparkSession.builder
    .appName("SparkJoinOptimization_Part2_Partitions")
    .config("spark.sql.adaptive.enabled", "false")
    .getOrCreate()
)

# Reduce Spark logging to WARN to avoid excessive console output
spark.sparkContext.setLogLevel("WARN")


# ============================================================
# FILE PATH DEFINITIONS
# ============================================================

# Path to the transactions (fact) dataset
TRANSACTIONS_PATH = "data/transactions_data.csv"

# Path to the users (dimension) dataset
USERS_PATH = "data/users_data.csv"

# Base output directory for partition experiments
OUTPUT_BASE_PATH = "output/local/part2_partitions"

# Print a header to identify this experiment in logs
print("=== Part 2: Impact of DataFrame Partitions ===")


# ============================================================
# LOAD DATA FROM CSV FILES
# ============================================================

# Read the transactions CSV file into a Spark DataFrame
# - header=True tells Spark the first row contains column names
# - inferSchema=True allows Spark to infer data types automatically
transactions = spark.read.csv(
    TRANSACTIONS_PATH,
    header=True,
    inferSchema=True
)

# Read the users CSV file into a Spark DataFrame
users = spark.read.csv(
    USERS_PATH,
    header=True,
    inferSchema=True
)


# ============================================================
# DATA CLEANING (SAME LOGIC AS PART 1)
# ============================================================

# Clean the transactions DataFrame
transactions_clean = (
    transactions
    # selectExpr allows SQL-style expressions
    .selectExpr(
        # Select client_id column as-is
        "client_id",

        # Remove currency symbols ($ and commas) from amount,
        # then safely cast to double using try_cast
        # try_cast prevents job failure if parsing fails
        "try_cast(regexp_replace(amount, '[$,]', '') as double) as amount"
    )
    # Filter out records with null client_id or amount
    .where("client_id IS NOT NULL AND amount IS NOT NULL")
)

# Persist (cache) the cleaned transactions DataFrame in memory
# This prevents Spark from re-reading and re-processing the CSV
# for each experiment iteration
transactions_clean = transactions_clean.persist()

# Trigger an action to force Spark to materialize the persisted DataFrame
transactions_clean.count()

# Clean the users DataFrame by removing null IDs
users_clean = users.where("id IS NOT NULL")

# Print schema to verify correct data types
print("Cleaned schema:")
transactions_clean.printSchema()


# ============================================================
# PARTITION COUNTS TO TEST
# ============================================================

# List of partition counts to experiment with
# These values represent under-partitioned to over-partitioned scenarios
partition_values = [8, 32, 64, 128, 256]

# List to store (partition_count, execution_time) results
results = []


# ============================================================
# RUN PARTITIONING EXPERIMENTS
# ============================================================

# Loop through each partition count
for partitions in partition_values:

    # Print separator for readability
    print("\n" + "=" * 60)
    print(f"Running with {partitions} partitions")
    print("=" * 60)

    # Explicitly repartition the transactions DataFrame
    # repartition(n):
    # - Forces a full shuffle
    # - Produces exactly `n` partitions
    # - Directly controls the number of shuffle tasks
    transactions_repart = transactions_clean.repartition(partitions)

    # Record start time before the join operation
    start_time = time.time()

    # Perform an inner join between transactions and users
    # This results in a shuffle-based Sort-Merge Join
    # The number of shuffle tasks is controlled by repartition()
    joined = (
        transactions_repart
        .join(
            users_clean,
            transactions_repart.client_id == users_clean.id,
            "inner"
        )
    )

    # Group the joined data by gender
    # Aggregate total transaction amount per gender
    result = (
        joined
        .groupBy("gender")
        .agg(
            spark_sum("amount").alias("total_transaction_amount")
        )
    )

    # Define output path specific to the partition count
    output_path = f"{OUTPUT_BASE_PATH}/partitions_{partitions}"

    # Write the result to Parquet format
    # This is an action and triggers execution of the full Spark DAG
    result.write.mode("overwrite").parquet(output_path)

    # Record end time after Spark job completes
    end_time = time.time()

    # Calculate execution duration in seconds
    duration = round(end_time - start_time, 2)

    # Store partition count and execution time
    results.append((partitions, duration))

    # Print execution time for this run
    print(f"Execution time ({partitions} partitions): {duration} seconds")


# ============================================================
# SUMMARY TABLE FOR REPORTING
# ============================================================

# Print summary header
print("\n" + "=" * 60)
print("Partition Impact Summary (Local Execution)")
print("=" * 60)

# Print table header
print(f"{'Partitions':<15}{'Execution Time (s)':<20}")
print("-" * 35)

# Print results for each partition count
for p, t in results:
    print(f"{p:<15}{t:<20}")


# ============================================================
# STOP SPARK SESSION
# ============================================================

# Stop Spark and release cluster resources
spark.stop()