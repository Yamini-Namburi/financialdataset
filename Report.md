Part 1: Naive Join (Baseline)
Objective

The objective of this experiment is to establish a baseline Spark job performance by performing a naive join between large transaction data and user demographic data without applying any explicit performance optimizations. This baseline serves as a reference point for evaluating the impact of later optimizations such as repartitioning, broadcast joins, and Adaptive Query Execution (AQE).

Dataset & Join Strategy

Input datasets:

transactions_data.csv (large fact-like dataset)

users_data.csv (smaller dimension-like dataset)

Join key: transactions.client_id = users.id

Join type: Inner Join

Optimizations applied: None

No repartitioning

No broadcast hints

AQE disabled

Execution trigger: Aggregation followed by Parquet write

Rows with null join keys were dropped prior to joining to avoid join inconsistencies.

Code Snippet (Naive Join)
joined = transactions_clean.join(
    users,
    transactions_clean.client_id == users.id,
    "inner"
)

result = (
    joined
    .groupBy("gender")
    .agg(sum("amount").alias("total_transaction_amount"))
)

result.write.mode("overwrite").parquet("output/local/part1_naive_join")


This aggregation forces Spark to materialize the join, as Spark transformations are lazily evaluated.

Execution Time Measurement

Execution time was measured programmatically using Python’s time.time() function, capturing the duration from the start of the Spark action to the completion of the Parquet write.

Execution Time Results
Environment	Execution Time (seconds)
Local	5.52

This timing includes join execution, shuffle, aggregation, and disk write.

Spark UI Observations
Jobs Tab

Multiple short-lived jobs were observed for CSV scanning and metadata operations.

One dominant job (~5 seconds) corresponded to the join, shuffle, aggregation, and write operations.

This aligns with the measured execution time from Python.

(Insert Jobs tab screenshot here)

Stages Tab

The Stages tab showed four completed stages, with one stage dominating overall execution time:

Stage 1

Duration: ~7 seconds

Tasks: 10 / 10

Input size: ~1.2 GB

Represents the join, shuffle, aggregation, and write operations

The remaining stages completed in milliseconds and corresponded to CSV scanning and schema inference.

The presence of multiple stages confirms that a shuffle occurred, which is expected when Spark performs a naive Sort-Merge Join.

(Insert Stages tab screenshot here)

DAG Visualization

The DAG visualization showed clear shuffle boundaries between stages, indicating data redistribution across tasks during the join and aggregation phase.

(Insert DAG screenshot here)

Executors Tab

The Executors tab appeared empty. This is expected because the application was executed in local mode, where the driver process also acts as the executor. No separate executor JVMs are created in this execution mode.

(Insert Executors tab screenshot here)

Analysis & Discussion

In the absence of optimization hints, Spark selected a Sort-Merge Join, which is the default strategy for large datasets when no side is broadcastable. This join strategy introduces a full shuffle of data across partitions, which becomes the dominant cost of execution.

The Stages tab confirms that the majority of runtime is spent in the shuffle-heavy stage, while CSV scanning contributes minimally to total execution time. Task parallelism was limited by local execution constraints, but Spark still distributed work across multiple tasks.

This experiment also highlighted practical challenges of working with CSV data, including schema inference issues and malformed numeric values, which required explicit data cleansing before aggregation.

Key Learnings (Part 1)

Spark joins are lazily evaluated and require an action to trigger execution

Naive joins introduce full shuffles, which significantly impact performance

CSV inputs are fragile and require careful data type handling

The longest stage in the Spark UI typically represents the true performance bottleneck

Execution time measured programmatically aligns with Spark UI job durations

Overall Summary & Reflection

This baseline experiment demonstrates how Spark behaves under default execution settings when performing large-scale data joins. Without optimization, Spark relies on shuffle-heavy Sort-Merge Joins, which can significantly increase latency and resource consumption.

In real-world fraud detection systems, transaction enrichment pipelines often require joining large transaction streams with user and card metadata. Naive join strategies like this would not scale efficiently in production environments. This experiment highlights the importance of join optimization techniques—such as repartitioning, broadcast joins, and AQE—to achieve performant and cost-effective fraud analytics pipelines.