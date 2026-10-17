"""PHASE 4b - Spark Structured Streaming job.

Kafka topic  ->  parse JSON  ->  3 parallel queries:
  A) raw events   : stored in HDFS (parquet) + a small recent copy for drill-down in the dashboard
  B) 10-second    : live metrics per service (fast charts)
  C) 1-minute     : metrics -> ML scoring -> anomaly detection -> incident correlation
Checkpoints live in HDFS, so the job can recover after a crash.
"""
import argparse
import time
import traceback

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

from src.common import config as C
from src.common.spark_utils import get_spark
from src.common.store import RollingTable
from src.streaming.processing import MinuteProcessor

SCHEMA = StructType([
    StructField("ts", StringType()), StructField("service", StringType()),
    StructField("level", StringType()), StructField("status_code", IntegerType()),
    StructField("latency_ms", DoubleType()), StructField("host", StringType()),
    StructField("endpoint", StringType()), StructField("message", StringType()),
    StructField("truth_k", DoubleType())])


def parse_events(source):
    """source has a 'value' column (JSON text) - exactly what the Kafka source delivers."""
    return (source.select(F.from_json(F.col("value").cast("string"), SCHEMA).alias("e")).select("e.*")
            .withColumn("event_time", F.to_timestamp("ts", "yyyy-MM-dd'T'HH:mm:ss.SSS"))
            .filter(F.col("event_time").isNotNull() & F.col("service").isNotNull())
            .withColumn("is_error", (F.col("status_code") >= 500).cast("int")))


def windowed(events, window, watermark):
    return (events.withWatermark("event_time", watermark)
            .groupBy(F.window("event_time", window).alias("w"), "service")
            .agg(F.count("*").alias("req_count"), F.sum("is_error").alias("error_count"),
                 F.avg("latency_ms").alias("avg_latency"),
                 F.percentile_approx("latency_ms", 0.95).alias("p95_latency"),
                 F.max("truth_k").alias("truth_k"))
            .select(F.col("w.start").alias("window_start"), "service", "req_count", "error_count",
                    "avg_latency", "p95_latency", "truth_k"))


def safe(name, fn):
    def wrapper(df, batch_id):
        try:
            fn(df, batch_id)
        except Exception:                                  # keep the stream alive, show the error
            print(f"[{name}] batch {batch_id} failed:\n{traceback.format_exc()}")
    return wrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files-dir", help="(testing only) read JSON-lines files instead of Kafka")
    ap.add_argument("--keep", action="store_true", help="keep the previous run's output tables")
    args = ap.parse_args()

    spark = get_spark("cloudlog-stream", kafka=not args.files_dir, shuffle_partitions=4, cores=4)
    if args.files_dir:
        source = spark.readStream.format("text").load(args.files_dir)
    else:
        source = (spark.readStream.format("kafka")
                  .option("kafka.bootstrap.servers", C.KAFKA_BOOTSTRAP).option("subscribe", C.KAFKA_TOPIC)
                  .option("startingOffsets", "latest").option("failOnDataLoss", "false")
                  .option("maxOffsetsPerTrigger", 20000).load())
    events = parse_events(source)
    run = f"{C.CHECKPOINT_ROOT}/run_{int(time.time())}"      # fresh checkpoint every run

    if not args.keep and C.STREAM_OUT.exists():
        for f in C.STREAM_OUT.glob("*.parquet"):
            f.unlink()

    raw_table = RollingTable(C.STREAM_OUT / "raw_events.parquet", 100_000)
    live_table = RollingTable(C.STREAM_OUT / "live_10s.parquet", 20_000)
    processor = MinuteProcessor()
    rng = np.random.default_rng(1)

    # ---- A) raw events -> HDFS + recent warnings/errors for drill-down
    def raw_batch(df, bid):
        df = df.persist()
        if df.count() > 0:
            df.coalesce(1).write.mode("append").parquet(C.STREAM_RAW_HDFS)
            p = df.select("event_time", "service", "level", "status_code", "latency_ms", "host",
                          "endpoint", "message", "truth_k").toPandas()
            keep = (p["level"] != "INFO") | (rng.random(len(p)) < 0.02)   # all WARN/ERROR + 2% of INFO
            raw_table.append(p[keep])
        df.unpersist()

    # ---- B) 10-second live metrics
    def live_batch(df, bid):
        p = df.toPandas()
        if len(p):
            p["error_rate"] = p["error_count"] / p["req_count"].clip(lower=1)
            live_table.append(p.sort_values("window_start"))

    # ---- C) 1-minute windows -> ML -> incidents
    def minute_batch(df, bid):
        p = df.toPandas()
        if len(p):
            n, a = processor.process(p)
            print(f"[minute] batch {bid}: scored {n} service-minutes, {a} anomalous")

    q = []
    q.append(events.writeStream.foreachBatch(safe("raw", raw_batch)).outputMode("append")
             .option("checkpointLocation", f"{run}/raw").trigger(processingTime="10 seconds").start())
    q.append(windowed(events, "10 seconds", "10 seconds").writeStream
             .foreachBatch(safe("live", live_batch)).outputMode("append")
             .option("checkpointLocation", f"{run}/live").trigger(processingTime="5 seconds").start())
    q.append(windowed(events, "1 minute", "20 seconds").writeStream
             .foreachBatch(safe("minute", minute_batch)).outputMode("append")
             .option("checkpointLocation", f"{run}/minute").trigger(processingTime="15 seconds").start())
    print("streaming started - Spark UI: http://localhost:4040   (Ctrl+C to stop)")
    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        for query in q:
            query.stop()
        spark.stop()


if __name__ == "__main__":
    main()