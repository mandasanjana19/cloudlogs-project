"""PHASE 2 - Spark batch job on the LogHub HDFS_v1 dataset (11M log lines).

Steps:  raw text (HDFS)  ->  parse + normalise  ->  parsed parquet (HDFS)
        -> analytics tables for the dashboard
        -> one feature row per HDFS block (event counts) + label  -> ML training data
"""
import time

from pyspark.sql import functions as F

from src.common import config as C
from src.common.spark_utils import get_spark
from src.common.store import write_json, write_parquet

TOP_K = 40  # number of most frequent event templates kept as features (the rest -> OTHER)
LOG_RE = r"^(\d{6})\s+(\d{6})\s+(\d+)\s+(\w+)\s+([\w\.\$]+):\s*(.*)$"
# example line: 081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-16 src: /10.2.1.3:54106 ...


def normalise(col):
    """Turn a log message into a template by masking the variable parts."""
    col = F.regexp_replace(col, r"blk_-?\d+", "<BLK>")
    col = F.regexp_replace(col, r"\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?", "<IP>")
    col = F.regexp_replace(col, r"\d+", "<NUM>")
    col = F.regexp_replace(col, r"(?:<BLK>\s?)+", "<BLK*> ")   # lists of blocks -> one token
    col = F.regexp_replace(col, r"(?:<IP>\s?)+", "<IP*> ")     # lists of addresses -> one token
    return F.trim(col)


def main():
    t0 = time.time()
    spark = get_spark("cloudlog-batch", shuffle_partitions=24)

    # ---------------------------------------------------------------- 1. parse raw lines
    raw = spark.read.text(C.RAW_LOG)
    parsed = (raw.select(
        F.regexp_extract("value", LOG_RE, 1).alias("d"),
        F.regexp_extract("value", LOG_RE, 2).alias("t"),
        F.regexp_extract("value", LOG_RE, 3).cast("int").alias("pid"),
        F.regexp_extract("value", LOG_RE, 4).alias("level"),
        F.regexp_extract("value", LOG_RE, 5).alias("component"),
        F.regexp_extract("value", LOG_RE, 6).alias("content"))
        .filter(F.col("d") != "")                                   # drop unparseable lines
        .withColumn("ts", F.to_timestamp(F.concat("d", "t"), "yyMMddHHmmss"))
        .filter(F.col("ts").isNotNull())                            # drop lines with invalid time
        .withColumn("event_date", F.to_date("ts"))
        .withColumn("template", normalise(F.col("content")))
        .withColumn("block_ids", F.regexp_extract_all(F.col("content"), F.lit(r"(blk_-?\d+)"), 1))
        .select("ts", "event_date", "pid", "level", "component", "content", "template", "block_ids"))

    parsed.write.mode("overwrite").partitionBy("event_date").parquet(C.PARSED)
    df = spark.read.parquet(C.PARSED)          # re-read the columnar copy: much faster from now on
    print(f"[1/5] parsed logs written to {C.PARSED}  ({time.time() - t0:.0f}s)")

    # ---------------------------------------------------------------- 2. dashboard tables
    per_minute = (df.groupBy(F.date_trunc("minute", "ts").alias("minute"))
                  .agg(F.count("*").alias("events"),
                       F.sum((F.col("level") == "WARN").cast("int")).alias("warns"),
                       F.sum((F.col("level") == "ERROR").cast("int")).alias("errors"))
                  .orderBy("minute")).toPandas()
    level_component = df.groupBy("component", "level").count().toPandas()
    hourly = (df.groupBy(F.date_trunc("hour", "ts").alias("hour"), "component")
              .count().orderBy("hour")).toPandas()
    template_counts = df.groupBy("template").count().orderBy(F.desc("count"))
    top = template_counts.limit(TOP_K).collect()
    tpl_pd = template_counts.limit(500).toPandas()
    event_map = {r["template"]: i for i, r in enumerate(top)}
    tpl_pd["event_id"] = tpl_pd["template"].map(event_map).fillna(TOP_K).astype(int)
    for name, pdf in [("events_per_minute", per_minute), ("level_by_component", level_component),
                      ("events_per_hour_component", hourly), ("templates", tpl_pd)]:
        write_parquet(pdf, C.BATCH_OUT / f"{name}.parquet")
    total_lines = int(per_minute["events"].sum())
    print(f"[2/5] dashboard tables saved ({time.time() - t0:.0f}s)")

    # ---------------------------------------------------------------- 3. per-block features
    map_df = spark.createDataFrame(list(event_map.items()), ["template", "event_id"])
    events = (df.select("ts", "level", "component", "template",
                        F.explode("block_ids").alias("block_id"))   # one row per (line, block)
              .join(F.broadcast(map_df), "template", "left")
              .withColumn("event_id", F.coalesce("event_id", F.lit(TOP_K))))
    aggs = [F.count("*").alias("n_events"),
            (F.unix_timestamp(F.max("ts")) - F.unix_timestamp(F.min("ts"))).alias("duration_s"),
            F.sum((F.col("level") == "WARN").cast("int")).alias("n_warn"),
            F.sum((F.col("level") == "ERROR").cast("int")).alias("n_error"),
            F.countDistinct("component").alias("n_components"),
            F.min("ts").alias("start_ts")]
    aggs += [F.sum((F.col("event_id") == i).cast("int")).alias(f"E{i}") for i in range(TOP_K + 1)]
    blocks = events.groupBy("block_id").agg(*aggs)
    labels = (spark.read.csv(C.RAW_LABELS, header=True)
              .select(F.col("BlockId").alias("block_id"),
                      (F.col("Label") == "Anomaly").cast("int").alias("label")))
    feats = blocks.join(labels, "block_id", "inner")
    feats.write.mode("overwrite").parquet(C.BLOCK_FEATURES)
    feats = spark.read.parquet(C.BLOCK_FEATURES)
    print(f"[3/5] block features written to {C.BLOCK_FEATURES}  ({time.time() - t0:.0f}s)")

    # ---------------------------------------------------------------- 4. export for ML + dashboard
    feats_pd = feats.toPandas()
    write_parquet(feats_pd, C.BATCH_OUT / "block_features.parquet")
    n_blocks, n_bad = len(feats_pd), int(feats_pd["label"].sum())
    print(f"[4/5] {n_blocks:,} blocks, {n_bad:,} anomalous exported ({time.time() - t0:.0f}s)")

    # a small sample of blocks with their raw lines, for the "Block explorer" in the dashboard
    sample_ids = (feats.filter("label = 1").orderBy(F.rand(42)).limit(500).select("block_id")
                  .union(feats.filter("label = 0").orderBy(F.rand(43)).limit(500).select("block_id")))
    sample_logs = (df.select("ts", "level", "component", "content",
                             F.explode("block_ids").alias("block_id"))
                   .join(F.broadcast(sample_ids), "block_id").orderBy("block_id", "ts")).toPandas()
    write_parquet(sample_logs, C.BATCH_OUT / "block_logs_sample.parquet")

    span = per_minute["minute"]
    write_json({"total_lines": total_lines, "blocks": n_blocks, "anomalous_blocks": n_bad,
                "start": str(span.min()), "end": str(span.max()),
                "n_templates_total": int(template_counts.count()), "top_k": TOP_K},
               C.BATCH_OUT / "overview.json")
    print(f"[5/5] done in {time.time() - t0:.0f}s")
    spark.stop()


if __name__ == "__main__":
    main()