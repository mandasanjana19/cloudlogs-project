"""Central configuration: every path, topic name and constant lives here."""
import os
from pathlib import Path

# ---- local folders (inside the container the project is mounted at /workspace)
ROOT = Path(os.getenv("CLOUDLOG_ROOT", "/workspace"))
OUT = ROOT / "output"
BATCH_OUT = OUT / "batch"        # small analytics tables from the Spark batch job
ML_OUT = OUT / "ml"              # model metrics, curves, importances
STREAM_OUT = OUT / "stream"      # live tables written by the streaming job
CONTROL_DIR = OUT / "control"    # dashboard -> producer messages (failure injection)
MODELS = ROOT / "models"

# ---- HDFS (the data lake)
HDFS_ROOT = os.getenv("CLOUDLOG_HDFS_ROOT", "hdfs://namenode:9000/cloudlog")
RAW_LOG = f"{HDFS_ROOT}/raw/HDFS.log"
RAW_LABELS = f"{HDFS_ROOT}/raw/anomaly_label.csv"
PARSED = f"{HDFS_ROOT}/processed/parsed_logs"
BLOCK_FEATURES = f"{HDFS_ROOT}/processed/block_features"
STREAM_RAW_HDFS = f"{HDFS_ROOT}/stream/raw_events"
CHECKPOINT_ROOT = f"{HDFS_ROOT}/checkpoints"

# ---- Kafka
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:19092")
KAFKA_TOPIC = "app-logs"
KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"

# ---- simulated microservices
SERVICES = ["gateway", "auth", "orders", "payments", "database"]
SERVICE_IDS = {s: i for i, s in enumerate(SERVICES)}