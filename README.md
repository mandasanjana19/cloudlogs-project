# CloudLog Intelligence
### Real-Time Distributed Log Analytics and Incident Detection

CloudLog Intelligence combines historical HDFS logs with simulated live microservice events. It uses Hadoop HDFS for storage, Apache Spark/PySpark for batch analytics, Apache Kafka for event ingestion, and Spark Structured Streaming for windowed processing. Rule-based checks and machine learning models help identify unusual service behaviour and group related anomalies into potential incidents.

> **Current test status:** The latest shared output recorded one high-severity incident (`INC-001`) involving `database` and `payments`. Two cascading-failure injection requests were issued, but the available output confirms only one incident record—not two separately detected incidents.

## Features

- Historical HDFS log parsing and feature extraction with PySpark.
- Kafka producer that simulates microservice events and publishes JSON records to the `app-logs` topic.
- Spark Structured Streaming with 10-second and 1-minute windows.
- Service metrics: request count, error rate, average latency, and p95 latency.
- Rule-based anomaly signals, Isolation Forest outlier detection, and XGBoost incident-risk scoring.
- Incident grouping and Parquet/HDFS output for inspection.

## Technology Stack

| Technology | Role |
|---|---|
| Python | Simulation, feature engineering, model training, and orchestration |
| Docker / Docker Compose | Local services and reproducible environment |
| Hadoop HDFS | Historical log and streaming output storage |
| Apache Kafka | Live event ingestion through the `app-logs` topic |
| Apache Spark / PySpark | Historical batch processing and feature extraction |
| Spark Structured Streaming | Live windowed metrics and scoring |
| XGBoost | Incident-risk model |
| Isolation Forest | Unsupervised outlier detection |
| Parquet | Columnar storage for processed data and streaming outputs |
| Streamlit | Dashboard/UI component where configured |

The current workflow is designed to run locally. Cloud services such as Amazon S3, MSK, EMR, and RDS are not required for the Docker-based pipeline described here.

## Architecture

```text
Historical HDFS logs
        |
        v
    Hadoop HDFS
        |
        v
 PySpark Batch Job
        |
        +----> Parsed logs and block features
        |
        +----> ML model training
               (XGBoost / Isolation Forest)


Simulated Microservice Events
        |
        v
 Python Kafka Producer
        |
        v
 Kafka Topic: app-logs
        |
        v
 Spark Structured Streaming
        |
        +----> Raw events ----------> HDFS
        |
        +----> 10-second metrics ---> live_10s.parquet
        |
        +----> 1-minute metrics
                    |
                    v
             ML and Rule Scoring
                    |
                    +----> minute_scores.parquet
                    +----> anomalies.parquet
                    +----> incidents.parquet
```

## Project Structure

```text
cloudlog/
├── src/
│   ├── batch/
│   │   └── hdfs_batch.py
│   ├── common/
│   │   ├── config.py
│   │   ├── features.py
│   │   ├── incidents.py
│   │   ├── simulator.py
│   │   ├── spark_utils.py
│   │   └── store.py
│   ├── generator/
│   │   ├── events.py
│   │   ├── inject.py
│   │   └── log_producer.py
│   ├── ml/
│   │   ├── train_hdfs_block_model.py
│   │   └── train_incident_model.py
│   ├── streaming/
│   │   ├── processing.py
│   │   └── stream_job.py
│   └── utils/
│       └── peek.py
├── docker-compose.yml
├── Dockerfile
└── README.md
```

## Prerequisites

- Windows with Docker Desktop and WSL 2 enabled, or a compatible Docker environment.
- Git.
- Project input datasets and model artifacts required by the relevant scripts.

## Running the Project Locally

Open PowerShell and navigate to the project directory:

```powershell
cd C:\Users\manda\cloudlog
```

Start the Docker services:

```powershell
docker compose up -d --build
docker compose ps
```

The following commands assume the containers are named `cloudlog-app`, `kafka`, and `namenode`, as in the current local setup.

### 1. Start Spark Structured Streaming

Open **Terminal 1** and run:

```powershell
docker exec -it cloudlog-app python -m src.streaming.stream_job
```

Leave this terminal running.

- Spark UI: `http://localhost:4040` while the streaming job is running.
- HDFS NameNode UI: `http://localhost:9870`.

### 2. Start the Simulated Log Producer

Open **Terminal 2** and run:

```powershell
docker exec -it cloudlog-app python -m src.generator.log_producer
```

The producer generates simulated microservice events and sends them to Kafka.

To enable automatic random fault injection:

```powershell
docker exec -it cloudlog-app python -m src.generator.log_producer --auto
```

Keep the producer and Spark stream job running together so Spark can consume the generated events.

### 3. Inject a Failure Scenario

With the producer and Spark stream job running, open another PowerShell terminal and execute:

```powershell
docker exec -it cloudlog-app python -m src.generator.inject cascading_failure
```

The current cascading-failure request targets the database for 420 seconds.

Other scenario names supported by the generator include:

- `error_spike`
- `service_down`
- `latency_degradation`
- `traffic_burst`

Check the generator's accepted arguments for service and duration options.

Allow enough time for events to arrive and for the 1-minute aggregation and scoring window to complete.

### 4. Inspect the Results

Run:

```powershell
docker exec -it cloudlog-app python -m src.utils.peek
```

The inspection utility displays summaries of the streaming outputs.

| Output | Description |
|---|---|
| `raw_events.parquet` | Rolling sample of individual events and counts by service and log level |
| `live_10s.parquet` | 10-second request counts, error rates, average latency, and p95 latency |
| `minute_scores.parquet` | 1-minute service metrics, risk scores, anomaly scores, and detection signals |
| `anomalies.parquet` | Records flagged by the detection logic |
| `incidents.parquet` | Related anomalies grouped into incident records |

Raw streaming Parquet files are written under `/cloudlog/stream/raw_events` in HDFS.

Checkpoints are created under `/cloudlog/checkpoints/` in separate run directories. The presence of checkpoint folders alone does not prove that restart recovery has been tested.

## Historical Batch Processing and Model Training

The project includes a Spark batch job for historical HDFS logs and model-training scripts.

When the required input datasets and configuration are available in the container, run:

```powershell
docker exec -it cloudlog-app python -m src.batch.hdfs_batch
```

Train the HDFS block anomaly model:

```powershell
docker exec -it cloudlog-app python -m src.ml.train_hdfs_block_model
```

Train the incident-risk model:

```powershell
docker exec -it cloudlog-app python -m src.ml.train_incident_model
```

These commands use the current module entry points and depend on the expected input data, dependencies, and configuration being available.

## Understanding the Streaming Output

### Service Metrics

- **`req_count`**: Number of events or requests for a service during a window.
- **`error_rate`**: Fraction of requests marked as errors.
- **`avg_latency`**: Average response latency.
- **`p95_latency`**: Latency value below which approximately 95% of requests fall.

### Detection Results

- **`risk`**: Model-estimated incident risk; it is not itself a confirmed incident.
- **`anomaly_score`**: Score from the anomaly-detection path.
- **`signals`**: Signals such as `error_spike`, `latency_high`, and `ml_outlier`.
- **`root_cause`**: A candidate selected by the incident tracker, not proof of causation.

A high risk score does not automatically create a new incident. The incident tracker can group anomalies from multiple services into one incident record.

## Latest Observed Streaming Test

The latest supplied `peek` output showed:

- The producer was running, with approximately 18,048 events sent at the time of inspection.
- The displayed 10-second window showed a payments error rate of `0.343` and an orders error rate of `0.172`.
- The 1-minute scores showed a database risk of `0.974` and payments risk of `0.967`.
- `INC-001` was recorded as **OPEN / HIGH**, involving `database` and `payments`.
- The selected root-cause candidate was `database`.
- The incident signals were `error_spike`, `latency_high`, and `ml_outlier`.
- Two cascading-failure injection requests had been issued, but the shown output confirmed only one incident record.

These are observations from one local simulation run, not performance guarantees. Results may vary depending on event timing, thresholds, model artifacts, and test duration.

## Git Hygiene

Commit source code, configuration, documentation, and small evaluation reports. Avoid committing:

- Raw datasets such as `HDFS.log` and `anomaly_label.csv`.
- Large generated Parquet outputs and Spark checkpoints.
- Credentials, secrets, and `.env` files.
- Python caches, temporary logs, and virtual environments.

Check `.gitignore` and run `git status` before committing.

## Next Validation Steps

- Repeat the controlled fault test and verify whether a second injection creates a new incident or is grouped into the open incident.
- Test Spark restart and recovery using checkpoints.
- Add dashboard screenshots and a reproducible demo walkthrough once the UI flow is finalized.
- Record repeatable model metrics and throughput results before making performance claims.

## Project Summary

CloudLog Intelligence demonstrates a compact big-data workflow combining historical log analytics with Hadoop HDFS and Apache Spark, live event ingestion with Apache Kafka, windowed processing with Spark Structured Streaming, and rule/ML-assisted anomaly and incident detection.

The project is designed as a local educational prototype that makes both batch and streaming outputs inspectable.
