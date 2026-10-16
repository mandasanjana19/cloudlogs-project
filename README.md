# Cloud Log Intelligence Platform

A containerized big-data analytics platform for processing HDFS logs, generating block-level features, detecting anomalous activity, and building toward real-time incident monitoring.

## Project Overview

This project combines Hadoop HDFS, Apache Kafka, Apache Spark, Python, and machine learning to explore batch log processing and real-time anomaly detection.

## Technology Stack

- **Hadoop HDFS:** Distributed storage for raw and processed logs
- **Apache Kafka:** Event streaming and message ingestion
- **Apache Spark / PySpark:** Log parsing, batch processing, and data transformation
- **Python, Pandas, NumPy:** Data processing and feature engineering
- **Parquet:** Structured, compressed data storage
- **Docker Compose:** Containerized development environment
- **Scikit-learn and XGBoost:** Planned model training and anomaly detection
- **Streamlit:** Planned analytics dashboard

## Implementation Progress

### Phase 1 — Infrastructure Setup

- Configured the Docker-based development environment.
- Started Hadoop NameNode, DataNode, Kafka, and the Python application container.
- Created the HDFS raw-data directory.
- Created a Kafka topic named `app-logs` and verified message production and consumption.
- Verified basic Spark integration with HDFS and Kafka.

### Phase 2 — Batch Log Processing

- Uploaded the HDFS log dataset and anomaly labels into HDFS.
- Used PySpark to parse raw log messages into structured records.
- Extracted timestamps, log levels, components, message content, block identifiers, and normalized log templates.
- Stored parsed logs in date-partitioned Parquet output.
- Generated dashboard analytics and block-level features.
- Joined block features with anomaly labels.

**Current dataset summary:**
- Total blocks: 575,061
- Normal blocks: 558,223
- Anomalous blocks: 16,838
- Block feature table: 49 columns

### Phase 3 — Machine Learning (Next)

Planned work:

- Train and compare Logistic Regression, Isolation Forest, and XGBoost on the labeled HDFS block dataset.
- Evaluate precision, recall, F1-score, ROC-AUC, PR-AUC, and the confusion matrix.
- Build a simulated microservice incident-risk model for the streaming pipeline.
- Save trained models, prediction scores, feature importance, and evaluation metrics.

The Phase 3 training scripts and results are not yet completed.

### Phase 4 — Real-Time Streaming (Planned)

- Stream log or service-metric events through Kafka.
- Process events using Spark Structured Streaming.
- Apply trained models to detect anomalies and estimate incident risk.
- Integrate the results into a Streamlit dashboard.

## Current Status

Infrastructure setup and the Phase 2 batch-processing pipeline have been completed. Machine learning and real-time streaming are the next development milestones.

## Dataset

The project uses the LogHub HDFS log dataset and its anomaly labels. Large datasets and generated artifacts should be kept outside Git version control unless intentionally published.
