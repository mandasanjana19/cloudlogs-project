"""One place to create a Spark session with settings that are safe for a 16 GB laptop."""
import os

from pyspark.sql import SparkSession
from src.common.config import KAFKA_PACKAGE


def get_spark(app_name, kafka=False, shuffle_partitions=24, cores=6,
              driver_mem="3g", extra_packages=None):
    packages = list(extra_packages or [])
    if kafka:
        packages.append(KAFKA_PACKAGE)
    builder = (SparkSession.builder.appName(app_name)
               .master(f"local[{int(os.getenv('CLOUDLOG_CORES', cores))}]")  # driver + executors in one JVM
               .config("spark.driver.memory", driver_mem)
               .config("spark.driver.maxResultSize", "1g")
               .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
               .config("spark.sql.session.timeZone", "UTC")     # all timestamps are UTC
               .config("spark.ui.showConsoleProgress", "false"))
    if packages:
        builder = builder.config("spark.jars.packages", ",".join(packages))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark