from pyspark.sql import SparkSession
# This starts a Spark session using four local worker threads.
spark = (SparkSession.builder.appName("stack-test").master("local[4]")
         .config("spark.driver.memory", "3g")
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.jars.packages",
                 "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1")
         .getOrCreate())

# This creates a tiny, artificial table with two rows and writes it to HDFS in Parquet format.
# We use sample rows instead of the real dataset because this is only a connectivity test.
# The purpose: Check whether Spark can store data in Hadoop.

# 1) Spark -> HDFS round trip
df = spark.createDataFrame([(1, "auth", "INFO"), (2, "db", "ERROR")],
                           ["id", "service", "level"])
df.write.mode("overwrite").parquet("hdfs://namenode:9000/cloudlog/test/parquet")
spark.read.parquet("hdfs://namenode:9000/cloudlog/test/parquet").show()

# 2) Kafka -> Spark

# This tells Spark to read messages from Kafka's app-logs topic, 
# including messages that were already there when the test started.

k = (spark.read.format("kafka")
     .option("kafka.bootstrap.servers", "kafka:19092")
     .option("subscribe", "app-logs")
     .option("startingOffsets", "earliest").load())
k.selectExpr("CAST(value AS STRING) AS msg", "partition", "offset").show(truncate=False)

spark.stop()