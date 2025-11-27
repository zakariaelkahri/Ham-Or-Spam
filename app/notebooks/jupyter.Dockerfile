# Use the official Jupyter PySpark notebook as base
FROM jupyter/pyspark-notebook:python-3.10

# Switch to root to install additional packages
USER root

# Update package list and install wget and OpenJDK 11
RUN apt-get update && \
    apt-get install -y wget curl openjdk-11-jdk-headless ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables for Java (OpenJDK 11)
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Set environment variables for Spark (already installed)
ENV SPARK_HOME=/usr/local/spark
ENV PATH=$SPARK_HOME/bin:$PATH

# Download and install Hadoop
ENV HADOOP_VERSION=3.3.6
ENV HADOOP_HOME=/opt/hadoop
ENV HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
ENV PATH=$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH

# Download and extract Hadoop
RUN wget https://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz && \
    tar -xzf hadoop-${HADOOP_VERSION}.tar.gz && \
    mv hadoop-${HADOOP_VERSION} ${HADOOP_HOME} && \
    rm hadoop-${HADOOP_VERSION}.tar.gz

# Set Hadoop environment
RUN echo "export JAVA_HOME=${JAVA_HOME}" >> ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh

# Create necessary directories for Hadoop
RUN mkdir -p ${HADOOP_HOME}/logs && \
    mkdir -p /tmp/hadoop-root

# Set proper permissions
RUN chown -R $NB_UID:$NB_GID ${HADOOP_HOME} && \
    chmod -R 755 ${HADOOP_HOME}

# Install additional Python packages for big data processing
RUN pip install --no-cache-dir \
    pymongo \
    hdfs3 \
    pyarrow \
    findspark \
    pyhive \
    nltk

# Create a startup script to initialize Spark properly
RUN echo '#!/bin/bash' > /usr/local/bin/init-spark.sh && \
    echo 'export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64' >> /usr/local/bin/init-spark.sh && \
    echo 'export SPARK_HOME=/usr/local/spark' >> /usr/local/bin/init-spark.sh && \
    echo 'export HADOOP_HOME=/opt/hadoop' >> /usr/local/bin/init-spark.sh && \
    echo 'export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop' >> /usr/local/bin/init-spark.sh && \
    echo 'export PYSPARK_PYTHON=python3' >> /usr/local/bin/init-spark.sh && \
    echo 'export PYSPARK_DRIVER_PYTHON=python3' >> /usr/local/bin/init-spark.sh && \
    chmod +x /usr/local/bin/init-spark.sh

# Configure PySpark to work with Hadoop
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# Configure Spark classpath and JVM options
ENV SPARK_CLASSPATH=$HADOOP_HOME/etc/hadoop:$SPARK_HOME/jars/*:$HADOOP_HOME/share/hadoop/common/*:$HADOOP_HOME/share/hadoop/hdfs/*:$HADOOP_HOME/share/hadoop/yarn/*
ENV SPARK_OPTS="--driver-java-options=-Xms1024M --driver-java-options=-Xmx4096M --driver-java-options=-Dlog4j.logLevel=INFO"

# Switch back to jovyan user
USER $NB_UID

# Expose port
EXPOSE 8889

# Set work directory
WORKDIR /home/jovyan/work