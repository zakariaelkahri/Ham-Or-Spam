# Base Python image
FROM python:3.10-slim

# Install Java for PySpark
RUN apt-get update && \
    apt-get install -y default-jre-headless wget curl && \
    rm -rf /var/lib/apt/lists/*

# Set Java environment
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$JAVA_HOME/bin:$PATH

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY app/requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY app/ .


# Expose Streamlit port
# EXPOSE 8501

# Run Streamlit
# CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
