FROM python:3.11-slim

WORKDIR /app

# system deps for numpy / pyarrow wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# data dir is mounted as a volume; pre-create so the pipeline can write
RUN mkdir -p /app/data
