#!/bin/bash

set -e

echo "🔄 Starting Retraining ETL Pipeline..."

# Step 1: Ensure required files exist
if [[ ! -f docker-compose-retraining-etl.yaml ]]; then
  echo "docker-compose-retraining-etl.yaml not found!"
  exit 1
fi

if [[ ! -f retraining_data_transform.py ]]; then
  echo "retraining_data_transform.py not found!"
  exit 1
fi

# Step 2: Pull base image
echo "Pulling base Python image..."
docker pull python:3.11

# Step 3: Run the ETL pipeline
echo "Launching Docker Compose for retraining ETL..."
docker compose -f docker-compose-retraining-etl.yaml up --build --abort-on-container-exit

# Step 4: Cleanup
echo "Stopping and cleaning up..."
docker compose -f docker-compose-retraining-etl.yaml down

echo "ETL completed successfully."
