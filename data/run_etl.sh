#!/bin/bash

echo "Running extract stage..."
docker compose -f ./data/etl.yaml run extract-data

echo "Running transform stage..."
docker compose -f ./data/etl.yaml run transform-data

echo "Setting RCLONE_CONTAINER..."
export RCLONE_CONTAINER=object-persist-project17

echo "Running load stage..."
docker compose -f ./data/etl.yaml run load-data

echo "Cleaning up Docker volume..."
docker volume ls
docker volume rm $(docker volume ls -q --filter name=medicaldata) || echo "No volume found to remove."
