#!/bin/bash

set -e

# run on node-persist
curl https://rclone.org/install.sh | sudo bash

sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf

mkdir -p ~/.config/rclone

echo "[chi_tacc]
type = swift
user_id = $YOUR_USER_ID
application_credential_id = $APP_CRED_ID
application_credential_secret = $APP_CRED_SECRET
auth = https://chi.tacc.chameleoncloud.org:5000/v3
region = CHI@TACC" > ~/.config/rclone/rclone.conf

rclone lsd chi_tacc:

echo "Setting RCLONE_CONTAINER..."
export RCLONE_CONTAINER=object-persist-project17

echo "Running extract stage..."
docker compose -f ./docker-compose-etl.yaml run extract-data

echo "Running transform stage..."
docker compose -f ./docker-compose-etl.yaml run transform-data

echo "Running load stage..."
docker compose -f ./docker-compose-etl.yaml run load-data

echo "Cleaning up Docker volume..."
docker volume ls
docker volume rm $(docker volume ls -q --filter name=medicaldata) || echo "No volume found to remove."

ls /mnt/object