#!/bin/bash

set -e  # Exit on any error

# Step 1: Create mount point if it doesn't exist
MOUNT_POINT="/mnt/block"
DEVICE="/dev/vdb1"  # Change if needed

echo "Creating mount point at $MOUNT_POINT..."
sudo mkdir -p "$MOUNT_POINT"

# Step 2: Mount the block volume
echo "Mounting $DEVICE to $MOUNT_POINT..."
sudo mount "$DEVICE" "$MOUNT_POINT"

# Step 3: Confirm contents (optional)
echo "Listing contents of $MOUNT_POINT:"
ls "$MOUNT_POINT"

# Step 4: Restart Docker services using block-volume-backed compose file
COMPOSE_FILE="$HOME/Medical-Chatbot-MLOps/docker/docker-compose-persistant-storage.yaml"
HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4)

echo "Bringing up Docker services from $COMPOSE_FILE using HOST_IP=$HOST_IP..."
HOST_IP="$HOST_IP" docker compose -f "$COMPOSE_FILE" up -d

echo "Block volume mounted and services started successfully."
