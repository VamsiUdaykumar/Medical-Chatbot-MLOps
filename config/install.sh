#!/bin/bash

set -e  # Stop on error
echo "Starting setup..."

# Step 1: System packages
echo "Installing Docker..."
sudo apt update
sudo apt install -y docker.io git

# Step 2: Clone your project repo
echo "Cloning project repo..."
git clone https://github.com/phoenix1881/Medical-Chatbot-MLOps.git
cd Medical-Chatbot-MLOps

# Step 3: Build Docker image
echo "Building Docker image..."
sudo docker build -t dr-dialog .

# Step 4: Run container
echo "Running container on port 5000..."
sudo docker run -d -p 5000:5000 --name dr-dialog-app dr-dialog

echo "Setup complete. App should be live on port 5000."
