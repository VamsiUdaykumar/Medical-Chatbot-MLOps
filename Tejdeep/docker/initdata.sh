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

echo "Configuring rclone credentials..."
mkdir -p ~/.config/rclone
cat <<EOF > ~/.config/rclone/rclone.conf
[chi_tacc]
type = swift
user_id = $YOUR_USER_ID
application_credential_id = $APP_CRED_ID
application_credential_secret = $APP_CRED_SECRET
auth = https://chi.tacc.chameleoncloud.org:5000/v3
region = CHI@TACC
EOF

echo "Listing remote storage..."
rclone lsd chi_tacc:

echo "Mounting on local file system..."
sudo mkdir -p /mnt/object
sudo chown -R cc /mnt/object
sudo chgrp -R cc /mnt/object

export RCLONE_CONTAINER=object-persist-project17
rclone mount chi_tacc:$RCLONE_CONTAINER /mnt/object --allow-other --daemon

echo "Contents of /mnt/object:"
ls /mnt/object
