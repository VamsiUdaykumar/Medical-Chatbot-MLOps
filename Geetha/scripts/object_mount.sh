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

export YOUR_USER_ID='d25b59d03a410639b308fc4180ae27bbdcb3eaafc746715255092b23166344ae'; 
export APP_CRED_ID='b927f3996058472a81c13c5f2f34d2e7'; 
export APP_CRED_SECRET='GiGd8lzNvcJ-m7qeZCU-5oQqDZtyhNrlMtlLdmGU07WRe76AxPDyglTj_ZKpS9P6csx3HSnDLFPQMP307QJxuw';

rclone lsd chi_tacc:

echo "Mounting on local file system"
sudo mkdir -p /mnt/object
sudo chown -R cc /mnt/object
sudo chgrp -R cc /mnt/object

rclone mount chi_tacc:object-persist-project17 /mnt/object --allow-other --daemon

ls /mnt/object