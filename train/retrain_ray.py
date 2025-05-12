import os
import pandas as pd

def get_latest_versioned_data_path(base_dir="/mnt/object/data/production/retraining_data_transformed"):
    version_file = os.path.join(base_dir, "version_tracker.txt")
    if not os.path.exists(version_file):
        raise FileNotFoundError(f"version_tracker.txt not found in {base_dir}")
    with open(version_file, "r") as f:
        last_version = f.read().strip()
    version_dir = f"v{last_version}"
    data_path = os.path.join(base_dir, version_dir, "retraining_data.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Expected data file not found: {data_path}")
    return data_path, version_dir

data_path, version_tag = get_latest_versioned_data_path()
df = pd.read_json(data_path, lines=True)