import os
import pandas as pd
import json
import shutil
from datetime import datetime

RAW_DIR = "/data_raw"
TRANSFORMED_DIR = "/data_transformed"
ARCHIVE_DIR = "/data_archive"
VERSION_FILE = os.path.join(TRANSFORMED_DIR, "version_tracker.txt")

#  Determine next version number
if os.path.exists(VERSION_FILE):
    with open(VERSION_FILE, "r") as f:
        last_version = int(f.read().strip())
else:
    existing_versions = [
        int(d.replace("v", "")) for d in os.listdir(TRANSFORMED_DIR)
        if os.path.isdir(os.path.join(TRANSFORMED_DIR, d)) and d.startswith("v") and d[1:].isdigit()
    ]
    last_version = max(existing_versions, default=0)

next_version = last_version + 1
version_tag = f"v{next_version}"

# Update version tracker
with open(VERSION_FILE, "w") as f:
    f.write(str(next_version))

# Create output and archive directories
version_dir = os.path.join(TRANSFORMED_DIR, version_tag)
versioned_archive_dir = os.path.join(ARCHIVE_DIR, version_tag)
os.makedirs(version_dir, exist_ok=True)
os.makedirs(versioned_archive_dir, exist_ok=True)

#  Read and process raw files
records = []
archived_files = []

print("Scanning raw data directory...")
for fname in os.listdir(RAW_DIR):
    file_path = os.path.join(RAW_DIR, fname)

    # Skip non-JSON files
    if not fname.endswith(".json"):
        print(f"Skipping non-JSON file: {fname}")
        continue

    print(f"Found file: {fname}")
    try:
        with open(file_path, "r") as f:
            try:
                # Try JSON array
                data = json.load(f)
                if isinstance(data, list):
                    records.extend(data)
                else:
                    print(f"Warning: Skipping {fname} (not a JSON list)")
                    continue
            except json.JSONDecodeError:
                # Try JSONL
                print(f"Falling back to JSONL for {fname}")
                f.seek(0)
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Skipping bad line in {fname}: {e}")

        # Move processed file to archive
        archived_files.append(fname)
        shutil.move(file_path, os.path.join(versioned_archive_dir, fname))

    except Exception as e:
        print(f"Failed to process {fname}: {e}")

if not records:
    print("No new valid retraining data to process.")
    exit(0)

# Clean the DataFrame
df = pd.DataFrame(records)
df = df.dropna(subset=["question", "answer"])
df = df[~df["question"].str.strip().eq("")]
df = df[~df["answer"].str.strip().eq("")]
df = df.drop(columns=["symptoms", "timestamp"], errors="ignore")

# Save cleaned data
output_path = os.path.join(version_dir, "retraining_data.json")
df.to_json(output_path, orient="records", lines=True)
print(f"Saved cleaned data to: {output_path}")

#  Write metadata
metadata = {
    "version": version_tag,
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "record_count": len(df),
    "archived_files": archived_files
}
metadata_path = os.path.join(version_dir, "metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Wrote metadata to: {metadata_path}")
