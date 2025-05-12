import time
import json
import requests

# Configuration
DATA_PATH = "/mnt/object/data/dataset-split/evaluation/testing.json"
INFERENCE_ENDPOINT = "http://fastapi_server:8000/ask"  # working endpoint
SLEEP_INTERVAL = 5  # seconds between each simulated record

def simulate_online_requests():
    with open(DATA_PATH, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            try:
                entry = json.loads(line)
                payload = {
                    "symptoms": entry.get("question_focus", ""),  
                    "question": entry.get("question", "")
                }

                response = requests.post(INFERENCE_ENDPOINT, json=payload)
                print(f"[{line_num}] Sent: {payload}")
                print(f"Response: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"[{line_num}] Error processing line: {e}")

            time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    simulate_online_requests()