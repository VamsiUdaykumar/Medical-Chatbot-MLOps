import time
import json
import requests

# Configuration
DATA_PATH = "/mnt/object/data/dataset-split/evaluation/testing.json" 
INFERENCE_ENDPOINT = "http://localhost:5000/predict"  
SLEEP_INTERVAL = 5  # seconds between each simulated record

def simulate_online_questions():
    with open(DATA_PATH, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            try:
                entry = json.loads(line)
                question = {"question": entry["question"]}

                response = requests.post(INFERENCE_ENDPOINT, json=question)
                print(f"[{line_num}] Sent: {question}")
                print(f"Response: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"[{line_num}] Error processing: {e}")

            time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    simulate_online_questions()
