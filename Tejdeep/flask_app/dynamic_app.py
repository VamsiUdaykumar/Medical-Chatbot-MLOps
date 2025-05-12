from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer
from litgpt.lora import GPT
from litgpt.prompts import PromptStyle
import torch
import threading
import queue
import time
import json
import os
from datetime import datetime

app = Flask(__name__)

# === Model Config ===
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
checkpoint_path = "/mnt/object/artifacts/medical-qa-model/model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# === Load model from checkpoint
print("🔄 Loading TinyLLaMA via Lit-GPT...")
model = GPT.from_name(model_name)
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

# === Batching setup
request_queue = queue.Queue()
BATCH_SIZE = 8
MAX_WAIT = 0.01

def batch_worker():
    while True:
        batch = []
        futures = []
        start_time = time.time()

        while len(batch) < BATCH_SIZE and (time.time() - start_time) < MAX_WAIT:
            try:
                item, fut = request_queue.get(timeout=MAX_WAIT)
                batch.append(item)
                futures.append(fut)
            except queue.Empty:
                break

        if batch:
            input_ids = torch.cat([item["input_ids"] for item in batch], dim=0).to(device)
            model.max_seq_length = input_ids.shape[-1]
            model.set_kv_cache(batch_size=len(batch))
            model.cos, model.sin = model.rope_cache(device=device)

            with torch.no_grad():
                outputs = model(input_ids)
                output_ids = torch.argmax(outputs, dim=-1)

            for i, fut in enumerate(futures):
                decoded = tokenizer.decode(output_ids[i], skip_special_tokens=True)
                answer = decoded.replace(batch[i]["prompt"], "").strip()
                fut["result"] = answer
                fut["event"].set()

threading.Thread(target=batch_worker, daemon=True).start()

# === Frontend route
@app.route('/')
def home():
    return render_template('index.html')

medical_context = """
Right middle lobe opacity with loss of the right heart border.
Left lower lobe opacity with loss of the left heart border.
No pleural effusion or pneumothorax.
"""

LOG_FILE = "/mnt/object/data/production/retraining_data_raw/online_log.json"

# === Inference endpoint
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')

    if not question:
        return jsonify({"error": "Question is required"}), 400

    prompt = f"""Medical Report:\n{medical_context}\n\nQuestion: {question}\nAnswer:"""
    styled_prompt = PromptStyle.from_name("alpaca").apply(prompt)
    input_ids = tokenizer(styled_prompt, return_tensors="pt").input_ids

    fut = {"event": threading.Event(), "result": None}
    request_queue.put(({"input_ids": input_ids, "prompt": styled_prompt}, fut))
    fut["event"].wait(timeout=10)

    answer = fut["result"] or "Timed out waiting for model response."

    # === Log to JSON ===
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "answer": answer
    }

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r+") as f:
            try:
                chat_history = json.load(f)
            except json.JSONDecodeError:
                chat_history = []
            chat_history.append(entry)
            f.seek(0)
            json.dump(chat_history, f, indent=4)
    else:
        with open(LOG_FILE, "w") as f:
            json.dump([entry], f, indent=4)

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
