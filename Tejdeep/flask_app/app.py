

from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer
from litgpt.lora import GPT
from litgpt.prompts import PromptStyle
import torch
# from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# app = Flask(__name__)

# # Load GPT-2 model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_name = "gpt2"

# model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# @app.route('/')
# def home():
#     return render_template('index.html')

# medical_context = """
# Right middle lobe opacity with loss of the right heart border.
# Left lower lobe opacity with loss of the left heart border.
# No pleural effusion or pneumothorax.
# """

# @app.route('/ask', methods=['POST'])
# def ask():
#     data = request.json
#     question = data.get('question', '')

#     if not question:
#         return jsonify({"error": "Question is required"}), 400

#     # Add context from medical report
#     prompt = f"""Medical Report:\n{medical_context}\n\nQuestion: {question}\nAnswer:"""

#     input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

#     output_ids = model.generate(
#         input_ids,
#         max_length=150,
#         temperature=0.7,
#         top_p=0.9,
#         repetition_penalty=1.2
#     )

#     answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     return jsonify({"answer": answer.replace(prompt, '').strip()})


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)


app = Flask(__name__)

# === Model Config ===
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
checkpoint_path = "/mnt/object/artifacts/medical-qa-model/model.pth"
#checkpoint_path = "/Users/tejdeepchippa/Desktop/All/model-optim/checkpoints/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T/lit_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# === Load model from checkpoint
print("Loading TinyLLaMA via Lit-GPT...")
model = GPT.from_name(model_name)
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.to(device) 
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

medical_context = """
Right middle lobe opacity with loss of the right heart border.
Left lower lobe opacity with loss of the left heart border.
No pleural effusion or pneumothorax.
"""

import json
import os
from datetime import datetime

#LOG_FILE = "/mnt/object/data/production/retraining_data_raw/online_log.json"
LOG_FILE = "../online_log.json"

import time
import csv

PERF_LOG = "/mnt/object/data/production/retraining_data_raw/perf_log.csv"

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')

    if not question:
        return jsonify({"error": "Question is required"}), 400

    prompt = f"""Medical Report:\n{medical_context}\n\nQuestion: {question}\nAnswer:"""
    styled_prompt = PromptStyle.from_name("alpaca").apply(prompt)

    input_ids = tokenizer(styled_prompt, return_tensors="pt").input_ids.to(device)
    model.max_seq_length = input_ids.shape[-1]
    model.set_kv_cache(batch_size=1)
    model.cos, model.sin = model.rope_cache(device=device)

    # === Measure latency ===
    start_time = time.time()
    with torch.no_grad():
        output = model(input_ids)
        output_ids = torch.argmax(output, dim=-1)
    latency = time.time() - start_time

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = decoded.replace(styled_prompt, "").strip()

    # === Log to JSON ===
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "answer": answer
    }

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            chat_history = json.load(f)
    else:
        chat_history = []

    chat_history.append(entry)

    with open(LOG_FILE, "w") as f:
        json.dump(chat_history, f, indent=4)

    # === Log perf data to CSV ===
    with open(PERF_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            round(latency * 1000, 2),
            input_ids.shape[-1],
            "flask"
        ])

    return jsonify({"answer": answer})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
