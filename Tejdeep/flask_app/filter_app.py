# flask_app/app.py  (trimmed to the essentials)
# ------------------------------------------------------------
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from litgpt.lora import GPT
from litgpt.prompts import PromptStyle
import torch, os, json, time, csv
from datetime import datetime
import shutil

app = Flask(__name__)

# ---------- Chat model --------------------------------------------------
MODEL_NAME   = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
CKPT_PATH    = "/mnt/object/artifacts/medical-qa-model/model.pth"
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tok_chat = AutoTokenizer.from_pretrained(MODEL_NAME)
tok_chat.pad_token = tok_chat.eos_token
chat_model = GPT.from_name(MODEL_NAME)
chat_model.load_state_dict(torch.load(CKPT_PATH, map_location=device), strict=False)
chat_model.to(device).eval()

# ---------- Zero-shot classifier (BART-MNLI) ----------------------------
clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

QUESTION_LABELS = {
    "General medical knowledge or educational inquiry": "non_critical",
    "Personal medical advice, urgent care, or critical concern": "critical",
    "Promotes good and legal behavior": "ethical",
    "Promotes illegal or harmful actions": "unethical",
}
ANSWER_LABELS = {
    "Promotes good and legal behavior": "ethical",
    "Promotes illegal or harmful actions": "unethical",
}

# ---------- Files for logs ---------------------------------------------
LOG_FILE  = "/home/cc/workspace/online_log.json"                 # append-only JSONL
PERF_LOG  = "/mnt/object/data/production/retraining_data_raw/perf_log.csv"
OBJ_FILE = "/mnt/object/data/production/retraining_data_raw/"

# ---------- Helper ------------------------------------------------------
def append_jsonl(path: str, obj: dict):
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# =======================================================================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    req = request.json or {}
    question = req.get("question", "").strip()
    symptoms = req.get("symptoms", "").strip()

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # -------- FIRST FILTER: incoming question --------------------------
    q_pred = clf(
        question,
        candidate_labels=[
            "General medical knowledge or educational inquiry",
            "Personal medical advice, urgent care, or critical concern",
            "Promotes good and legal behavior",
            "Promotes illegal or harmful actions",
        ],
        multi_class=True,
    )
    q_tags = [QUESTION_LABELS[l] for l in q_pred["labels"][:2]]

    if "unethical" in q_tags:
        refusal = "⚠️ Sorry, I can’t help with that."
        return jsonify({"answer": refusal}), 200
    if "critical" in q_tags:
        refusal = (
            "🚑 This sounds urgent. Please seek immediate medical attention "
            "or call emergency services (e.g., 911)."
        )
        return jsonify({"answer": refusal}), 200

    # -------- GENERATE ANSWER -----------------------------------------
    prompt = (
        f"Medical Report:\n{symptoms}\n\nQuestion: {question}\nAnswer:"
        if symptoms else f"Question: {question}\nAnswer:"
    )
    styled = PromptStyle.from_name("alpaca").apply(prompt)

    ids = tok_chat(styled, return_tensors="pt").input_ids.to(device)
    chat_model.max_seq_length = ids.shape[-1]
    chat_model.set_kv_cache(batch_size=1)
    chat_model.cos, chat_model.sin = chat_model.rope_cache(device=device)

    t0 = time.time()
    with torch.no_grad():
        logits = chat_model(ids)
        out_ids = torch.argmax(logits, dim=-1)
    latency = time.time() - t0

    raw_answer = tok_chat.decode(out_ids[0], skip_special_tokens=True)
    answer = raw_answer.replace(styled, "").strip()

    # -------- SECOND FILTER: generated answer -------------------------
    a_pred = clf(
        answer,
        candidate_labels=[
            "Promotes good and legal behavior",
            "Promotes illegal or harmful actions",
        ],
        multi_class=False,
    )
    if ANSWER_LABELS[a_pred["labels"][0]] == "unethical":
        answer = "⚠️ Apologies, I’m unable to provide that information."

    # -------- Persist logs --------------------------------------------
    record = {
        "ts": datetime.utcnow().isoformat(),
        "question": question,
        "symptoms": symptoms,
        "answer": answer,
        "lat_ms": round(latency * 1000, 1),
        "q_tags": q_tags,
    }
    append_jsonl(LOG_FILE, record)

    shutil.copy(LOG_FILE, OBJ_FILE)

    with open(PERF_LOG, "a", newline="") as f:
        csv.writer(f).writerow([record["ts"], record["lat_ms"], ids.shape[-1], "flask"])
        

    return jsonify({"answer": answer})

# =======================================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
