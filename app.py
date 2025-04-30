from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from prometheus_flask_exporter import PrometheusMetrics


app = Flask(__name__)
metrics = PrometheusMetrics(app)


# Load GPT-2 model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "gpt2"

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    symptoms = data.get('symptoms', '')
    question = data.get('question', '')

    if not symptoms or not question:
        return jsonify({"error": "Both symptoms and question are required"}), 400

    # Format the prompt
    prompt = f"Symptoms: {symptoms}\nQuestion: {question}\nAnswer:"

    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate response
    output_ids = model.generate(
        input_ids,
        max_length=100,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return jsonify({"answer": answer.replace(prompt, '').strip()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
