from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# Config
model_name = "gpt2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model/tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

# FastAPI setup
app = FastAPI(title="Medical Chatbot")

class PromptRequest(BaseModel):
    symptoms: str
    question: str

@app.post("/ask")
def ask(request: PromptRequest):
    if not request.symptoms or not request.question:
        raise HTTPException(status_code=400, detail="Both symptoms and question are required.")

    prompt = f"Symptoms: {request.symptoms}\nQuestion: {request.question}\nAnswer:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    start = time.time()
    output_ids = model.generate(
        input_ids,
        max_length=100,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )
    latency = time.time() - start

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {
        "answer": answer.replace(prompt, "").strip(),
        "latency_sec": latency,
        "device": device.type
    }
