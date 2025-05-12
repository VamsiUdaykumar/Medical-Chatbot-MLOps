from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from prometheus_client import Histogram, Counter, make_asgi_app
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer
from litgpt.lora import GPT
from litgpt.prompts import PromptStyle
import torch
import os
import json
import time
import asyncio
import aiofiles
from typing import Optional

# ------------------- Config -------------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
CHECKPOINT_PATH = "/mnt/object/artifacts/medical-qa-model/model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- Load Model -------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = GPT.from_name(MODEL_NAME)
state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.eval()

# ------------------- Metrics -------------------
LATENCY = Histogram("chatbot_latency_seconds", "Request latency", ["endpoint", "slice"])
ERRORS = Counter("chatbot_errors_total", "Count of errors", ["endpoint", "slice"])

# ------------------- FastAPI Init -------------------
middleware = [
    Middleware(CORSMiddleware, allow_origins=["*"])
]
app = FastAPI(title="TinyLLaMA Medical Chatbot", middleware=middleware)
app.mount("/metrics", make_asgi_app())

# ------------------- Request Schema -------------------
class PromptRequest(BaseModel):
    question: str
    slice: Optional[str] = "unspecified"

# ------------------- Prompt Generator -------------------
medical_context = """
Right middle lobe opacity with loss of the right heart border.
Left lower lobe opacity with loss of the left heart border.
No pleural effusion or pneumothorax.
"""

def generate_answer(question: str) -> str:
    prompt = f"Medical Report:\n{medical_context}\n\nQuestion: {question}\nAnswer:"
    styled_prompt = PromptStyle.from_name("alpaca").apply(prompt)
    input_ids = tokenizer(styled_prompt, return_tensors="pt").input_ids.to(DEVICE)

    model.max_seq_length = input_ids.shape[-1]
    model.set_kv_cache(batch_size=1)
    model.cos, model.sin = model.rope_cache(device=DEVICE)

    with torch.no_grad():
        output = model(input_ids)
        output_ids = torch.argmax(output, dim=-1)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(styled_prompt, "").strip()

# ------------------- Async Logger -------------------
async def write_jsonl(path: str, obj: dict):
    async with aiofiles.open(path, "a") as f:
        await f.write(json.dumps(obj, ensure_ascii=False) + "\n")

LOG_PATH = "../../../online_log.jsonl"

# ------------------- Endpoint -------------------
@app.post("/api/v1/answer")
async def ask(req: PromptRequest, raw_request: Request):
    slice_name = req.slice or "unspecified"
    endpoint = "/answer"

    if not req.question:
        ERRORS.labels(endpoint, slice_name).inc()
        raise HTTPException(400, "question is required")

    t0 = time.perf_counter()

    try:
        answer = generate_answer(req.question)
    except Exception as e:
        ERRORS.labels(endpoint, slice_name).inc()
        raise HTTPException(500, str(e))
    finally:
        LATENCY.labels(endpoint, slice_name).observe(time.perf_counter() - t0)

    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": req.question,
        "slice": slice_name,
        "answer": answer,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1)
    }
    asyncio.create_task(write_jsonl(LOG_PATH, record))

    return {"answer": answer, "device": DEVICE.type}
