# tests/conftest.py
import json, torch, pytest
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "your-trained-model-name"          # <<<<<< replace

@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)

@pytest.fixture(scope="session")
def model():
    m = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    m.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return m

@pytest.fixture(scope="session")
def predict(model, tokenizer):
    def _predict(text, max_len=256):
        ids = tokenizer(text, return_tensors="pt").to(model.device)
        out = model.generate(ids["input_ids"], max_length=max_len, do_sample=False)
        return tokenizer.decode(out[0], skip_special_tokens=True).strip().lower()
    return _predict

@pytest.fixture(scope="session")
def paraphrase_sets():
    path = Path("templates/paraphrase_sets.json")
    with path.open() as f:
        return json.load(f)

from pathlib import Path, PurePath

@pytest.fixture(scope="session")
def slice_bleu():
    path = Path("templates/slice_bleu.json")
    if not path.exists():
        raise FileNotFoundError("Run scripts/build_slice_cache.py first")
    return json.loads(path.read_text())

import json, re
from pathlib import Path

@pytest.fixture(scope="session")
def failure_cases():
    path = Path("templates/failure_modes.json")
    return json.loads(path.read_text())