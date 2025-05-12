import tritonclient.http as httpclient
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
prompt = "Symptoms: fever\nQuestion: What should I do?\nAnswer:"
input_ids = tokenizer.encode(prompt, return_tensors="np").astype(np.int64)

client = httpclient.InferenceServerClient(url="localhost:8000")

inputs = [httpclient.InferInput("input_ids", input_ids.shape, "INT64")]
inputs[0].set_data_from_numpy(input_ids)

outputs = [httpclient.InferRequestedOutput("logits")]

response = client.infer(model_name="gpt2_quantized", inputs=inputs, outputs=outputs)

logits = response.as_numpy("logits")
print(f"Logits shape: {logits.shape}")
