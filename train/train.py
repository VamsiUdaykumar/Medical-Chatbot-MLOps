from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import lightning as L
import torch
import litgpt
from litgpt import LLM
from litgpt.lora import GPT, merge_lora_weights
from lightning.pytorch.strategies import DeepSpeedStrategy, FSDPStrategy, DDPStrategy
from peft import get_peft_model, LoraConfig, TaskType
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
import mlflow.pytorch
import os

# --------------------------
# Config & MLflow Setup
# --------------------------
strategy_type = "ddp"  # Options: "deepspeed", "fsdp", "ddp"
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

mlflow.set_tracking_uri("http://192.5.87.181:8000/") 
mlflow.set_experiment("medical-qa-tinyllama")

mlflow_logger = MLFlowLogger(
    experiment_name="medical-qa-tinyllama"
)

# --------------------------
# Strategy Configuration
# --------------------------
if strategy_type == "deepspeed":
    strategy = DeepSpeedStrategy(stage=3, offload_optimizer=True)
elif strategy_type == "fsdp":
    strategy = FSDPStrategy(sharding_strategy='FULL_SHARD')
else:
    strategy = DDPStrategy()

# --------------------------
# Dataset Loading
# --------------------------
train_dataset = load_dataset("ruslanmv/ai-medical-dataset", split="train[:1000]")
val_dataset = load_dataset("ruslanmv/ai-medical-dataset", split="train[1000:1500]")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

class MedicalQADataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        prompt = f"Question: {item['question']}\nAnswer: {item['context']}"
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

train_data = MedicalQADataset(train_dataset, tokenizer)
val_data = MedicalQADataset(val_dataset, tokenizer)

class MedicalQADataModule(L.LightningDataModule):
    def __init__(self, train_data, val_data, batch_size=4):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

# --------------------------
# LightningModule with MLflow
# --------------------------
class LitLLM(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = GPT.from_name(
            name=model_name,
            lora_r=2,
            lora_alpha=4,
            lora_dropout=0.05,
            lora_query=True,
            lora_key=False,
            lora_value=True,
        )
        litgpt.lora.mark_only_lora_as_trainable(self.model)

    def on_train_start(self):
        state_path = f"checkpoints/{model_name}/lit_model.pth"
        if os.path.exists(state_path):
            state_dict = torch.load(state_path, mmap=True)
            self.model.load_state_dict(state_dict, strict=False)

        if self.global_rank == 0:
            mlflow.log_params({
                "model_name": model_name,
                "batch_size": self.trainer.datamodule.batch_size,
                "learning_rate": 0.0002,
                "lora_r": 2,
                "lora_alpha": 4,
                "lora_dropout": 0.05,
                "strategy": strategy_type
            })

    def training_step(self, batch, batch_idx):
        input_ids, targets = batch["input_ids"], batch["labels"]
        logits = self.model(input_ids)
        loss = litgpt.utils.chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        if self.global_rank == 0:
            mlflow.log_metric("train_loss_step", loss.item(), step=self.global_step)
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                mlflow.log_metric("gpu_mem_allocated", allocated, step=self.global_step)
                mlflow.log_metric("gpu_mem_reserved", reserved, step=self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, targets = batch["input_ids"], batch["labels"]
        logits = self.model(input_ids)
        loss = litgpt.utils.chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        if self.global_rank == 0:
            mlflow.log_metric("val_loss_epoch", loss.item(), step=self.current_epoch)
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                mlflow.log_metric("gpu_mem_allocated_val", allocated, step=self.current_epoch)
                mlflow.log_metric("gpu_mem_reserved_val", reserved, step=self.current_epoch)

        return loss

    def configure_optimizers(self):
        warmup_steps = 10
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0002, weight_decay=0.0, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, step / warmup_steps))
        return [optimizer], [scheduler]

# --------------------------
# Training
# --------------------------
data_module = MedicalQADataModule(train_data, val_data, batch_size=32)
lit_model = LitLLM()

trainer = L.Trainer(
    devices=1,
    strategy=strategy,
    accelerator="gpu",
    max_epochs=1,
    accumulate_grad_batches=4,
    precision="bf16-true",
    limit_val_batches=1.0,
    enable_checkpointing=False,
    # logger=mlflow_logger,
    log_every_n_steps=10
)

trainer.fit(lit_model, data_module)

# --------------------------
# Save and log final model
# --------------------------
merge_lora_weights(lit_model.model)
output_dir = "checkpoints/finetuned_model"
os.makedirs(output_dir, exist_ok=True)
torch.save(lit_model.model.state_dict(), os.path.join(output_dir, "model.pth"))

# Log final model to MLflow
if lit_model.global_rank == 0:
    print("Logging final model to MLflow...")
    mlflow.pytorch.log_model(lit_model.model, artifact_path="final_model")

mlflow.end_run()