import os
import torch
import lightning as L
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from litgpt import LLM
from litgpt.lora import GPT, merge_lora_weights
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.strategies import DDPStrategy
import mlflow
import mlflow.pytorch

# 🔁 New Ray imports
import ray
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment, prepare_trainer
from ray.air import session

# --------------------------
# Ray-compatible Train Function
# --------------------------
def train_func(config):

    # Setup MLflow logging
    mlflow.set_tracking_uri("http://192.5.87.181:8000/")
    mlflow.set_experiment("medical-qa-tinyllama")

    mlflow_logger = MLFlowLogger(experiment_name="medical-qa-tinyllama")

    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("ruslanmv/ai-medical-dataset")
    train_dataset = dataset["train"].select(range(1000))
    val_dataset = dataset["train"].select(range(1000, 1500))

    class MedicalQADataset(Dataset):
        def __init__(self, dataset, tokenizer, max_length=512):
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self): return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            prompt = f"Question: {item['question']}\nAnswer: {item['context']}"
            encoding = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
            input_ids = encoding["input_ids"].squeeze()
            attention_mask = encoding["attention_mask"].squeeze()
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}

    class MedicalQADataModule(L.LightningDataModule):
        def __init__(self, train_data, val_data, batch_size=32):
            super().__init__()
            self.train_data = train_data
            self.val_data = val_data
            self.batch_size = batch_size

        def train_dataloader(self):
            return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

        def val_dataloader(self):
            return DataLoader(self.val_data, batch_size=self.batch_size)

    train_data = MedicalQADataset(train_dataset, tokenizer)
    val_data = MedicalQADataset(val_dataset, tokenizer)
    data_module = MedicalQADataModule(train_data, val_data)

    class LitLLM(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = GPT.from_name(
                name=config["model_name"],
                lora_r=2, lora_alpha=4, lora_dropout=0.05,
                lora_query=True, lora_key=False, lora_value=True,
            )
            litgpt.lora.mark_only_lora_as_trainable(self.model)

        def training_step(self, batch, batch_idx):
            logits = self.model(batch["input_ids"])
            loss = litgpt.utils.chunked_cross_entropy(logits[..., :-1, :], batch["labels"][..., 1:])
            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            return loss

        def validation_step(self, batch, batch_idx):
            logits = self.model(batch["input_ids"])
            loss = litgpt.utils.chunked_cross_entropy(logits[..., :-1, :], batch["labels"][..., 1:])
            self.log("val_loss", loss, prog_bar=True, on_epoch=True)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["lr"])
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, step / 10))
            return [optimizer], [scheduler]

    model = LitLLM()

    # ⚡ Ray-compatible trainer
    trainer = L.Trainer(
        max_epochs=config["epochs"],
        accelerator="gpu",
        devices="auto",
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        logger=mlflow_logger,
        enable_checkpointing=True,
        log_every_n_steps=10
    )

    trainer = prepare_trainer(trainer)

    # 👇 Optional fault-tolerant resume
    ckpt = session.get_checkpoint()
    if ckpt:
        with ckpt.as_directory() as dir:
            trainer.fit(model, data_module, ckpt_path=os.path.join(dir, "model.ckpt"))
    else:
        trainer.fit(model, data_module)

    # ✅ Save final model
    merge_lora_weights(model.model)
    torch.save(model.model.state_dict(), "model.pth")

# --------------------------
# Launch with Ray
# --------------------------
if __name__ == "__main__":
    ray.init(address="auto")

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "model_name": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            "lr": 2e-4,
            "epochs": 1,
        },
        run_config=RunConfig(
            name="ray-medical-qa",
            storage_path="s3://ray",  # Replace with MinIO or Ray cluster storage
            checkpoint_config=CheckpointConfig(num_to_keep=1)
        ),
        scaling_config=ScalingConfig(
            num_workers=2,
            use_gpu=True,
            resources_per_worker={"CPU": 8, "GPU": 1}
        )
    )

    results = trainer.fit()
    print("Training completed.")
    ray.shutdown()
    # Clean up
    mlflow.end_run()
    mlflow.pytorch.log_model(model.model, artifact_path="final_model")
    print("Final model logged to MLflow.")
