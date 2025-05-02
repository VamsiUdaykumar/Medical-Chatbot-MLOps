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
# from lightning.pytorch.loggers import MLFlowLogger
from pytorch_lightning.loggers.mlflow import MLFlowLogger
import mlflow
import mlflow.pytorch
import os
import json

# 🔁 New Ray imports
import ray
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig, FailureConfig
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment, prepare_trainer, RayTrainReportCallback
from ray import train
from ray import tune
from ray.tune.schedulers import ASHAScheduler

floating_ip = os.getenv("FLOATING_IP", "")

# --------------------------
# Ray-compatible Train Function
# --------------------------
def train_func(config):
    print("Training with config:", config)

    import mlflow

    # Setup mlflow logging
    mlflow_logger = MLFlowLogger(experiment_name="medical-qa-tinyllama", tracking_uri=f"http://{floating_ip}:8000")

    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset("ruslanmv/ai-medical-dataset", split="train[:500]")
    val_dataset = load_dataset("ruslanmv/ai-medical-dataset", split="train[500:750]")

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
        def __init__(self, train_data, val_data, batch_size=config["batch_size"]):
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
            self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["lr"])
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, step / 10))
            return [optimizer], [scheduler]

    model = LitLLM()

    # ⚡ Ray-compatible trainer
    trainer = L.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",
        devices="auto",
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        logger=mlflow_logger,
        log_every_n_steps=5,
        callbacks=[RayTrainReportCallback()],
    )

    trainer = prepare_trainer(trainer)

    trainer.fit(model, data_module)

    # ✅ Save final model
    merge_lora_weights(model.model)
    torch.save(model.model.state_dict(), "model.pth")

# --------------------------
# Launch with Ray
# --------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info("Connecting to Ray...")
    ray.init(address="auto")
    logging.info("Connected to Ray.")

    config = {
            "model_name": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            "lr": tune.uniform(5e-6, 2e-4),
            "epochs": tune.choice([1, 2, 4]),
            "batch_size": tune.choice([4, 8])
        }

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        run_config=RunConfig(
            name="ray-medical-qa",
            storage_path="s3://ray"
        ),
        scaling_config=ScalingConfig(
            num_workers=1,
            use_gpu=True,
            resources_per_worker={"CPU": 8, "GPU": 1}
        ),
        train_loop_config=config
    )

    logging.info("Starting Ray training job...")
    ### New for Ray Tune
    def tune_asha(num_samples):
        scheduler = ASHAScheduler(max_t=3, grace_period=1, reduction_factor=2)
        tuner = tune.Tuner(
            trainer,
            param_space={"train_loop_config": config},
            tune_config=tune.TuneConfig(
                metric="val_loss",
                mode="min",
                num_samples=num_samples,
                scheduler=scheduler,
            ),
        )
        return tuner.fit()

    results = tune_asha(num_samples=5)

