import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset as TorchDataset, DataLoader
import lightning as L
import torch
import litgpt
from litgpt import LLM
from litgpt.lora import GPT, merge_lora_weights
from lightning.pytorch.strategies import DeepSpeedStrategy, FSDPStrategy, DDPStrategy
from pytorch_lightning.loggers.mlflow import MLFlowLogger
import mlflow
import mlflow.pytorch
import boto3
from botocore.client import Config
import json
import shutil
import ray
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig, FailureConfig
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment, prepare_trainer, RayTrainReportCallback
from ray import train
from time import time

floating_ip = os.getenv("FLOATING_IP", "129.114.25.221")
num_workers = 2

def train_func(config):
    print("Training with config:", config)
    mlflow_logger = MLFlowLogger(experiment_name="medical-qa-tinyllama", tracking_uri=f"http://{floating_ip}:8000")

    use_mixed_precision = True
    accumulate_grad_batches = 4

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    split_dir = os.getenv("DATA_SPLIT_ROOT", "/mnt/object/data/dataset-split")
    train_path = os.path.join(split_dir, "training", "training.json")
    val_path = os.path.join(split_dir, "validation", "validation.json")
    artifact_dir = os.getenv("ARTIFACT_PATH", "/mnt/object/artifacts")

    train_df = pd.read_json(train_path, lines=True)
    val_df = pd.read_json(val_path, lines=True)

    print(f"Loaded {len(train_df)} training samples and {len(val_df)} validation samples.")

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    class MedicalQADataset(TorchDataset):
        def __init__(self, dataset, tokenizer, max_length=512):
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.prompt_template = """### Context:
You are a helpful, accurate, and safety-aware medical assistant.

### Instruction:
Answer the following medical question with factual and reliable information.

### Question:
{question}

### Answer:
{answer}
"""

        def __len__(self): return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            prompt = self.prompt_template.format(question = item['question'], answer = item['answer'])
            encoding = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
            input_ids = encoding["input_ids"].squeeze()
            attention_mask = encoding["attention_mask"].squeeze()
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}

    class MedicalQADataModule(L.LightningDataModule):
        def __init__(self, train_data, val_data, batch_size=8):
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

    start_time = time()
    model = LitLLM()

    if use_mixed_precision:
        trainer = L.Trainer(
            max_epochs=config["epochs"],
            accelerator="auto",
            devices="auto",
            strategy=RayDDPStrategy(),
            precision="16-mixed",  # Enable mixed precision
            accumulate_grad_batches=accumulate_grad_batches,  # Gradient accumulation to simulate larger batch size
            plugins=[RayLightningEnvironment()],
            logger=mlflow_logger,
            log_every_n_steps=5,
            callbacks=[RayTrainReportCallback()],
        )
    else:
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

    if trainer.global_rank == 0:
        mlflow_logger.experiment.log_param(mlflow_logger.run_id, "model_name", config["model_name"])
        mlflow_logger.experiment.log_param(mlflow_logger.run_id, "learning_rate", config["lr"])
        mlflow_logger.experiment.log_param(mlflow_logger.run_id, "epochs", config["epochs"])
        mlflow_logger.experiment.log_param(mlflow_logger.run_id, "batch_size", 8)
        mlflow_logger.experiment.log_param(mlflow_logger.run_id, "use_mixed_precision", use_mixed_precision)
        mlflow_logger.experiment.log_param(mlflow_logger.run_id, "gradient_accumulation_steps", accumulate_grad_batches)
        mlflow_logger.experiment.log_param(mlflow_logger.run_id, "num_gpus", num_workers)

    ckpt = train.get_checkpoint()
    if ckpt:
        with ckpt.as_directory() as ckpt_dir:
            trainer.fit(model, data_module, ckpt_path=os.path.join(ckpt_dir, "checkpoint.ckpt"))
    else:
        trainer.fit(model, data_module)
    end_time = time()

    merge_lora_weights(model.model)
    # torch.save(model.model.state_dict(), "model.pth")
    # print(f"Model saved")

    if trainer.global_rank == 0:
        model_save_path = os.path.join(artifact_dir, "medical-qa-model")
        if os.path.exists(os.path.join(model_save_path, "optimal_model.pth")):
            os.remove(os.path.join(model_save_path, "optimal_model.pth"))
        os.makedirs(model_save_path, exist_ok=True)
        torch.save(model.model.state_dict(), os.path.join(model_save_path, "optimal_model.pth"))
        print(f"Model saved to {model_save_path}/optimal_model.pth")
        print(f"Time taken to train: {end_time - start_time} Seconds")
        mlflow_logger.experiment.log_param(mlflow_logger.run_id, "run_time", end_time - start_time)
        
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info("Connecting to Ray...")
    ray.init(address="auto")
    logging.info("Connected to Ray.")
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "model_name": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            "lr": 2e-5,
            "epochs": 2,
        },
        run_config=RunConfig(
            name="ray-medical-qa",
            storage_path="s3://mlflow-artifacts",
            checkpoint_config=CheckpointConfig(num_to_keep=1),
            failure_config=FailureConfig(max_failures=1)
        ),
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=True,
            resources_per_worker={"CPU": 8, "GPU": 1}
        )
    )
    logging.info("Starting Ray training job...")
    results = trainer.fit()
