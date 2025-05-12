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
from time import time

# 🔁 New Ray imports
import ray
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig, FailureConfig
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment, prepare_trainer, RayTrainReportCallback
from ray import train
from ray import tune
from ray.tune.schedulers import ASHAScheduler

floating_ip = os.getenv("FLOATING_IP", "")
num_workers = 2

# --------------------------
# Ray-compatible Train Function
# --------------------------
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

    train_df = pd.read_json(train_path, lines=True, nrows=100)
    val_df = pd.read_json(val_path, lines=True, nrows=50)

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
            strategy=DeepSpeedStrategy(),
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
            storage_path="s3://mlflow-artifacts"
        ),
        scaling_config=ScalingConfig(
            num_workers=2,
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

    results = tune_asha(num_samples=3)
