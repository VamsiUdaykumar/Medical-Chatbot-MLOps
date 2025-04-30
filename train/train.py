from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import lightning as L
import torch
import litgpt
from litgpt import LLM
from lightning.pytorch.strategies import DeepSpeedStrategy, FSDPStrategy, DDPStrategy
from peft import get_peft_model, LoraConfig, TaskType
from litgpt.lora import GPT, merge_lora_weights

strategy_type = "ddp" #"deepspeed" or "fsdp" or "ddp"
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

if strategy_type == "deepspeed":
    strategy=DeepSpeedStrategy(
        stage=3,                 # Similar to FULL_SHARD
        offload_optimizer=True   # Enable CPU offloading of optimizer
    )
elif strategy_type == "fsdp":
    strategy=FSDPStrategy(sharding_strategy='FULL_SHARD')
elif strategy_type == "ddp":
    strategy = DDPStrategy()

# # Load 10% of the dataset for training and 2% for validation
# train_dataset = load_dataset("ruslanmv/ai-medical-dataset", split="train[:1%]")
# val_dataset = load_dataset("ruslanmv/ai-medical-dataset", split="train[1%:2%]")

# Load 1000 training samples
train_dataset = load_dataset("ruslanmv/ai-medical-dataset", split="train[:1000]")

# Load 500 validation samples (next 500 samples after the first 1000)
val_dataset = load_dataset("ruslanmv/ai-medical-dataset", split="train[1000:1500]")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

# Custom Dataset class
class MedicalQADataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item["question"]
        context = item["context"]
        prompt = f"Question: {question}\nAnswer:"
        input_text = f"{prompt} {context}"
        encoding = self.tokenizer(
            input_text,
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

# Create dataset instances
train_data = MedicalQADataset(train_dataset, tokenizer, max_length=512)
val_data = MedicalQADataset(val_dataset, tokenizer, max_length=512)

# # Define LoRA configuration
# lora_config = LoraConfig(
#     r=2,
#     lora_alpha=4,
#     target_modules=["q_proj", "v_proj"],
#     lora_dropout=0.1,
#     bias="none",
#     task_type=TaskType.CAUSAL_LM
# )

# DataModule
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

# class LitLLM(L.LightningModule):
#     def __init__(self, checkpoint_dir, tokenizer_dir=None, trainer_ckpt_path=None):
#         super().__init__()
 
#         self.llm = LLM.load(checkpoint_dir, tokenizer_dir=tokenizer_dir, distribute=None,\
#                             )
#         litgpt.lora.mark_only_lora_as_trainable(self.llm)
#         # llm = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_13b")
#         self.trainer_ckpt_path = trainer_ckpt_path

#     def setup(self, stage):
#         self.llm.trainer_setup(trainer_ckpt=self.trainer_ckpt_path)
        
#     def training_step(self, batch):
#         logits, loss = self.llm(input_ids=batch["input_ids"], target_ids=batch["labels"])
#         self.log("train_loss", loss, prog_bar=True)
#         if torch.cuda.is_available():
#             allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
#             reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # Convert to GB
#             print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
#         return loss

#     def validation_step(self, batch):
#         logits, loss = self.llm(input_ids=batch["input_ids"], target_ids=batch["labels"])
#         self.log("validation_loss", loss, prog_bar=True)
#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.llm.model.parameters(), lr=0.0002, weight_decay=0.0, betas=(0.9, 0.95))
#         return optimizer

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
        state_dict = torch.load(f"checkpoints/{model_name}/lit_model.pth", mmap=True)
        self.model.load_state_dict(state_dict, strict=False)

    def training_step(self, batch):
        input_ids, targets = batch["input_ids"], batch["labels"]
        logits = self.model(input_ids)
        loss = litgpt.utils.chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        warmup_steps = 10
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0002, weight_decay=0.0, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
        return [optimizer], [scheduler]

# Initialize DataModule
data_module = MedicalQADataModule(train_data, val_data, batch_size=32)

# Initialize model
lit_model = LitLLM()

# Trainer
trainer = L.Trainer(
    devices=1,
    strategy=strategy,
    accelerator="gpu",
    max_epochs=1,
    accumulate_grad_batches=4,
    precision="bf16-true",
    limit_val_batches=0,
    enable_checkpointing=False
)

# Train the model
trainer.fit(lit_model, data_module)

# Save final checkpoint
merge_lora_weights(lit_model.model)
trainer.save_checkpoint("checkpoints/finetuned.ckpt", weights_only=True)
