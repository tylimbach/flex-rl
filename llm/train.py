import math
import time

import torch
from datasets import load_dataset
from torch.amp.grad_scaler import GradScaler
from torch.cuda.amp import autocast
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
)

# Configuration
MODEL_NAME = "gpt2-medium"
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
GRADIENT_CLIP_VALUE = 1.0
BATCH_SIZE = 2
NUM_EPOCHS = 1
MAX_TOKENS = 256
LR = 2e-5
USE_AMP = torch.cuda.is_available()
USE_PROFILER = False

EVAL_BEFORE_TRAINING = True
TRAIN = True
EVAL_AFTER_TRAINING = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# Load tokenizer and model
print(f"Loading model {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
tokenizer.pad_token = tokenizer.eos_token  # Required for GPT models with padding

# Load and tokenize dataset
print(f"Loading dataset {DATASET_NAME}...")
train_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train[:1%]")
val_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="validation[:1%]")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_TOKENS,
    )

print("Tokenizing dataset...")
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

# Setup optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=10,
    num_training_steps=len(train_loader) * NUM_EPOCHS,
)

scaler = GradScaler("cuda") if USE_AMP else None

def train():
    print(f"\nStarting training on {len(train_loader)} steps per epoch, total {len(train_loader) * NUM_EPOCHS} steps.")
    model.train()
    total_tokens = 0
    start_time = time.time()

    try:
        for epoch in range(NUM_EPOCHS):
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(DEVICE) for k, v in batch.items()}

                with autocast(enabled=USE_AMP):
                    outputs = model(**batch)
                    loss = outputs.loss

                if USE_AMP and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)  # Add this line to unscale before clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)  # Add gradient clipping
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)  # Add gradient clipping
                    optimizer.step()

                optimizer.zero_grad()
                lr_scheduler.step()

                total_tokens += batch["input_ids"].numel()

                if (step + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    tokens_per_sec = total_tokens / elapsed
                    print(f"Epoch {epoch+1} Step {step+1} | Loss: {loss.item():.4f} | Tokens/sec: {tokens_per_sec:.2f}")

                if USE_PROFILER and step == 20:
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
                        batch = {k: v.to(DEVICE) for k, v in next(iter(train_loader)).items()}
                        with autocast(enabled=USE_AMP):
                            model(**batch)
                        print("\n\n🔥 Top CUDA ops by time:")
                        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        elapsed = time.time() - start_time
        print(f"\n✅ Training complete in {elapsed:.2f} seconds")
        print(f"Avg tokens/sec: {total_tokens / elapsed:.2f}")
    except RuntimeError as e:
        print(f"❌ RuntimeError: {e}")
        if DEVICE.type == "cuda":
            print(torch.cuda.memory_summary(device=DEVICE, abbreviated=True))

def evaluate_model(model, val_loader):
    model.eval()
    total_loss = 0
    total_batches = 0
    skipped = 0
    eval_start = time.time()
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with autocast(enabled=USE_AMP):
                outputs = model(**batch)
                loss = outputs.loss
                if not torch.isnan(loss):
                    total_loss += loss.item()
                    total_batches += 1
                else:
                    skipped += 1
    elapsed = time.time() - eval_start
    print(f"Evaluation done in {elapsed: .2f} sec | Skipped {skipped} NaN batches | Completed {total_batches} batches")
    return total_loss / total_batches if total_batches > 0 else float('nan')

if EVAL_BEFORE_TRAINING:
    print("\n📏 Evaluating pretrained model...")
    baseline_loss = evaluate_model(model, val_loader)
    print(f"Pretrained Eval Loss: {baseline_loss:.4f} | Perplexity: {math.exp(baseline_loss):.2f}")

if TRAIN:
    train()

if EVAL_AFTER_TRAINING:
    print("\n📏 Evaluating fine-tuned model...")
    finetuned_loss = evaluate_model(model, val_loader)
    print(f"Fine-tuned Eval Loss: {finetuned_loss:.4f} | Perplexity: {math.exp(finetuned_loss):.2f}")
