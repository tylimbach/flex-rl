import time
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    get_scheduler,
)
from datasets import load_dataset

# Configuration
MODEL_NAME = "distilgpt2"
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
BATCH_SIZE = 4
NUM_EPOCHS = 1
MAX_TOKENS = 1024  # truncate long sequences
LR = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
print(f"Loading model {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
tokenizer.pad_token = tokenizer.eos_token  # GPT-style models often need this

# Load and tokenize dataset
print(f"Loading dataset {DATASET_NAME}...")
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train[:2%]")  # small subset for testing

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="pt"
    )

print("Tokenizing dataset...")
tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
loader = DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)

# Setup optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=10,
    num_training_steps=len(loader) * NUM_EPOCHS,
)

# Training loop
print("Starting training loop...")
model.train()
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    total_tokens = 0
    start_time = time.time()

    for step, batch in enumerate(loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        total_tokens += batch["input_ids"].numel()

        if (step + 1) % 10 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed
            print(f"Epoch {epoch+1} Step {step+1} | Loss: {loss.item():.4f} | "
                  f"Tokens/sec: {tokens_per_sec:.2f}")

print("Training complete.")
