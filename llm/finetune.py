from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import yaml

with open("llm/config.yaml") as f:
	config = yaml.safe_load(f)

ds = load_dataset("json", data_files="llm/dataset.jsonl")["train"]
split = ds.train_test_split(test_size=0.1, seed=42)
train_ds = split["train"]
val_ds = split["test"]

print(f"Loaded {len(train_ds)} training samples, {len(val_ds)} validation samples.")

tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
model = AutoModelForCausalLM.from_pretrained(config["model_name"]).cuda()

def preprocess(sample):
	prompt = f"Instruction: {sample['instruction']}\nReasoning:"
	sample["input_ids"] = tokenizer(prompt, truncation=True, max_length=config["max_seq_length"]).input_ids
	sample["labels"] = tokenizer(sample["reasoning"], truncation=True, max_length=config["max_seq_length"]).input_ids
	return sample

train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)

args = TrainingArguments(
	output_dir=config["output_dir"],
	per_device_train_batch_size=config["train_batch_size"],
	per_device_eval_batch_size=config["eval_batch_size"],
	num_train_epochs=config["num_train_epochs"],
	learning_rate=config["learning_rate"],
	warmup_steps=config["warmup_steps"],
	logging_steps=config["logging_steps"],
	save_steps=config["save_steps"],
	evaluation_strategy="epoch",
)

trainer = Trainer(
	model=model,
	args=args,
	train_dataset=train_ds,
	eval_dataset=val_ds,
)

trainer.train()
