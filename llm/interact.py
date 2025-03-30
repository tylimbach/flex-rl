import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model).cuda()

print("LLM loaded. Type instructions:")

while True:
	try:
		instruction = input("> ")
		prompt = f"Instruction: {instruction}\nReasoning:"
		inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
		output = model.generate(**inputs, max_length=128)
		text = tokenizer.decode(output[0], skip_special_tokens=True)
		print(text)
	except KeyboardInterrupt:
		print("\nExiting...")
		break
