import os
import logging

import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	Llama4ForConditionalGeneration,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Llama-4")
ATTN_IMPL = os.getenv("ATTN_IMPL", "flash_attention_2")
TORCH_DTYPE = getattr(torch, os.getenv("TORCH_DTYPE", "bfloat16"))

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if "Llama-4" in MODEL_ID:
	model = Llama4ForConditionalGeneration.from_pretrained(
		MODEL_ID,
		attn_implementation=ATTN_IMPL,
		device_map="auto",
		torch_dtype=TORCH_DTYPE,
	)
else:
	model = AutoModelForCausalLM.from_pretrained(
		MODEL_ID,
		attn_implementation=ATTN_IMPL,
		device_map="auto",
		torch_dtype=TORCH_DTYPE,
	)

log.info(f"Model {MODEL_ID} loaded")

app = FastAPI()

class PromptRequest(BaseModel):
	prompt: str

@app.post("/infer")
def infer(request: PromptRequest):
	messages = [{"role": "user", "content": request.prompt}]
	inputs = tokenizer.apply_chat_template(
		messages,
		add_generation_prompt=True,
		return_tensors="pt",
		return_dict=True,
	)
	inputs = {k: v.to(model.device) for k, v in inputs.items()}
	outputs = model.generate(
		**inputs,
		max_new_tokens=128,
		temperature=0.7,
		top_p=0.9,
		do_sample=True,
	)
	response = tokenizer.batch_decode(
		outputs[:, inputs["input_ids"].shape[-1]:],
		skip_special_tokens=True
	)[0]
	return {"response": response}
