import logging
import os

import hydra
import torch
from huggingface_hub import login
from omegaconf import DictConfig
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	Llama4ForConditionalGeneration,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_model(model_id: str, attn_impl: str, torch_dtype: str):
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	dtype = getattr(torch, torch_dtype)

	if "Llama-4" in model_id:
		model = Llama4ForConditionalGeneration.from_pretrained(
			model_id,
			attn_implementation=attn_impl,
			device_map="auto",
			torch_dtype=dtype,
		)
		return tokenizer, model
	else:
		model = AutoModelForCausalLM.from_pretrained(
			model_id,
			attn_implementation=attn_impl,
			device_map="auto",
			torch_dtype=dtype,
		)

	return tokenizer, model


def generate(prompt: str, tokenizer, model):
	messages = [{"role": "user", "content": prompt}]
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
	log.info(response)


@hydra.main(config_path="configs", config_name="qwen-1.5", version_base="1.3")
def main(cfg: DictConfig):
	tokenizer, model = load_model(
		model_id=cfg.model_id,
		attn_impl=cfg.attn_implementation,
		torch_dtype=cfg.torch_dtype,
	)
	log.info("Model loaded")
	generate(cfg.prompt, tokenizer, model)


if __name__ == "__main__":
	main()
