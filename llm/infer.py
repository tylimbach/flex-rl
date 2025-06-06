import logging
import os

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from huggingface_hub import login
from omegaconf import DictConfig
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	Llama4ForConditionalGeneration,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_model(model_id: str, attn_impl: str, torch_dtype: str, device):
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	dtype = getattr(torch, torch_dtype)

	if "Llama-4" in model_id:
		model = Llama4ForConditionalGeneration.from_pretrained(
			model_id,
			attn_implementation=attn_impl,
			device_map={"": device},
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


def setup_distributed(rank, world_size):
	os.environ["RANK"] = str(rank)
	os.environ["WORLD_SIZE"] = str(world_size)
	dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size)


def run(rank, world_size, cfg):
	try:
		setup_distributed(rank, world_size)
		torch.cuda.set_device(rank)
		device = torch.device(f"cuda:{rank}")
		tokenizer, model = load_model(cfg.model_id, cfg.attn_implementation, cfg.torch_dtype, device)
		log.info(f"Model loaded on rank {rank}")
		if rank == 0:
			generate(cfg.prompt, tokenizer, model)
	except Exception as e:
		log.exception(f"Rank {rank} failed: {e}")


@hydra.main(config_path="configs", config_name="qwen-1.5", version_base="1.3")
def main(cfg: DictConfig):
	os.environ.setdefault("MASTER_ADDR", "localhost")
	os.environ.setdefault("MASTER_PORT", "12355")

	world_size = torch.cuda.device_count()
	mp.spawn(run, args=(world_size, cfg), nprocs=world_size)


if __name__ == "__main__":
	main()
