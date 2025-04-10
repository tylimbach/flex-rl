import os
import sys
import logging
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from hydra import initialize, compose

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class PromptRequest(BaseModel):
	prompt: str


@asynccontextmanager
async def lifespan(app: FastAPI):
	log.info("Starting model load...")
	sys.stdout.flush()

	try:
		initialize(config_path="configs", version_base="1.3")
		cfg = compose(config_name="qwen-1.5")
		log.info(f"Loaded config: {cfg}")
		sys.stdout.flush()

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
		model = AutoModelForCausalLM.from_pretrained(
			cfg.model_id,
			device_map={"": device},
			attn_implementation=cfg.attn_implementation,
			torch_dtype=getattr(torch, cfg.torch_dtype)
		)

		app.state.model = model
		app.state.tokenizer = tokenizer
		app.state.device = device

		log.info("Model loaded.")
		sys.stdout.flush()

		yield

	except Exception as e:
		log.exception("Startup failed")
		sys.stdout.flush()
		raise e


app = FastAPI(lifespan=lifespan)


@app.post("/infer")
async def infer(request: PromptRequest):
	model = app.state.model
	tokenizer = app.state.tokenizer
	device = app.state.device

	inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
	outputs = model.generate(**inputs)
	result = tokenizer.decode(outputs[0], skip_special_tokens=True)

	return {"response": result}
