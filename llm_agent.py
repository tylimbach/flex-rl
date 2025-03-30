from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()


def query_llm(goal, obs):
	prompt = f"""
	You are controlling a humanoid robot. Goal: {goal}
	Current observation: {obs}
	What is a good action strategy? Reply concisely.
	"""
	inputs = tokenizer(prompt, return_tensors="pt")
	with torch.no_grad():
		outputs = model.generate(**inputs, max_new_tokens=50)
	response = tokenizer.decode(outputs[0], skip_special_tokens=True)
	return response.split("\n")[-1]


if __name__ == "__main__":
	goal = "walk forward"
	obs = "[0.3, -0.1, 0.2, ...]"
	print(query_llm(goal, obs))
