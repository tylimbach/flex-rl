import argparse
import subprocess
import yaml

def submit_sweep(sweep_path):
	with open(sweep_path) as f:
		sweep = yaml.safe_load(f)

	sweep_name = sweep.get("sweep_name", "sweep")
	experiments = sweep["experiments"]
	resources = sweep.get("resources", {})

	cpu = resources.get("cpu", 4)
	memory = resources.get("memory", "8Gi")
	gpu = resources.get("gpu", 0)

	for exp in experiments:
		exp_name = exp["name"]
		config = exp["config"]
		subprocess.run([
			"python", "orchestrator/scripts/submit_experiment.py",
			"--exp_name", exp_name,
			"--config", config,
			"--cpu", str(cpu),
			"--memory", memory,
			"--gpu", str(gpu)
		], check=True)
		print(f"🚀 Submitted {exp_name}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Submit hyperparameter sweep")
	parser.add_argument("--sweep", required=True)
	args = parser.parse_args()

	submit_sweep(args.sweep)
