import argparse
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import subprocess
import re


def sanitize_name(name):
    return re.sub(r"[^a-z0-9-]", "-", name.lower())


def render_template(template_path, output_path, context):
    template_dir = Path(template_path).parent
    template_name = Path(template_path).name
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template(template_name)
    rendered = template.render(**context)
    Path(output_path).write_text(rendered)


def submit_job(job_yaml):
    subprocess.run(["kubectl", "apply", "-f", job_yaml], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit K8s RL Experiment")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--template", default="orchestrator/k8s/jobs/train_job.yaml")
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--memory", type=str, default="2Gi")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    job_name = f"train-{sanitize_name(args.exp_name)}"
    context = {
        "job_name": job_name,
        "container_name": "train",
        "docker_image": "flex-rl-runtime:latest",
        "command": ["python", "train.py"],
        "args": ["--name", args.exp_name, "--config", args.config],
        "cpu": args.cpu,
        "memory": args.memory,
        "gpu": args.gpu,
    }

    output_path = f"/tmp/{args.exp_name}_job.yaml"
    render_template(args.template, output_path, context)
    submit_job(output_path)
    print(f"✅ Submitted {args.exp_name} to Kubernetes")
