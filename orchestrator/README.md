# Kubernetes Orchestration for flex-rl

## Persistent Volume

Apply the following to provision shared storage:

```bash
kubectl apply -f orchestrator/k8s/pvc.yaml
```

## Submit Training Job

```bash
python orchestrator/scripts/submit_experiment.py \
  --exp_name humanoid_exp1 \
  --config config/humanoid_default.yaml
```

## Submit Sweep

```bash
python orchestrator/scripts/submit_sweep.py \
  --sweep humanoid_sweep.yaml
```
