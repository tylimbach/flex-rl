# LLM Reasoning Module

This folder contains the lightweight reasoning & instruction-following component
for the RL project.

## Structure

- `interact.py` → Query the LLM interactively
- `finetune.py` → Fine-tune reasoning skills on trajectory data
- `dataset.jsonl` → Instruction-reasoning samples
- `config.yaml` → Model & training settings

## Example

```bash
python llm/interact.py
> Walk forward
Instruction: Walk forward
Reasoning: To walk forward, lean torso forward and push with back foot.

python llm/finetune.py
```
