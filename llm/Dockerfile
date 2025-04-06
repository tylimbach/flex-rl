FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app
COPY llm/ .
ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}

RUN pip install --upgrade pip && pip install -r requirements.txt \
	& python -c "import os; from huggingface_hub import login; login(token=os.environ.get('HUGGINGFACE_TOKEN'))"

ENTRYPOINT ["python", "-m", "llm.infer", "--config-name=local_mistral"]
