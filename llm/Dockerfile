FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app
COPY llm/ .

RUN pip install --upgrade pip && pip install -r requirements.txt

# ENTRYPOINT ["python", "-m", "llm.infer"]

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
