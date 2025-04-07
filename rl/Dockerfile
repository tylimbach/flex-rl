FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /project

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY rl/requirements.txt /tmp/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
	python -m pip install --upgrade pip && \
	pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /project

CMD [ "bash" ]
