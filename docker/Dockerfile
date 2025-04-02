FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /project

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    ca-certificates \
    software-properties-common \
    gnupg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    xvfb && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu jammy stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update && \
    apt-get install -y docker-ce-cli && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LO https://dl.k8s.io/release/v1.30.0/bin/linux/amd64/kubectl && \
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && \
    rm kubectl

RUN curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64 && \
    chmod +x ./kind && mv ./kind /usr/local/bin/kind

RUN ln -s /usr/bin/python3.10 /usr/bin/python

COPY docker/requirements.txt /tmp/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
	python -m pip install --upgrade pip==23.3.1 --no-cache-dir && \
	pip install --no-cache-dir -r /tmp/requirements.base.txt

CMD [ "bash" ]

FROM base AS perf
RUN apt-get update && apt-get install -y --no-install-recommends \
    nsight-systems-2024.6.2 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
