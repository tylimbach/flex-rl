FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    xvfb \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python
COPY rl/docker/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip==23.3.1 --no-cache-dir && \
    pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
CMD [ "bash" ]
