FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /project

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

COPY docker/runtime-requirements.txt /tmp/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
	python -m pip install --upgrade pip==23.3.1 --no-cache-dir && \
	pip install --no-cache-dir -r /tmp/requirements.base.txt

COPY . /project

CMD [ "bash" ]
