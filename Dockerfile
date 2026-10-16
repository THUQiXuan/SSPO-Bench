# Optional Dockerfile for running SSPO v1 in an isolated container (can be run with `--network none` after data/model download)
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_TOKEN=
ENV HUGGING_FACE_HUB_TOKEN=

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git curl wget ca-certificates util-linux build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/sspo_v1
COPY requirements-gpu.txt /workspace/sspo_v1/requirements-gpu.txt

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements-gpu.txt

COPY . /workspace/sspo_v1
RUN python3 -m pip install --no-deps -e .

CMD ["bash", "scripts/run_all.sh", "--help"]
