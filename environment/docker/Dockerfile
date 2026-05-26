# Canonical Docker image for FAIR benchmark runs on CUDA 12.4.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ARG DOCKER_USER=ubuntu
ARG USER_HOME=/workspace

ENV CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONUNBUFFERED=1 \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0 10.0+PTX"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install --no-install-recommends -y \
      git ninja-build libglib2.0-0 libegl1 libgl1 libgomp1 tmux htop vim sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    if id -u "$DOCKER_USER" >/dev/null 2>&1; then \
      mkdir -p "$USER_HOME" && usermod -d "$USER_HOME" -m "$DOCKER_USER"; \
    else \
      adduser --disabled-password --gecos '' --home "$USER_HOME" "$DOCKER_USER"; \
    fi && \
    adduser "$DOCKER_USER" sudo && \
    chown -R "$DOCKER_USER:$DOCKER_USER" "$USER_HOME"

WORKDIR /workspace/PixARMesh
COPY environment/requirements.lock.txt /tmp/requirements.lock.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.lock.txt

USER $DOCKER_USER
