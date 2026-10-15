FROM ubuntu:22.04

# Install Python and tools inside the image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a dedicated Python environment.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m pip install pip==24.3.1

# Install the selected CUDA build of PyTorch.
RUN python -m pip install torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies while preserving the PyTorch version.
COPY requirements-docker.txt /tmp/requirements-docker.txt

RUN python -m pip install \
    torch==2.5.1 \
    -r /tmp/requirements-docker.txt \
    && python -m pip check

# Record all installed Python package versions.
RUN python -m pip freeze > /opt/requirements-resolved.txt

# Match the container user's IDs to the server user.
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid ${USER_GID} researcher \
    && useradd --uid ${USER_UID} --gid ${USER_GID} \
       --create-home --shell /bin/bash researcher

ENV HOME=/home/researcher
USER researcher

WORKDIR /workspace/FinTexTS

CMD ["sleep", "infinity"]