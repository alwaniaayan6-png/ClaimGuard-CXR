# ClaimGuard Path B — reproducible execution environment.
#
# Usage:
#   cd claimguard_nmi/
#   docker build -t claimguard-nmi:0.1.0a0 -f Dockerfile .
#   docker run --rm -v $(pwd)/..:/workspace claimguard-nmi:0.1.0a0 \
#       python -m pytest claimguard_nmi/tests/ -q

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    git ca-certificates curl build-essential \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /workspace/verifact

# Install deps first so layer caches between code edits.
COPY claimguard_nmi/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt

# Copy the package into the image.
COPY claimguard_nmi /workspace/verifact/claimguard_nmi

# Smoke-test import on build.
RUN python -c "import claimguard_nmi; print(claimguard_nmi.__version__)"

CMD ["python", "-m", "pytest", "claimguard_nmi/tests/", "-q"]
