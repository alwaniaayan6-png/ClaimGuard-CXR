FROM python:3.10-slim

LABEL maintainer="Aayan Alwani <alwaniaayan6@gmail.com>"
LABEL description="ClaimGuard-CXR v5 — image-grounded radiology claim verification"
LABEL version="5.0.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Python dependencies — pinned for reproducibility
COPY v5/requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /workspace/requirements.txt

# Copy project source
COPY v5/ /workspace/v5/
COPY verifact/ /workspace/verifact/ 2>/dev/null || true

# Set PYTHONPATH so `from v5.xxx import yyy` resolves
ENV PYTHONPATH=/workspace

# Smoke test — import core modules (no network, no GPU needed)
RUN python -c "from v5.model import V5Config, build_v5_model; print('model OK')" && \
    python -c "from v5.data.claim_parser import rule_parse; print('parser OK')" && \
    python -c "from v5.data.claim_matcher import ClaimMatcher; print('matcher OK')" && \
    python -c "from v5.conformal.inverted_cfbh import inverted_cfbh; print('conformal OK')"

# Default command: run offline test suite
CMD ["python", "-m", "pytest", "v5/tests/", "-v", "--tb=short", "-m", "not slow"]
