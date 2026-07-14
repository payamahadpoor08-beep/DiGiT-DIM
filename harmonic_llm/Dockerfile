# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

# System deps kept minimal; torch wheels bundle their own CUDA runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the package.
COPY . .
RUN pip install --no-cache-dir -e .

# Validate the install at build time: the tiny model must construct + forward.
RUN python -c "import harmonic_llm as hl, torch; \
    m=hl.build_model(hl.ModelConfig.tiny()); \
    o=m(torch.randint(0,256,(1,16)),start_pos=0); \
    print('build-time smoke test OK')"

ENTRYPOINT ["harmonic-llm"]
CMD ["demo"]
