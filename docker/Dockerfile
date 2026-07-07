# syntax=docker/dockerfile:1.7
# foamagent-mcp image: Foundation OpenFOAM v10 base + a slim, CPU-only Python
# environment for the key-free MCP server.
#
# Dependencies come from pyproject.toml (single manifest — no separate
# environment file). torch is installed CPU-only first so sentence-transformers
# never drags in the multi-GB CUDA wheel stack; the container does local
# embeddings on CPU and nothing else needs a GPU.
#
# Build (context must be the repo root, FAISS indices pulled via git-lfs):
#   docker build -f docker/Dockerfile -t foamagent:latest .
FROM openfoam/openfoam10-paraview510:latest
USER root
SHELL ["/bin/bash", "-c"]

# Runtime OS deps: git/git-lfs for the optional in-container self-update,
# xvfb for headless PyVista rendering.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        git-lfs \
        xvfb \
    && git lfs install \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# uv supplies both a standalone CPython 3.12 (the Ubuntu base only has 3.8)
# and fast dependency installs. Pin the uv tag when cutting a release.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Dependency layer: keyed on pyproject.toml only, so code edits never
# re-install packages. The cache mount keeps downloaded wheels across builds.
# `all`/`bedrock` extras are skipped: `all` is self-referential (breaks
# deps-only resolution) and `bedrock` is a subset of pipeline+hpc.
COPY pyproject.toml /tmp/deps/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.12 "$VIRTUAL_ENV" \
    && uv pip install "torch==2.5.1+cpu" \
        --extra-index-url https://download.pytorch.org/whl/cpu \
    && uv pip install -r /tmp/deps/pyproject.toml \
        --extra pipeline --extra hpc --extra ollama --extra tools

# Project code + baked-in FAISS database (see .dockerignore for exclusions)
ENV FoamAgent_PATH=/home/openfoam/Foam-Agent
COPY . $FoamAgent_PATH

# Interactive shells: OpenFOAM env + venv on PATH + land in the project
RUN echo '' >> /root/.bashrc \
    && echo '# Foam-Agent shell setup' >> /root/.bashrc \
    && echo 'source /opt/openfoam10/etc/bashrc 2>/dev/null || true' >> /root/.bashrc \
    && echo 'export PATH="/opt/venv/bin:$PATH"' >> /root/.bashrc \
    && echo 'cd "$FoamAgent_PATH" 2>/dev/null || true' >> /root/.bashrc

COPY docker/foamagent-entrypoint.sh /usr/local/bin/foamagent-entrypoint.sh
RUN chmod +x /usr/local/bin/foamagent-entrypoint.sh

WORKDIR /home/openfoam/

ENTRYPOINT ["/usr/local/bin/foamagent-entrypoint.sh"]
CMD ["/bin/bash"]
