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
# The `all` extra is skipped: it is self-referential (breaks deps-only
# resolution) — list the concrete extras instead.
COPY pyproject.toml /tmp/deps/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.12 "$VIRTUAL_ENV" \
    && uv pip install "torch==2.5.1+cpu" \
        --extra-index-url https://download.pytorch.org/whl/cpu \
    && uv pip install -r /tmp/deps/pyproject.toml \
        --extra ollama --extra tools

# Project code + baked-in FAISS database (see .dockerignore for exclusions).
# --chown hands the tree to the non-root 'openfoam' user (see USER switch below)
# without duplicating the layer, so the server can write runs/ and self-update.
ENV FoamAgent_PATH=/home/openfoam/Foam-Agent
COPY --chown=openfoam:openfoam . $FoamAgent_PATH

# Interactive shells: OpenFOAM env + venv on PATH + land in the project.
# Target the openfoam user's bashrc since that's the runtime user.
RUN echo '' >> /home/openfoam/.bashrc \
    && echo '# Foam-Agent shell setup' >> /home/openfoam/.bashrc \
    && echo 'source /opt/openfoam10/etc/bashrc 2>/dev/null || true' >> /home/openfoam/.bashrc \
    && echo 'export PATH="/opt/venv/bin:$PATH"' >> /home/openfoam/.bashrc \
    && echo 'cd "$FoamAgent_PATH" 2>/dev/null || true' >> /home/openfoam/.bashrc

COPY docker/foamagent-entrypoint.sh /usr/local/bin/foamagent-entrypoint.sh
RUN chmod +x /usr/local/bin/foamagent-entrypoint.sh

# The runtime 'openfoam' user needs a writable HOME cache: the HuggingFace
# embedding model downloads to $HOME/.cache/huggingface on the first retrieval,
# but earlier root build steps leave /home/openfoam/.cache owned by root, so that
# first write fails with EACCES. Hand the cache tree to the runtime user.
RUN mkdir -p /home/openfoam/.cache && chown -R openfoam:openfoam /home/openfoam/.cache

# Runtime user story. The MCP server and the solvers it launches must run as
# the non-root 'openfoam' user: OpenFOAM v10 refuses #codeStream / dynamic-code
# compilation when it detects the root user (isAdministrator() check in
# dynamicCode.C), so cases using codeStream — the cylinder tutorial,
# codedFixedValue BCs, etc. — fail as root (#3). But bind-mounted runs/ dirs
# can carry root-owned files left by pre-non-root images, which the openfoam
# user cannot write (#16). So the container starts as root and the entrypoint
# repairs runs/ ownership, then immediately drops to 'openfoam' via setpriv —
# every process after that first block is non-root. Side effect: a plain
# `docker exec` lands in a root shell; use `docker exec -u openfoam` to match
# the server's user.
USER root

WORKDIR /home/openfoam/

ENTRYPOINT ["/usr/local/bin/foamagent-entrypoint.sh"]
CMD ["/bin/bash"]
