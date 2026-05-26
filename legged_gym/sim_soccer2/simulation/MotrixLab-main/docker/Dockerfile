FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    USERNAME=root \
    HOMEDIR=/root \
    UV_LINK_MODE=copy \
    TZ=Asia/Shanghai

RUN apt-get update  \
    && apt-get install -y --no-install-recommends \
        curl ca-certificates tzdata \
    && ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime \
    && echo ${TZ} > /etc/timezone \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.14 /uv /uvx /bin/

WORKDIR ${HOMEDIR}/motrixlab

# =============================================================================
# Two-Stage Dependency Installation Strategy
# =============================================================================
#
# This Dockerfile uses a two-stage installation approach to optimize Docker layer
# caching and reduce rebuild times when source code changes.
#
# Stage 1 (below): Install heavy framework dependencies
#   - Installs large ML frameworks (SKRL, JAX, PyTorch, TensorFlow) (~3GB)
#   - Skips workspace members (motrix-envs, motrixsim) that depend on source code
#   - These dependencies rarely change, so this layer is heavily cached
#   - Mounts only configuration files (pyproject.toml, uv.lock), no source code
#
# Stage 2 (after source copy): Install workspace members and remaining dependencies
#   - Installs all workspace members (motrix-envs, motrix-rl, motrixsim)
#   - Source code changes only invalidate this stage, not Stage 1
#   - Significantly faster rebuilds during development
#
# Benefits:
#   - Source code changes don't require re-downloading 3GB+ of dependencies
#   - Dependencies are cached across builds using UV cache
# =============================================================================

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=motrix_envs/pyproject.toml,target=motrix_envs/pyproject.toml \
    --mount=type=bind,source=motrix_rl/pyproject.toml,target=motrix_rl/pyproject.toml \
    --mount=type=bind,source=uv.toml,target=uv.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-workspace --package motrix-rl --no-install-package motrix-envs --no-install-package motrixsim --extra skrl-jax --extra skrl-torch

COPY --chown=${USERNAME}:${USERNAME} . .

# Stage 2: Install workspace members with source code
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-packages --extra skrl-jax --extra skrl-torch

ENTRYPOINT [ "uv", "run" ]