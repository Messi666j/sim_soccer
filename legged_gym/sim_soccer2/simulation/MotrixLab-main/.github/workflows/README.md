# GitHub Actions Workflows

This directory contains GitHub Actions workflows for automating CI/CD processes.

## Docker Image Build Workflow

### File: `docker-build.yml`

Automatically builds and pushes Docker images to Docker Hub when a new version tag is pushed.

#### Trigger Conditions

The workflow is triggered only when you push a git tag that matches the pattern `v*`:

```bash
git tag v0.1.0
git push origin v0.1.0
```

#### What It Does

1. **Extracts Version**: Reads the version from `pyproject.toml` (currently `0.1.0`)
2. **Builds Docker Image**: Uses the Dockerfile in `docker/Dockerfile`
3. **Pushes Multiple Tags**:
    - `motphys/motrixlab:0.1.0` (version from pyproject.toml)
    - `motphys/motrixlab:latest` (always points to the latest version)
    - `motphys/motrixlab:0.1` (major.minor version)

#### Required Secrets

You need to configure the following secrets in your GitHub repository settings:

1. **`DOCKER_USERNAME`**: Your Docker Hub username
2. **`DOCKER_PASSWORD`**: Your Docker Hub password or access token

To add secrets:

1. Go to your repository on GitHub
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add the secrets listed above

#### Usage Example

```bash
# 1. Update version in pyproject.toml if needed
# 2. Commit your changes
git add .
git commit -m "Release v0.1.0"

# 3. Create and push a version tag
git tag v0.1.0
git push origin v0.1.0

# 4. The workflow will automatically build and push the Docker image
# 5. Monitor the build at: https://github.com/Motphys/MotrixLab/actions
```

#### Built Image Tags

After the workflow completes, the following Docker images will be available:

```bash
# Pull the latest version
docker pull motphys/motrixlab:latest

# Pull a specific version
docker pull motphys/motrixlab:0.1.0

# Pull major.minor version
docker pull motphys/motrixlab:0.1
```

#### Workflow Features

-   ✅ **Optimized Caching**: Uses GitHub Actions cache to speed up builds
-   ✅ **Multi-tag Support**: Automatically tags with version, major.minor, and latest
-   ✅ **Version Extraction**: Automatically reads version from pyproject.toml
-   ✅ **Docker Layer Caching**: Uses UV cache mounts for faster dependency installation
-   ✅ **Tag-based Trigger**: Only builds on version tags, not on every commit

#### Docker Image Contents

The resulting Docker image includes:

-   Base: NVIDIA CUDA 12.8.1 Runtime + Ubuntu 24.04
-   UV package manager
-   MotrixLab with all dependencies
-   SKRL (both JAX and PyTorch backends)
-   TensorBoard
-   MotrixSim physics engine

#### Testing the Docker Image Locally

Before tagging a release, you can test the Docker build locally:

```bash
cd docker
docker build -t motphys/motrixlab:test .
docker run --gpus all motphys/motrixlab:test scripts/view.py --env cartpole
```

#### Troubleshooting

**Build fails with authentication error:**

-   Verify Docker Hub credentials are correctly set in GitHub secrets
-   Ensure your Docker Hub account has permission to push to the `motphys/motrixlab` repository

**Version extraction fails:**

-   Ensure `pyproject.toml` has a valid `version = "x.y.z"` line
-   Check the workflow logs for the exact extraction command output

**Tag not triggering the workflow:**

-   Ensure the tag starts with `v` (e.g., `v0.1.0`, not `0.1.0`)
-   Verify the tag was pushed to the correct branch: `git push origin v0.1.0`

#### See Also

-   [Docker Hub Repository](https://hub.docker.com/r/motphys/motrixlab)
-   [Container Deployment Documentation](../../docs/source/zh_CN/user_guide/getting_started/container_deployment.md)
-   [Dockerfile](../../docker/Dockerfile)
