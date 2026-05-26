# syntax=docker/dockerfile:1
# 基于 NVIDIA Isaac Lab 官方镜像（已内置 Isaac Sim + Isaac Lab + torch + CUDA）
FROM nvcr.nju.edu.cn/nvidia/isaac-lab:2.3.2

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y

WORKDIR /workspace/mos-sim

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    /isaac-sim/python.sh -m pip install \
    -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    --extra-index-url https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url https://pypi.mirrors.ustc.edu.cn/simple/ \
    --timeout 120 -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["/isaac-sim/python.sh", "-m", "gunicorn", \
     "simulation.isaac_sim.labWebView.source.labwebView.labWebView.wrapper:app", \
     "--bind=0.0.0.0:8000"]
