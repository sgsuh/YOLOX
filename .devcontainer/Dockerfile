FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN apt update && \
    apt install -y sudo \
    libgl1-mesa-glx

RUN pip install --upgrade pyopenssl && \
    pip install megengine -f https://megengine.org.cn/whl/mge.html && \
    pip install loguru \
    pillow && \
    pip install --ignore-installed PyYAML==6.0.3 && \
    pip install openvino==2022.3.1 \
    numpy==1.21.0 \
    nebullvm==0.8.0 \
    jupytext==1.14.0 \
    mdit-py-plugins==0.3.0 \
    onnx-simplifier==0.4.10 \
    urllib3==1.26.20 \
    wandb==0.18.0 \
    protobuf==4.23.0 \
    packaging==24.0 \
    mlflow==2.14.0 \
    python-dotenv \
    thop \
    ninja \
    pycocotools==2.0.2 \
    onnx==1.14.0

RUN git clone -b v0.2.0 https://github.com/NVIDIA-AI-IOT/torch2trt /opt/torch2trt && \
    cd /opt/torch2trt && \
    python setup.py install

ENV SHELL=/bin/bash
SHELL ["/bin/bash", "-c"]