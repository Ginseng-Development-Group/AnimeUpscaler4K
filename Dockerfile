FROM nvidia/cuda:12.6.2-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    REALCUGAN_DIR=/opt/realcugan \
    REALCUGAN_BIN=/opt/realcugan/realcugan-ncnn-vulkan \
    PIPELINE_CONFIG=/config/pipeline.json

# Install FFmpeg, MKVToolNix, Vulkan runtime, and Python 3
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    mkvtoolnix \
    python3 \
    libvulkan1 \
    mesa-vulkan-drivers \
    vulkan-tools \
    ca-certificates \
    wget \
    unzip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Ensure NVIDIA Vulkan ICD is registered inside the container
RUN mkdir -p /etc/vulkan/icd.d && \
    printf '{\n  "file_format_version" : "1.0.0",\n  "ICD" : {\n    "library_path" : "libGLX_nvidia.so.0",\n    "api_version" : "1.3.194"\n  }\n}\n' > /etc/vulkan/icd.d/nvidia_icd.json

# Download and install Real-CUGAN ncnn Vulkan (includes models-pro, models-se, models-nose)
RUN wget -q https://github.com/nihui/realcugan-ncnn-vulkan/releases/download/20220728/realcugan-ncnn-vulkan-20220728-ubuntu.zip -O /tmp/realcugan.zip && \
    unzip -q /tmp/realcugan.zip -d /tmp && \
    mv /tmp/realcugan-ncnn-vulkan-20220728-ubuntu /opt/realcugan && \
    chmod +x /opt/realcugan/realcugan-ncnn-vulkan && \
    rm -f /tmp/realcugan.zip

WORKDIR /app
COPY src/ /app/src/
COPY config/ /app/config/

RUN chmod +x /app/src/*.py && \
    mkdir -p /input /output /workspace /config && \
    cp /app/config/pipeline.example.json /config/pipeline.json

ENTRYPOINT ["python3", "/app/src/pipeline_chunked.py"]
CMD ["--config", "/config/pipeline.json"]
