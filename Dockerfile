# Stage 1: Base image
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8
ENV CC=gcc

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get install -y \
    gcc-14 g++-14 python3.12 python3.12-venv python3.12-dev \
    git wget build-essential cmake \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg && \
    ln -sf /usr/bin/gcc-14 /usr/bin/gcc && \
    ln -sf /usr/bin/g++-14 /usr/bin/g++ && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

RUN uv pip install comfy-cli pip setuptools wheel
RUN /usr/bin/yes | comfy --workspace /comfyui install --version 0.3.30 --cuda-version 12.6 --nvidia

WORKDIR /comfyui
ADD src/extra_model_paths.yaml ./
WORKDIR /

RUN uv pip install runpod requests websocket-client safetensors
RUN uv pip install xformers --index-url https://download.pytorch.org/whl/cu126

ADD src/start.sh handler.py test_input.json ./
RUN chmod +x /start.sh

COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /usr/local/bin/comfy-node-install

WORKDIR /comfyui

# ✅ All nodes required by workflow
RUN comfy node install --exit-on-fail comfyui_controlnet_aux@1.1.3 --mode remote
RUN comfy node install --exit-on-fail comfyui-rmbg@3.0.0
RUN comfy node install --exit-on-fail sdxl_prompt_styler

WORKDIR /
ENV PIP_NO_INPUT=1

COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode

CMD ["/start.sh"]

# Stage 2: Download models
FROM base AS downloader

WORKDIR /comfyui
RUN mkdir -p models/checkpoints/sdxl \
              models/controlnet/sdxl/controlnet-union-sdxl-1.0 \
              models/upscale_models
RUN mkdir -p models/sams \
              models/grounding-dino   

# SAM model (2.56GB)
RUN wget -q -O models/sams/sam_vit_h_4b8939.pth \
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# GroundingDINO model (938MB)
RUN wget -q -O models/grounding-dino/groundingdino_swinb_cogcoor.pth \
    "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth"

RUN wget -q -O models/grounding-dino/GroundingDINO_SwinB.cfg.py \
    "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py"              

# ✅ Upscaler
RUN wget -q -O models/upscale_models/RealESRGAN_x4plus.pth \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

# ✅ ControlNet
RUN wget -q -O models/controlnet/sdxl/controlnet-union-sdxl-1.0/diffusion_pytorch_model_promax.safetensors \
    "https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors"

# ✅ Juggernaut XL - included directly in image
RUN wget -q -O models/checkpoints/sdxl/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors \
    "https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"

# Stage 3: Final image
FROM base AS final
COPY --from=downloader /comfyui/models /comfyui/models




































































































# # Stage 1: Base image with common dependencies
# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# # Prevents prompts from packages asking for user input during installation
# ENV DEBIAN_FRONTEND=noninteractive
# # Prefer binary wheels over source distributions for faster pip installations
# ENV PIP_PREFER_BINARY=1
# # Ensures output from python is printed immediately to the terminal without buffering
# ENV PYTHONUNBUFFERED=1 
# # Speed up some cmake builds
# ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# # Install Python, git and other necessary tools
# RUN apt-get update && apt-get install -y \
#     python3.10 \
#     python3-pip \
#     git \
#     wget \
#     libgl1 \
#     && ln -sf /usr/bin/python3.10 /usr/bin/python \
#     && ln -sf /usr/bin/pip3 /usr/bin/pip

# # Clean up to reduce image size
# RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# # Install comfy-cli
# RUN pip install comfy-cli

# # Install ComfyUI
# RUN /usr/bin/yes | comfy --workspace /comfyui install --cuda-version 11.8 --nvidia --version 0.3.26

# # Change working directory to ComfyUI
# WORKDIR /comfyui

# # Install runpod
# RUN pip install runpod requests

# # Support for the network volume
# ADD src/extra_model_paths.yaml ./

# # Go back to the root
# WORKDIR /

# # Add scripts
# ADD src/start.sh src/restore_snapshot.sh src/rp_handler.py test_input.json ./
# RUN chmod +x /start.sh /restore_snapshot.sh

# # Optionally copy the snapshot file
# ADD *snapshot*.json /

# # Restore the snapshot to install custom nodes
# RUN /restore_snapshot.sh

# # Start container
# CMD ["/start.sh"]

# # Stage 2: Download models
# FROM base as downloader

# ARG HUGGINGFACE_ACCESS_TOKEN
# ARG MODEL_TYPE

# # Change working directory to ComfyUI
# WORKDIR /comfyui

# # Create necessary directories
# RUN mkdir -p models/checkpoints models/vae

# # Download checkpoints/vae/LoRA to include in image based on model type
# RUN if [ "$MODEL_TYPE" = "sdxl" ]; then \
#       wget -O models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors && \
#       wget -O models/vae/sdxl_vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors && \
#       wget -O models/vae/sdxl-vae-fp16-fix.safetensors https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors; \
#     elif [ "$MODEL_TYPE" = "sd3" ]; then \
#       wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/checkpoints/sd3_medium_incl_clips_t5xxlfp8.safetensors https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors; \
#     elif [ "$MODEL_TYPE" = "flux1-schnell" ]; then \
#       wget -O models/unet/flux1-schnell.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors && \
#       wget -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
#       wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
#       wget -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors; \
#     elif [ "$MODEL_TYPE" = "flux1-dev" ]; then \
#       wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/unet/flux1-dev.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors && \
#       wget -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
#       wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
#       wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors; \
#     fi

# # Stage 3: Final image
# FROM base as final

# # Copy models from stage 2 to the final image
# COPY --from=downloader /comfyui/models /comfyui/models

# # Start container
# CMD ["/start.sh"]
