# 使用基础镜像
FROM hpcaitech/pytorch-cuda:2.1.0-12.1.0

# 设置代理
ENV http_proxy=http://bj-rd-proxy.byted.org:3128
ENV https_proxy=http://bj-rd-proxy.byted.org:3128

# 安装依赖工具
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libaio-dev \
    cmake

# 创建 /workspace 目录
RUN mkdir -p /workspace

# 克隆 Open-Sora 仓库
RUN git clone https://github.com/hpcaitech/Open-Sora.git /workspace/Open-Sora

# 设置工作目录
WORKDIR /workspace/Open-Sora

# 安装 Python 依赖
RUN pip install --upgrade pip
RUN pip install packaging ninja
RUN pip install xformers --index-url https://download.pytorch.org/whl/cu121
RUN pip install flash-attn --no-build-isolation
RUN pip install git+https://github.com/fangqi-Zhu/TensorNVMe.git
RUN pip install git+https://github.com/hpcaitech/ColossalAI.git
RUN pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/fangqi-Zhu/apex.git

RUN pip install -v -e .

CMD ["tail", "-f", "/dev/null"]


pip install --upgrade pip
pip install packaging ninja
pip install xformers --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation
pip install git+https://github.com/fangqi-Zhu/TensorNVMe.git
pip install git+https://github.com/hpcaitech/ColossalAI.git
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/fangqi-Zhu/apex.git

packaging
ninja
tensorflow
imageio
xformers --index-url https://download.pytorch.org/whl/cu121
flash-attn --no-build-isolation
git+https://github.com/fangqi-Zhu/TensorNVMe.git
git+https://github.com/hpcaitech/ColossalAI.git
-v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/fangqi-Zhu/apex.git
-v -e --no-cache-dir git+https://github.com/hpcaitech/Open-Sora.git