# 🔧 Base compatível com GPU (RTX 4050), PyTorch e CUDA 12.1
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# 🧰 Sistema e utilitários essenciais
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        libffi-dev \
        libssl-dev \
        git \
        curl \
        vim \
        sudo \
        ca-certificates \
        unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 📦 Instalação de bibliotecas Python para Data Science e RL
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    yfinance \
    gym \
    tqdm \
    plotly \
    jupyterlab \
    notebook \
    ipykernel \
    nbformat \
    stable-baselines3

# ✍️ Garante o kernel visível no Jupyter
RUN python -m ipykernel install --user --name=rl_jupyter --display-name "Python (RL GPU)"

# 🔐 Permissões completas de escrita na pasta de trabalho
RUN chmod -R a+rwx /workspace

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
