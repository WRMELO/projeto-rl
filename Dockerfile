FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# Instala pacotes básicos e compatíveis com a base anterior
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
        ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copia e instala os pacotes Python existentes
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install notebook

# Comando padrão para iniciar o Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''"]
