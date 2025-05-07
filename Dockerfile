# Dockerfile para ambiente Jupyter + Reinforcement Learning
# Inclui: yfinance, numpy 1.26, pandas, matplotlib, gym, stable-baselines3, ta

FROM python:3.11-slim

# Variáveis de ambiente para evitar interações e garantir UTF-8
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

# Instalar dependências de sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instalar pacotes Python específicos para RL e análise financeira
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas \
    matplotlib \
    yfinance \
    ta \
    jupyter \
    ipykernel \
    seaborn \
    scikit-learn \
    gym==0.26.2 \
    stable-baselines3[extra] \
    ale-py \
    # pybox2d \
    autorom[accept-rom-license]

# Criar diretório de trabalho
WORKDIR /workspace

# Expor a porta padrão do Jupyter
EXPOSE 8888

# Comando de inicialização
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
