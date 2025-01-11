# --- Imagem Base ---
FROM python:3.12-slim

# --- Configuração do Ambiente ---
WORKDIR /app

# --- Instalação de Dependências (Otimização de Cache) ---
COPY requirements.txt .

# Instalamos as dependências. A flag --no-cache-dir reduz o tamanho da imagem.
RUN pip install --no-cache-dir -r requirements.txt

# --- Copiando o Código da Aplicação ---
COPY . .

# --- Expondo a Porta ---
EXPOSE 8000

# --- Comando de Execução ---
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
