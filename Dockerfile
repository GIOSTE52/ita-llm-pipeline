FROM python:3.12-sli

WORKDIR /app

# Installo le dipendenze di sistema necessarie
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
    # la riga sopra pulisce la cache di apt per ridurre la dimensione dell'immagine finale 

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models

# creo le cartelle per non riscontrare errori
RUN mkdir -p /app/output /app/logs 

# Le variabili di environment le manteniamo nel docker-compose

CMD ["python3", "src/main.py", "--config", "configs/default.conf"]