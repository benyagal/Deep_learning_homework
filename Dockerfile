# --- Dockerfile a projekthez ---

# Python alapimage
FROM python:3.9-slim

# Munkakönyvtár beállítása a konténeren belül
WORKDIR /app

# Build-eszközök (pl. gcc), Git és Git LFS telepítése
RUN apt-get update && \
    apt-get install -y build-essential git git-lfs --no-install-recommends && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Függőségek másolása és telepítése
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# spaCy modell letöltése és telepítése Git-ből
RUN git clone https://huggingface.co/huspacy/hu_core_news_md /app/hu_core_news_md

RUN cd /app/hu_core_news_md && git lfs pull

RUN pip install /app/hu_core_news_md/hu_core_news_md-3.8.1-py3-none-any.whl

# Projekt forráskódjának másolása
COPY src/ /app/src/

# A log és models könyvtárak létrehozása a konténeren belül, hogy a futás során írni lehessen beléjük
RUN mkdir -p /app/log /app/models

# Futtatási parancs
# A run.sh fogja vezérelni a végrehajtást
COPY run.sh .
RUN sed -i 's/\r$//' run.sh
RUN chmod +x run.sh

CMD ["./run.sh"]
