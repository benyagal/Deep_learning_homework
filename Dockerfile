# --- Dockerfile a projekthez ---

# Python alapimage
FROM python:3.9-slim

# Munkakönyvtár beállítása a konténeren belül
WORKDIR /app

# Függőségek másolása és telepítése
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Git telepítése a spaCy modell letöltéséhez
RUN apt-get update && apt-get install -y git

# spaCy modell letöltése és telepítése Git-ből
RUN git clone https://huggingface.co/huspacy/hu_core_news_md /app/hu_core_news_md
RUN pip install /app/hu_core_news_md/

# Projekt forráskódjának másolása
COPY src/ /app/src/

# A log és models könyvtárak létrehozása a konténeren belül, hogy a futás során írni lehessen beléjük
RUN mkdir -p /app/log /app/models

# Futtatási parancs
# A run.sh fogja vezérelni a végrehajtást
COPY run.sh .
RUN chmod +x run.sh

CMD ["./run.sh"]
