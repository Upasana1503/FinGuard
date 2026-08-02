FROM python:3.11-slim

WORKDIR /code

# Install CPU-only torch first (avoids pulling in ~2GB of unused CUDA libs --
# free hosting tiers here have no GPU anyway).
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Model + trained artifacts download/load on first startup via HF Hub cache;
# HF_HOME set to a writable dir some hosts (e.g. HF Spaces) require.
ENV HF_HOME=/code/.cache/huggingface
RUN mkdir -p /code/.cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
