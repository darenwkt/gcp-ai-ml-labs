FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./
COPY run_ddp.sh ./
RUN chmod +x run_ddp.sh

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "train.py"]
