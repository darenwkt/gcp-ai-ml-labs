FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Gemini Enterprise Agent Platform Custom Container standard port
EXPOSE 8080

ENV AIP_HTTP_PORT=8080
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
