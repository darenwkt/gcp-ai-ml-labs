FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy serving script
COPY main.py ./

EXPOSE 8080
ENV AIP_HTTP_PORT=8080
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
