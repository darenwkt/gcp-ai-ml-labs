import os
import torch
import shutil
from urllib.parse import urlparse
from google.cloud import storage
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# Load model globally on startup
model = None
tokenizer = None

def download_model_artifacts():
    aip_storage_uri = os.environ.get("AIP_STORAGE_URI")
    if not aip_storage_uri:
        raise ValueError("AIP_STORAGE_URI environment variable is not set!")
        
    local_dir = "/tmp/model"
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"Downloading model artifacts from GCS URI: {aip_storage_uri} to {local_dir}...")
    parsed = urlparse(aip_storage_uri)
    bucket_name = parsed.netloc
    prefix = parsed.path.lstrip("/")
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    for blob in blobs:
        relative_path = os.path.relpath(blob.name, prefix)
        local_file_path = os.path.join(local_dir, relative_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        blob.download_to_filename(local_file_path)
        print(f"Downloaded: {relative_path}")
        
    return local_dir

def init_model():
    global model, tokenizer
    try:
        model_path = download_model_artifacts()
    except Exception as e:
        print(f"Failed to load from GCS: {e}. Attempting local fallback...")
        model_path = os.environ.get("MODEL_PATH", "gpt2")
        
    print(f"Loading tokenizer and model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None
    )
    model.to(device)
    model.eval()
    print("Model initialized successfully and ready for serving.")

@app.route("/health", methods=["GET"])
def health():
    if model is not None and tokenizer is not None:
        return jsonify({"status": "healthy"}), 200
    return jsonify({"status": "initializing"}), 503

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "instances" not in data:
        return jsonify({"error": "Invalid request. Expected 'instances' in JSON payload"}), 400
        
    instances = data["instances"]
    predictions = []
    
    device = next(model.parameters()).device
    
    for instance in instances:
        prompt = instance.get("prompt", "")
        max_new_tokens = instance.get("max_new_tokens", 50)
        
        # Apply the same SFT template used during fine-tuning
        formatted_prompt = (
            "Below is an IT support ticket description. Write a response that resolves the ticket.\n\n"
            f"### Ticket:\n{prompt}\n\n"
            f"### Response:\n"
        )
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the response portion
        try:
            resolution = decoded.split("### Response:\n")[1].strip()
        except IndexError:
            resolution = decoded.replace(formatted_prompt, "").strip()
            
        predictions.append(resolution)
        
    return jsonify({"predictions": predictions}), 200

# Initialize model before startup
init_model()

if __name__ == "__main__":
    port = int(os.environ.get("AIP_HTTP_PORT", 8080))
    app.run(host="0.0.0.0", port=port)
