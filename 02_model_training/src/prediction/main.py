import os
import torch
import torch.nn as nn
import tiktoken
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from google.cloud import storage
from urllib.parse import urlparse

# Define the model config
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}

# --- GPT-2 Model Architecture (must match training code) ---

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1), persistent=False)

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"])
        self.gelu = GELU()
        self.fc2 = nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=True)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# --- Helper Generation Functions ---

def generate_text(model, idx, max_new_tokens, context_size, temperature=1.0, repetition_penalty=1.0):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        
        if repetition_penalty != 1.0:
            # Apply repetition penalty
            for token_id in set(idx[0].tolist()):
                val = logits[0, token_id].item()
                if val > 0:
                    logits[0, token_id] = val / repetition_penalty
                else:
                    logits[0, token_id] = val * repetition_penalty
                    
        if temperature > 0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# --- FastAPI Setup ---

app = FastAPI(title="GPT-2 Serving Service")

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None

def download_model():
    model_path = "model.pth"
    if os.path.exists(model_path):
        print(f"Model file {model_path} already exists locally.")
        return
        
    storage_uri = os.environ.get("AIP_STORAGE_URI")
    if not storage_uri:
        print("Warning: AIP_STORAGE_URI not set. Attempting to run with dummy model weights if model.pth is missing.")
        return
        
    print(f"Downloading model from {storage_uri}...")
    parsed = urlparse(storage_uri)
    bucket_name = parsed.netloc
    blob_prefix = parsed.path.lstrip("/")
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Locate model.pth in GCS bucket
    blob_path = os.path.join(blob_prefix, "model.pth") if not blob_prefix.endswith("model.pth") else blob_prefix
    blob = bucket.blob(blob_path)
    
    if not blob.exists():
        # Fallback to search prefix
        blobs = list(bucket.list_blobs(prefix=blob_prefix))
        for b in blobs:
            if b.name.endswith("model.pth"):
                blob = b
                break
                
    if not blob.exists():
        raise FileNotFoundError(f"Could not find model.pth in storage URI: {storage_uri}")
        
    print(f"Downloading blob {blob.name} from bucket {bucket_name} to local {model_path}")
    blob.download_to_filename(model_path)
    print("Download complete.")

@app.on_event("startup")
def startup_event():
    global model, tokenizer, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        download_model()
    except Exception as e:
        print(f"Error downloading model: {e}")
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Initialize and load model
    model = GPTModel(GPT_CONFIG_124M)
    model_path = "model.pth"
    if os.path.exists(model_path):
        print(f"Loading state dict from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Warning: No model.pth found. Predictions will use uninitialized weights!")
    model.to(device)
    model.eval()
    print("Model initialized and ready.")

class PredictionInstance(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 50
    temperature: Optional[float] = 1.0
    repetition_penalty: Optional[float] = 1.0


class PredictionRequest(BaseModel):
    instances: List[PredictionInstance]

class PredictionResponse(BaseModel):
    predictions: List[dict]

@app.get("/healthz")
@app.get("/")
def health():
    # Return 200 to signal healthy to Gemini Enterprise Agent Platform
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
        
    predictions = []
    for instance in request.instances:
        prompt = instance.prompt
        max_new_tokens = instance.max_new_tokens
        
        # Tokenize prompt
        encoded = tokenizer.encode(prompt)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
        
        temperature = instance.temperature
        repetition_penalty = instance.repetition_penalty
        
        # Generate text
        context_size = GPT_CONFIG_124M["context_length"]
        with torch.no_grad():
            token_ids = generate_text(
                model=model,
                idx=encoded_tensor,
                max_new_tokens=max_new_tokens,
                context_size=context_size,
                temperature=temperature,
                repetition_penalty=repetition_penalty
            )

        
        # Decode text
        decoded = tokenizer.decode(token_ids.squeeze(0).tolist())
        predictions.append({"generated_text": decoded})
        
    return PredictionResponse(predictions=predictions)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("AIP_HTTP_PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
