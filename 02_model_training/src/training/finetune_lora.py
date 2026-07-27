import os
import argparse
import json
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
from google.cloud import storage
from urllib.parse import urlparse
try:
    import wandb
except ImportError:
    wandb = None

from models import GPT_CONFIG_124M, GPTModel

# --- LoRA Linear Layer Wrapper ---
class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=8, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        
        # Freeze base parameters
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        return self.linear(x) + self.scaling * (self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t())

def apply_lora_to_model(model, rank=8, lora_alpha=16):
    for block in model.trf_blocks:
        block.att.W_query = LoRALinear(block.att.W_query, rank=rank, lora_alpha=lora_alpha)
        block.att.W_key = LoRALinear(block.att.W_key, rank=rank, lora_alpha=lora_alpha)
        block.att.W_value = LoRALinear(block.att.W_value, rank=rank, lora_alpha=lora_alpha)
        block.att.out_proj = LoRALinear(block.att.out_proj, rank=rank, lora_alpha=lora_alpha)

def merge_lora_weights(model):
    print("Merging LoRA weights back into base model...")
    for block in model.trf_blocks:
        # Merge W_query
        wq = block.att.W_query
        wq.linear.weight.data += wq.scaling * (wq.lora_B @ wq.lora_A)
        block.att.W_query = wq.linear
        
        # Merge W_key
        wk = block.att.W_key
        wk.linear.weight.data += wk.scaling * (wk.lora_B @ wk.lora_A)
        block.att.W_key = wk.linear

        # Merge W_value
        wv = block.att.W_value
        wv.linear.weight.data += wv.scaling * (wv.lora_B @ wv.lora_A)
        block.att.W_value = wv.linear

        # Merge out_proj
        op = block.att.out_proj
        op.linear.weight.data += op.scaling * (op.lora_B @ op.lora_A)
        block.att.out_proj = op.linear

# --- Alpaca Instruction SFT Dataset ---
class AlpacaDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=512):
        print(f"Loading SFT dataset from {json_path}...")
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item["instruction"]
        user_input = item.get("input", "")
        output = item["output"]

        # Prompt formatting
        if user_input:
            prompt = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Input:\n{user_input}\n\n### Response:\n"
            )
        else:
            prompt = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Response:\n"
            )

        full_text = prompt + output + "<|endoftext|>"
        encoded_prompt = self.tokenizer.encode(prompt, allowed_special="all")
        encoded_full = self.tokenizer.encode(full_text, allowed_special="all")

        if len(encoded_full) > self.max_length:
            encoded_full = encoded_full[:self.max_length]

        input_ids = torch.tensor(encoded_full)
        labels = torch.tensor(encoded_full).clone()
        
        # Mask prompt tokens in loss
        prompt_len = min(len(encoded_prompt), len(encoded_full))
        labels[:prompt_len] = -100

        return input_ids, labels

def collate_fn(batch, pad_token_id):
    input_ids, labels = zip(*batch)
    padded_inputs = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return padded_inputs[:, :-1], padded_labels[:, 1:]

def format_duration(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"

def format_tokens(n):
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)

def download_from_huggingface(repo_id, filename, local_path):
    print(f"Attempting to download {filename} from Hugging Face dataset repo {repo_id}...")
    try:
        from huggingface_hub import hf_hub_download
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        import shutil
        shutil.copy(downloaded_path, local_path)
        print(f"Successfully downloaded via huggingface_hub to {local_path}")
        return True
    except Exception as e:
        print(f"huggingface_hub download failed: {e}. Falling back to direct HTTP request...")
        
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    if repo_id == "tatsu-lab/alpaca" and filename == "alpaca_data.json":
        url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    print(f"Downloading from {url} to {local_path}...")
    try:
        import urllib.request
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req) as response, open(local_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"Successfully downloaded via direct HTTP to {local_path}")
        return True
    except Exception as e:
        print(f"Direct HTTP download from Hugging Face failed: {e}")
        return False

# --- Main Fine-tuning Entrypoint ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-uri", type=str, required=True, help="GCS or local path to base model.pth")
    parser.add_argument("--alpaca-json-uri", type=str, required=True, help="GCS or local path to alpaca_data.json")
    parser.add_argument("--hf-dataset-repo", type=str, default="tatsu-lab/alpaca", help="Hugging Face dataset repository fallback")
    parser.add_argument("--hf-dataset-file", type=str, default="alpaca_data.json", help="Hugging Face dataset file path fallback")
    parser.add_argument("--output-model-uri", type=str, required=True, help="GCS or local path to output model.pth")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum training steps limit")
    parser.add_argument("--wandb-api-key", type=str, default="", help="Weights & Biases API Key")
    args = parser.parse_args()

    if wandb and args.wandb_api_key:
        print("Initializing Weights & Biases run for SFT fine-tuning...")
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.init(
            project="gpt2-sft-lora",
            config={
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "max_length": args.max_length,
            }
        )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning using device: {device}")

    # 1. Download base model.pth if in GCS
    local_base_model_path = "base_model.pth"
    if args.base_model_uri.startswith("gs://"):
        parsed = urlparse(args.base_model_uri)
        bucket_name = parsed.netloc
        blob_name = parsed.path.lstrip("/")
        print(f"Downloading base model weights from {args.base_model_uri}...")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_base_model_path)
    else:
        local_base_model_path = args.base_model_uri

    # 2. Download alpaca dataset if in GCS
    local_alpaca_path = "alpaca_data.json"
    if args.alpaca_json_uri.startswith("gs://"):
        parsed = urlparse(args.alpaca_json_uri)
        bucket_name = parsed.netloc
        blob_name = parsed.path.lstrip("/")
        print(f"Downloading Alpaca dataset from {args.alpaca_json_uri}...")
        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_alpaca_path)
            print("Dataset download complete.")
        except Exception as e:
            print(f"GCS download failed: {e}")
            success = download_from_huggingface(
                repo_id=args.hf_dataset_repo,
                filename=args.hf_dataset_file,
                local_path=local_alpaca_path
            )
            if not success:
                raise RuntimeError("CRITICAL: Failed to download dataset from both GCS and Hugging Face fallback.")
    else:
        local_alpaca_path = args.alpaca_json_uri

    # 3. Load Base Model and apply LoRA
    model = GPTModel(GPT_CONFIG_124M)
    print(f"Loading weights from {local_base_model_path}...")
    model.load_state_dict(torch.load(local_base_model_path, map_location="cpu"))
    apply_lora_to_model(model, rank=args.lora_rank, lora_alpha=args.lora_alpha)
    model.to(device)

    # Print trainable vs frozen parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Params: {total_params:,} | Trainable (LoRA) Params: {trainable_params:,} ({100 * trainable_params / total_params:.4f}%)")

    # 4. Load Dataset
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = AlpacaDataset(local_alpaca_path, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer.eot_token))

    # 5. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 6. Training Loop
    print("Beginning LoRA fine-tuning loop...")
    global_step = 0
    tokens_trained = 0
    start_time = time.time()
    step_start_time = time.time()
    step_times = []
    total_steps = args.max_steps if args.max_steps is not None else (args.epochs * len(dataloader))
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for step, (inputs, targets) in enumerate(dataloader, 1):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            
            # Compute cross entropy loss ignoring index -100
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)
            loss.backward()
            optimizer.step()
            
            # Track steps, time, tokens
            global_step += 1
            tokens_trained += inputs.numel()
            
            step_time = time.time() - step_start_time
            step_times.append(step_time)
            if len(step_times) > 100:
                step_times.pop(0)
            avg_step_time = sum(step_times) / len(step_times)
            
            # Reset step start time
            step_start_time = time.time()
            
            loss_val = loss.item()
            epoch_loss += loss_val
            
            if wandb and args.wandb_api_key:
                wandb.log({
                    "loss": loss_val,
                    "epoch": epoch,
                    "global_step": global_step,
                    "tokens": tokens_trained
                })

            if global_step % 50 == 0 or global_step == total_steps:
                try:
                    perplexity = math.exp(loss_val)
                except OverflowError:
                    perplexity = float("inf")
                
                # Elapsed & ETA
                elapsed = time.time() - start_time
                dur_str = format_duration(elapsed)
                
                steps_remaining = total_steps - global_step
                eta_seconds = steps_remaining * avg_step_time
                eta_str = format_duration(eta_seconds)
                tokens_str = format_tokens(tokens_trained)
                
                print(f"Step {global_step:06d}/{total_steps:06d} - Loss: {loss_val:.4f} | PPL: {perplexity:.2f} | Time/step: {avg_step_time:.3f}s | Tokens: {tokens_str} | Epoch: {epoch} ({step}/{len(dataloader)}) | Duration: {dur_str} | ETA: {eta_str}")
                
            if args.max_steps is not None and global_step >= args.max_steps:
                print(f"Reached max steps limit of {args.max_steps}. Stopping fine-tuning SFT loop.")
                break
        else:
            continue
        break

    if wandb and args.wandb_api_key:
        print("Finishing Weights & Biases SFT run...")
        wandb.finish()


    # 7. Merge weights and save
    merge_lora_weights(model)
    local_output_path = "model.pth"
    torch.save(model.state_dict(), local_output_path)
    print(f"Saved merged model weights to {local_output_path}")

    # 8. Upload to GCS if needed
    if args.output_model_uri.startswith("gs://"):
        parsed = urlparse(args.output_model_uri)
        bucket_name = parsed.netloc
        blob_name = parsed.path.lstrip("/")
        print(f"Uploading consolidated model to {args.output_model_uri}...")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_output_path)
        print("Upload complete.")

if __name__ == "__main__":
    main()
