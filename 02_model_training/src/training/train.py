import os
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from google.cloud import storage
from urllib.parse import urlparse
import time
import math
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import wandb
except ImportError:
    wandb = None

# --- Pre-Tokenized Offline Dataset ---

class GPTOfflineDataset(IterableDataset):
    def __init__(self, bin_path=None, stream_url=None, context_length=1024, rank=0, world_size=1, total_tokens=None, hf_dataset_repo=None, hf_dataset_subset=None):
        self.bin_path = bin_path
        self.stream_url = stream_url
        self.context_length = context_length
        self.rank = rank
        self.world_size = world_size
        self.total_tokens = total_tokens
        self.hf_dataset_repo = hf_dataset_repo
        self.hf_dataset_subset = hf_dataset_subset
        
        if self.hf_dataset_repo:
            print(f"[Rank {rank}] Initializing GPTOfflineDataset in Hugging Face Streaming mode: {hf_dataset_repo} ({hf_dataset_subset})...")
        elif self.stream_url:
            print(f"[Rank {rank}] Initializing GPTOfflineDataset in HTTP Streaming mode: {stream_url}...")
        else:
            print(f"[Rank {rank}] Initializing GPTOfflineDataset with binary file: {bin_path}...")

    def __iter__(self):
        import numpy as np
        
        if self.hf_dataset_repo:
            from datasets import load_dataset
            import tiktoken
            
            dataset = load_dataset(self.hf_dataset_repo, name=self.hf_dataset_subset, split="train", streaming=True)
            enc = tiktoken.get_encoding("gpt2")
            eot_token = 50256
            
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                worker_id = 0
                num_workers = 1
            else:
                worker_id = worker_info.id
                num_workers = worker_info.num_workers
                
            total_workers = self.world_size * num_workers
            global_worker_id = self.rank * num_workers + worker_id
            
            token_buffer = []
            doc_idx = 0
            for item in dataset:
                if (doc_idx % total_workers) != global_worker_id:
                    doc_idx += 1
                    continue
                doc_idx += 1
                
                text = item.get("text", "")
                tokens = enc.encode_ordinary(text)
                token_buffer.extend(tokens)
                token_buffer.append(eot_token)
                
                while len(token_buffer) >= self.context_length + 1:
                    chunk = token_buffer[:self.context_length + 1]
                    chunk_tensor = torch.tensor(chunk, dtype=torch.long)
                    yield chunk_tensor[:-1], chunk_tensor[1:]
                    token_buffer = token_buffer[self.context_length:]
                    
        elif self.stream_url:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                worker_id = 0
                num_workers = 1
            else:
                worker_id = worker_info.id
                num_workers = worker_info.num_workers

            total_blocks = (self.total_tokens - 1) // self.context_length
            blocks_per_rank = total_blocks // self.world_size
            rank_start_block = self.rank * blocks_per_rank
            
            blocks_per_worker = blocks_per_rank // num_workers
            start_block = rank_start_block + worker_id * blocks_per_worker
            end_block = start_block + blocks_per_worker

            worker_tokens = (end_block - start_block) * self.context_length + 1
            start_byte = start_block * self.context_length * 2
            end_byte = start_byte + worker_tokens * 2 - 1

            import urllib.request
            req = urllib.request.Request(
                self.stream_url, 
                headers={
                    'User-Agent': 'Mozilla/5.0',
                    'Range': f'bytes={start_byte}-{end_byte}'
                }
            )
            
            try:
                resp = urllib.request.urlopen(req)
                
                # Yield first block
                first_block_bytes = resp.read((self.context_length + 1) * 2)
                if len(first_block_bytes) < (self.context_length + 1) * 2:
                    return
                first_chunk = np.frombuffer(first_block_bytes, dtype=np.uint16)
                chunk_tensor = torch.tensor(first_chunk.astype(np.int64))
                yield chunk_tensor[:-1], chunk_tensor[1:]
                
                last_token = chunk_tensor[-1:]
                
                # Yield remaining blocks
                read_size = self.context_length * 2
                for _ in range(1, blocks_per_worker):
                    block_bytes = resp.read(read_size)
                    if len(block_bytes) < read_size:
                        break
                    np_chunk = np.frombuffer(block_bytes, dtype=np.uint16)
                    chunk_tensor = torch.tensor(np_chunk.astype(np.int64))
                    
                    full_tensor = torch.cat([last_token, chunk_tensor])
                    yield full_tensor[:-1], full_tensor[1:]
                    last_token = chunk_tensor[-1:]
                    
            except Exception as e:
                print(f"[Rank {self.rank} Worker {worker_id}] Error streaming from HTTP: {e}")
                raise e
        else:
            data = np.memmap(self.bin_path, dtype=np.uint16, mode='r')

            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                worker_id = 0
                num_workers = 1
            else:
                worker_id = worker_info.id
                num_workers = worker_info.num_workers

            total_blocks = (len(data) - 1) // self.context_length
            blocks_per_rank = total_blocks // self.world_size
            rank_start_block = self.rank * blocks_per_rank
            
            blocks_per_worker = blocks_per_rank // num_workers
            start_block = rank_start_block + worker_id * blocks_per_worker
            end_block = start_block + blocks_per_worker

            for block_idx in range(start_block, end_block):
                start_idx = block_idx * self.context_length
                end_idx = start_idx + self.context_length + 1

                chunk = data[start_idx:end_idx]
                chunk_tensor = torch.tensor(chunk.astype(np.int64))

                input_chunk = chunk_tensor[:-1]
                target_chunk = chunk_tensor[1:]
                yield input_chunk, target_chunk


# --- Model Architecture ---
from models import GPT_CONFIG_124M, GPTModel

# --- Helper Functions ---



def save_checkpoint(model, optimizer, output_uri, step, pp_coord=0, tp_coord=0):
    suffix = f"_pp{pp_coord}_tp{tp_coord}"
    local_model_path = os.path.join(output_uri, f"model{suffix}.pth") if not output_uri.startswith("gs://") else f"model{suffix}.pth"
    local_ckpt_path = os.path.join(output_uri, f"checkpoint{suffix}.pth") if not output_uri.startswith("gs://") else f"checkpoint{suffix}.pth"
    
    # Ensure local directory exists if not saving to GCS
    if not output_uri.startswith("gs://"):
        os.makedirs(output_uri, exist_ok=True)
        
    # Get uncompiled raw state_dict if model was wrapped via torch.compile or DDP
    raw_model = model.module if hasattr(model, "module") else model
    raw_model = raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model
    raw_state_dict = raw_model.state_dict()
    
    torch.save(raw_state_dict, local_model_path)
    print(f"Saved prediction-friendly model weights at step {step} for PP={pp_coord}, TP={tp_coord}")
    
    torch.save({
        "step": step,
        "model_state_dict": raw_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
    }, local_ckpt_path)
    print(f"Saved training-friendly checkpoint file at step {step} for PP={pp_coord}, TP={tp_coord}")
    
    if output_uri.startswith("gs://"):
        parsed = urlparse(output_uri)
        bucket_name = parsed.netloc
        blob_path = parsed.path.lstrip("/")
        
        if blob_path and not blob_path.endswith("/"):
            blob_path += "/"
            
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Upload prediction-friendly model weights
        dest_model_name = os.path.join(blob_path, f"model{suffix}.pth")
        print(f"Uploading model weights to gs://{bucket_name}/{dest_model_name}...")
        blob = bucket.blob(dest_model_name)
        blob.upload_from_filename(local_model_path)
        
        # Upload training-friendly full checkpoint
        dest_ckpt_name = os.path.join(blob_path, f"checkpoint{suffix}.pth")
        print(f"Uploading full checkpoint state to gs://{bucket_name}/{dest_ckpt_name}...")
        blob_ckpt = bucket.blob(dest_ckpt_name)
        blob_ckpt.upload_from_filename(local_ckpt_path)
        
        # Save a historical step backup (as a full checkpoint so we can resume from it)
        backup_name = os.path.join(blob_path, f"checkpoint{suffix}_step_{step}.pth")
        print(f"Saving historical checkpoint backup to gs://{bucket_name}/{backup_name}...")
        backup_blob = bucket.blob(backup_name)
        backup_blob.upload_from_filename(local_ckpt_path)
        print("Checkpoint uploads complete.")
        
        # Clean up local temporary files
        try:
            os.remove(local_model_path)
            os.remove(local_ckpt_path)
        except OSError:
            pass
    else:
        # Save historical step backup locally
        backup_path = os.path.join(output_uri, f"checkpoint{suffix}_step_{step}.pth")
        torch.save({
            "step": step,
            "model_state_dict": raw_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
        }, backup_path)
        print(f"Saved local historical checkpoint backup at {backup_path}")

def format_tokens(n):
    if n >= 1e9: return f"{n / 1e9:.2f}B"
    if n >= 1e6: return f"{n / 1e6:.2f}M"
    if n >= 1e3: return f"{n / 1e3:.1f}K"
    return str(n)

def train_step_3d(input_batch, target_batch, model, optimizer, pp_coord, pp_size, prev_rank, next_rank, tp_group, dp_group, device, batch_size, context_length, emb_dim, scaler=None):
    optimizer.zero_grad()
    
    # 1. Forward and Backward Passes
    if pp_size == 1:
        # Local training (Single Stage)
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), target_batch.flatten())
        loss_val = loss.item()
        
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
            
    else:
        # Pipeline Parallel training (Multi-Stage)
        if pp_coord == 0:
            # First Stage
            input_batch = input_batch.to(device)
            x = model(input_batch)
            dist.send(x, dst=next_rank)
            
            # Backward pass: receive gradients from Stage 1
            grad_x_out = torch.empty(batch_size, context_length, emb_dim, device=device)
            dist.recv(grad_x_out, src=next_rank)
            
            # Initialize scaler._scale if scaler is active
            if scaler is not None:
                scaler.scale(torch.tensor(0.0, device=device))
                
            torch.autograd.backward(tensors=[x], grad_tensors=[grad_x_out])
            loss_val = None
            
        elif pp_coord == pp_size - 1:
            # Last Stage
            x = torch.empty(batch_size, context_length, emb_dim, device=device)
            dist.recv(x, src=prev_rank)
            x.requires_grad_()
            
            logits = model(x)
            target_batch = target_batch.to(device)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), target_batch.flatten())
            loss_val = loss.item()
            
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            dist.send(x.grad, dst=prev_rank)
            
        else:
            # Intermediate Stages
            x = torch.empty(batch_size, context_length, emb_dim, device=device)
            dist.recv(x, src=prev_rank)
            x.requires_grad_()
            
            x_out = model(x)
            dist.send(x_out, dst=next_rank)
            
            # Backward pass: receive gradients from next stage
            grad_x_out = torch.empty(batch_size, context_length, emb_dim, device=device)
            dist.recv(grad_x_out, src=next_rank)
            
            # Initialize scaler._scale if scaler is active
            if scaler is not None:
                scaler.scale(torch.tensor(0.0, device=device))
                
            torch.autograd.backward(tensors=[x_out], grad_tensors=[grad_x_out])
            dist.send(x.grad, dst=prev_rank)
            loss_val = None

    if torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.synchronize()

    # 2. Gradient synchronization across Data Parallel (DP) group
    if dp_group is not None and dist.get_world_size(dp_group) > 1:
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=dp_group)
                param.grad /= dist.get_world_size(dp_group)

    # 3. Optimizer Step
    if scaler is not None:
        scaler.unscale_(optimizer)
        
        # Synchronize found_inf across all ranks to prevent PP/DP divergence
        optimizer_state = scaler._per_optimizer_states[id(optimizer)]
        found_inf = torch.tensor(
            0.0 if len(optimizer_state["found_inf_per_device"]) == 0 else
            sum(v.item() for v in optimizer_state["found_inf_per_device"].values()),
            device=device
        )
        if dist.is_initialized():
            dist.all_reduce(found_inf, op=dist.ReduceOp.MAX)
            
        # Write back synchronized found_inf to scaler state so it knows to skip step if needed
        for k in optimizer_state["found_inf_per_device"].keys():
            optimizer_state["found_inf_per_device"][k].fill_(found_inf.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
    return loss_val

def get_url_metadata(url):
    import urllib.request
    req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as resp:
            final_url = resp.geturl()
            content_length = int(resp.getheader('Content-Length'))
            return final_url, content_length
    except Exception as e:
        print(f"Error retrieving metadata for {url}: {e}")
        raise e


def consolidate_checkpoints(output_uri, tp_size, pp_size, cfg):
    print("Consolidating sharded checkpoints into single model.pth...")
    is_gcs = output_uri.startswith("gs://")
    bucket = None
    blob_path = ""
    
    if is_gcs:
        parsed = urlparse(output_uri)
        bucket_name = parsed.netloc
        blob_path = parsed.path.lstrip("/")
        if blob_path and not blob_path.endswith("/"):
            blob_path += "/"
        client = storage.Client()
        bucket = client.bucket(bucket_name)

    shards = {}
    for p in range(pp_size):
        shards[p] = {}
        for t in range(tp_size):
            fn = f"model_pp{p}_tp{t}.pth"
            if is_gcs:
                local_tmp_path = f"tmp_{fn}"
                blob = bucket.blob(os.path.join(blob_path, fn))
                blob.download_to_filename(local_tmp_path)
                shards[p][t] = torch.load(local_tmp_path, map_location="cpu")
                try:
                    os.remove(local_tmp_path)
                except OSError:
                    pass
            else:
                shards[p][t] = torch.load(os.path.join(output_uri, fn), map_location="cpu")
                
    stage_state_dicts = {}
    for p in range(pp_size):
        stage_sd = {}
        keys = list(shards[p][0].keys())
        for k in keys:
            is_col_parallel = any(suffix in k for suffix in [
                "W_query.weight", "W_query.bias",
                "W_key.weight", "W_key.bias",
                "W_value.weight", "W_value.bias",
                "fc1.weight", "fc1.bias"
            ])
            is_row_parallel = any(suffix in k for suffix in [
                "out_proj.weight", "fc2.weight"
            ])
            
            if is_col_parallel:
                stage_sd[k] = torch.cat([shards[p][t][k] for t in range(tp_size)], dim=0)
            elif is_row_parallel:
                stage_sd[k] = torch.cat([shards[p][t][k] for t in range(tp_size)], dim=1)
            else:
                stage_sd[k] = shards[p][0][k]
        stage_state_dicts[p] = stage_sd
        
    consolidated_sd = {}
    layers_per_stage = cfg["n_layers"] // pp_size
    for p in range(pp_size):
        for k, v in stage_state_dicts[p].items():
            if k.startswith("trf_blocks."):
                parts = k.split(".", 2)
                local_block_idx = int(parts[1])
                global_block_idx = p * layers_per_stage + local_block_idx
                new_key = f"trf_blocks.{global_block_idx}.{parts[2]}"
                consolidated_sd[new_key] = v
            else:
                consolidated_sd[k] = v
                
    local_consolidated_path = "model.pth" if is_gcs else os.path.join(output_uri, "model.pth")
    torch.save(consolidated_sd, local_consolidated_path)
    
    if is_gcs:
        dest_model_name = os.path.join(blob_path, "model.pth")
        print(f"Uploading consolidated model weights to gs://{bucket_name}/{dest_model_name}...")
        blob = bucket.blob(dest_model_name)
        blob.upload_from_filename(local_consolidated_path)
        try:
            os.remove(local_consolidated_path)
        except OSError:
            pass
            
    print("Checkpoint consolidation finished successfully!")

# --- Main pretraining loop ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-steps", type=int, default=1000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--model-output-uri", type=str, default=".", help="Local path or GCS URI (gs://...) to save outputs")
    parser.add_argument("--log-freq", type=int, default=10, help="Steps frequency to log loss")
    parser.add_argument("--save-freq", type=int, default=0, help="Steps frequency to save checkpoints (0 to disable)")
    parser.add_argument("--dataset-subset", type=str, default="sample-10BT", help="FineWeb-Edu subset name")
    parser.add_argument("--restore-from", type=str, default=None, help="Local path or GCS URI to checkpoint.pth to restore training")
    parser.add_argument("--shuffle-buffer", type=int, default=2000, help="Shuffle buffer size for streaming")
    parser.add_argument("--wandb-api-key", type=str, default=None, help="Weights & Biases API Key")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of dataloader worker processes")
    parser.add_argument("--dataset-bin", type=str, default="gs://<YOUR_GCS_BUCKET>/dataset/train.bin", help="GCS URI or local path to pre-tokenized train.bin")
    parser.add_argument("--hf-dataset-repo", type=str, default="HuggingFaceFW/fineweb-edu", help="Hugging Face dataset repository fallback")
    parser.add_argument("--hf-dataset-file", type=str, default="sample/10BT/000_00000.bin", help="Hugging Face dataset file path fallback")
    parser.add_argument("--checkpoint-activations", type=str, default="False", help="Enable activation checkpointing (True/False)")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor Parallelism size")
    parser.add_argument("--pp-size", type=int, default=1, help="Pipeline Parallelism size")
    args = parser.parse_args()
    
    args.checkpoint_activations = args.checkpoint_activations.lower() in ("true", "1", "yes")
    
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        
    output_uri = args.model_output_uri

    torch.manual_seed(123)
    
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        
        if torch.cuda.is_available():
            device = f"cuda:{ddp_local_rank}"
            torch.cuda.set_device(device)
            backend = "nccl"
        else:
            device = "cpu"
            backend = "gloo"
            
        import datetime as dt_module
        from torch.distributed import init_process_group
        init_process_group(backend=backend, timeout=dt_module.timedelta(seconds=3600))
        
        # 3D Grid Topology setup
        tp_size = args.tp_size
        pp_size = args.pp_size
        assert ddp_world_size % (tp_size * pp_size) == 0, f"World size ({ddp_world_size}) must be divisible by TP ({tp_size}) * PP ({pp_size})"
        dp_size = ddp_world_size // (tp_size * pp_size)
        
        dp_coord = ddp_rank // (pp_size * tp_size)
        pp_coord = (ddp_rank // tp_size) % pp_size
        tp_coord = ddp_rank % tp_size
        
        # Create TP Groups
        tp_group = None
        for d in range(dp_size):
            for p in range(pp_size):
                ranks = [d * (pp_size * tp_size) + p * tp_size + t for t in range(tp_size)]
                group = dist.new_group(ranks)
                if ddp_rank in ranks:
                    tp_group = group
                    
        # Create DP Groups
        dp_group = None
        for p in range(pp_size):
            for t in range(tp_size):
                ranks = [d * (pp_size * tp_size) + p * tp_size + t for d in range(dp_size)]
                group = dist.new_group(ranks)
                if ddp_rank in ranks:
                    dp_group = group
                    
        # Create PP coordinates
        prev_rank = None
        next_rank = None
        if pp_coord > 0:
            prev_rank = dp_coord * (pp_size * tp_size) + (pp_coord - 1) * tp_size + tp_coord
        if pp_coord < pp_size - 1:
            next_rank = dp_coord * (pp_size * tp_size) + (pp_coord + 1) * tp_size + tp_coord
            
        master_process = ddp_rank == 0
        loss_master = (pp_coord == pp_size - 1) and (tp_coord == 0) and (dp_coord == 0)
        
        print(f"[Rank {ddp_rank}] Coordinate: DP={dp_coord}, PP={pp_coord}, TP={tp_coord} | Prev Rank={prev_rank}, Next Rank={next_rank}")
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        tp_size = 1
        pp_size = 1
        dp_size = 1
        dp_coord = 0
        pp_coord = 0
        tp_coord = 0
        tp_group = None
        dp_group = None
        prev_rank = None
        next_rank = None
        master_process = True
        loss_master = True
        device = torch.device(
            "cuda" if torch.cuda.is_available() else 
            "mps" if torch.backends.mps.is_available() else 
            "cpu"
        )
        
    if master_process:
        print(f"Using device: {device}")
        if ddp:
            print(f"3D Parallelism Grid Topology: DP_SIZE={dp_size}, PP_SIZE={pp_size}, TP_SIZE={tp_size}")
            print(f"Effective batch size: {args.batch_size * dp_size} (batch size per GPU: {args.batch_size}, total GPUs: {ddp_world_size})")

    is_cuda_device = (
        (hasinstance := isinstance(device, torch.device)) and device.type == "cuda"
    ) or (not hasinstance and "cuda" in str(device))

    if is_cuda_device:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if master_process:
        print("-" * 50)
        print("SYSTEM HARDWARE SPECIFICATIONS:")
        print(f"CPUs available: {os.cpu_count()}")
        if is_cuda_device:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"GPUs available: {gpu_count} x {gpu_name} ({total_mem:.2f} GB VRAM)")
        print("-" * 50)

    if is_cuda_device:
        gpu_name = torch.cuda.get_device_name(0)
        hw_suffix = "T4" if "T4" in gpu_name else "L4" if "L4" in gpu_name else "A100" if "A100" in gpu_name else "GPU"
    else:
        hw_suffix = "CPU"

    # Initialize wandb only on loss_master (the aggregator rank of the last stage)
    if loss_master and os.environ.get("WANDB_API_KEY") and wandb is not None:
        print("Initializing Weights & Biases (wandb) logging...")
        run_name = f"gpt2-{hw_suffix}-3d-dp{dp_size}-pp{pp_size}-tp{tp_size}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project="gpt2-pretraining",
            name=run_name,
            config={
                "learning_rate": args.learning_rate,
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
                "dp_size": dp_size,
                "pp_size": pp_size,
                "tp_size": tp_size,
                "effective_batch_size": args.batch_size * dp_size,
                "weight_decay": args.weight_decay,
                "dataset_subset": args.dataset_subset,
                "device": str(device)
            },
            notes="3D Parallel pretraining run"
        )



    # Determine mixed-precision dtype: prefer bfloat16 on Ampere+ (capability >= 8.0)
    device_dtype = torch.float16
    if is_cuda_device:
        major, minor = torch.cuda.get_device_capability(0)
        device_dtype = torch.bfloat16 if major >= 8 else torch.float16
        if loss_master:
            print(f"CUDA Compute Capability {major}.{minor} detected. Using mixed-precision dtype: {device_dtype}")

    model = GPTModel(
        GPT_CONFIG_124M, 
        parallel=True,
        checkpoint_activations=args.checkpoint_activations,
        pp_rank=pp_coord,
        pp_size=pp_size,
        tp_group=tp_group
    )
    model.to(device)
    use_fused = is_cuda_device
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay,
        fused=use_fused
    )
    
    start_step = 1
    
    # Broadcast weights across Data Parallel (DP) replicas
    if ddp and dp_group is not None and dist.get_world_size(dp_group) > 1:
        dp_master_rank = pp_coord * tp_size + tp_coord
        for param in model.parameters():
            dist.broadcast(param.data, src=dp_master_rank, group=dp_group)

    # Compilation disabled per user preference (eager mode execution)
    if loss_master:
        print("Running in Eager Mode (torch.compile disabled)...")
        
    scaler = torch.cuda.amp.GradScaler() if (is_cuda_device and device_dtype == torch.float16) else None

    # Compute parameter count of local model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Rank {ddp_rank}] Local shard parameter size: {num_params:,}")
    
    full_params = 163000000
    flops_per_step = 6 * full_params * (args.batch_size * dp_size) * GPT_CONFIG_124M["context_length"]
    tflops_per_step = flops_per_step / 1e12

    # Download dataset binary
    dataset_bin_uri = args.dataset_bin
    use_streaming = False
    stream_url = None
    total_tokens = None
    
    if dataset_bin_uri.startswith("gs://"):
        parsed = urlparse(dataset_bin_uri)
        bucket_name = parsed.netloc
        blob_name = parsed.path.lstrip("/")
        
        local_bin_path = "/tmp/train.bin"
        os.makedirs(os.path.dirname(local_bin_path), exist_ok=True)
        
        if ddp:
            from torch.distributed import barrier
        local_master = (ddp_local_rank == 0) if ddp else True
        
        gcs_success = False
        if local_master:
            if not os.path.exists(local_bin_path):
                print(f"Downloading pre-tokenized dataset from {dataset_bin_uri} to {local_bin_path}...")
                try:
                    client = storage.Client()
                    bucket = client.bucket(bucket_name)
                    blob = bucket.blob(blob_name)
                    blob.download_to_filename(local_bin_path)
                    print("Dataset download complete.")
                    gcs_success = True
                except Exception as e:
                    print(f"GCS download failed: {e}. Falling back to Hugging Face streaming fallback...")
            else:
                print(f"Using cached local dataset at {local_bin_path}.")
                gcs_success = True
                
        if ddp:
            gcs_success_tensor = torch.tensor(1.0 if gcs_success else 0.0, device=device)
            dist.broadcast(gcs_success_tensor, src=0)
            gcs_success = gcs_success_tensor.item() > 0.5
            dist.barrier()
            
        if not gcs_success:
            use_streaming = True
            print(f"[Rank {ddp_rank}] GCS dataset unavailable. Initializing Hugging Face parquet streaming fallback for: {args.hf_dataset_repo} ({args.dataset_subset})...")
    else:
        local_bin_path = dataset_bin_uri

    if use_streaming:
        dataset = GPTOfflineDataset(
            hf_dataset_repo=args.hf_dataset_repo,
            hf_dataset_subset=args.dataset_subset,
            context_length=GPT_CONFIG_124M["context_length"],
            rank=dp_coord,
            world_size=dp_size
        )
    else:
        dataset = GPTOfflineDataset(
            bin_path=local_bin_path,
            context_length=GPT_CONFIG_124M["context_length"],
            rank=dp_coord,
            world_size=dp_size
        )
        
    pin_memory = is_cuda_device
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory)
    data_iter = iter(dataloader)
    
    import numpy as np
    if use_streaming:
        total_epoch_tokens = total_tokens
    else:
        temp_data = np.memmap(local_bin_path, dtype=np.uint16, mode='r')
        total_epoch_tokens = len(temp_data)
        del temp_data

    total_run_tokens = args.max_steps * (args.batch_size * dp_size) * GPT_CONFIG_124M["context_length"]
    run_epoch_percent = (total_run_tokens / total_epoch_tokens) * 100
    
    if loss_master:
        print(f"Beginning training loop for {args.max_steps} steps (total tokens: {format_tokens(total_run_tokens)}, approx {run_epoch_percent:.5f}% of an epoch)...")
    
    model.train()
    start_time = time.time()
    last_log_time = start_time
    
    if is_cuda_device:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=device_dtype)
    else:
        autocast_ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)

    for step in range(start_step, args.max_steps + 1):
        try:
            input_batch, target_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            input_batch, target_batch = next(data_iter)

        with autocast_ctx:
            loss_val = train_step_3d(
                input_batch=input_batch,
                target_batch=target_batch,
                model=model,
                optimizer=optimizer,
                pp_coord=pp_coord,
                pp_size=pp_size,
                prev_rank=prev_rank,
                next_rank=next_rank,
                tp_group=tp_group,
                dp_group=dp_group,
                device=device,
                batch_size=args.batch_size,
                context_length=GPT_CONFIG_124M["context_length"],
                emb_dim=GPT_CONFIG_124M["emb_dim"],
                scaler=scaler
            )

        if step % args.log_freq == 0:
            current_time = time.time()
            step_elapsed = current_time - last_log_time
            avg_step_time = step_elapsed / args.log_freq
            
            total_elapsed = current_time - start_time
            avg_time_per_step = total_elapsed / step
            steps_remaining = args.max_steps - step
            eta_seconds = int(steps_remaining * avg_time_per_step)
            
            if eta_seconds < 3600:
                eta_str = f"{eta_seconds // 60:02d}m {eta_seconds % 60:02d}s"
            else:
                eta_str = f"{eta_seconds // 3600:02d}h {(eta_seconds % 3600) // 60:02d}m"
                
            dur_seconds = int(total_elapsed)
            if dur_seconds < 3600:
                dur_str = f"{dur_seconds // 60:02d}m {dur_seconds % 60:02d}s"
            else:
                dur_str = f"{dur_seconds // 3600:02d}h {(dur_seconds % 3600) // 60:02d}m {dur_seconds % 60:02d}s"
                
            interval_tflops = tflops_per_step * args.log_freq
            tflops_per_sec = interval_tflops / step_elapsed
            
            tokens_trained = step * (args.batch_size * dp_size) * GPT_CONFIG_124M["context_length"]
            epoch_percent = (tokens_trained / total_epoch_tokens) * 100
            
            if loss_master and loss_val is not None:
                try:
                    perplexity = math.exp(loss_val)
                except OverflowError:
                    perplexity = float("inf")
                tokens_str = format_tokens(tokens_trained)
                print(f"Step {step:06d}/{args.max_steps:06d} - Loss: {loss_val:.4f} | PPL: {perplexity:.2f} | Time/step: {avg_step_time:.3f}s | TFLOPS: {tflops_per_sec:.4f} | Tokens: {tokens_str} | Epoch: {epoch_percent:.5f}% | Duration: {dur_str} | ETA: {eta_str}")
                
                if wandb is not None and wandb.run is not None:
                    wandb.log({
                        "loss": loss_val,
                        "ppl": perplexity,
                        "tflops": tflops_per_sec,
                        "tokens": tokens_trained,
                        "epoch_percent": epoch_percent,
                        "time_per_step": avg_step_time
                    }, step=step)
                
            last_log_time = current_time

        # Periodic checkpointing
        if (args.save_freq > 0 and step % args.save_freq == 0) or step == args.max_steps:
            if dp_coord == 0:
                save_checkpoint(model, optimizer, output_uri, step, pp_coord=pp_coord, tp_coord=tp_coord)
            
            if ddp:
                dist.barrier()
                
            if loss_master and step == args.max_steps:
                consolidate_checkpoints(output_uri, tp_size, pp_size, GPT_CONFIG_124M)

    if ddp:
        from torch.distributed import destroy_process_group
        destroy_process_group()

if __name__ == "__main__":
    main()
