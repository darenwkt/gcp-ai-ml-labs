import os
import sys
from types import ModuleType

# Create mock torch module to bypass macOS shared memory/multiprocessing crashes
class DummyTensor:
    def share_memory_(self):
        return self
    def __int__(self):
        return 0
    def __index__(self):
        return 0

def mock_tensor(*args, **kwargs):
    return DummyTensor()

import importlib.machinery

mock_torch = ModuleType("torch")
mock_torch.tensor = mock_tensor
mock_torch.Tensor = DummyTensor
mock_torch.Generator = DummyTensor
mock_torch.nn = ModuleType("torch.nn")
mock_torch.nn.Module = DummyTensor

# Populate __spec__ to avoid importlib ValueError
mock_torch.__spec__ = importlib.machinery.ModuleSpec("torch", None)
mock_torch.nn.__spec__ = importlib.machinery.ModuleSpec("torch.nn", None)

# Add mock submodules torch.utils and torch.utils.data
mock_torch_utils = ModuleType("torch.utils")
mock_torch_utils_data = ModuleType("torch.utils.data")

class DummyIterableDataset:
    pass

mock_torch_utils_data.IterableDataset = DummyIterableDataset
mock_torch_utils_data.get_worker_info = lambda: None

mock_torch_utils.__spec__ = importlib.machinery.ModuleSpec("torch.utils", None)
mock_torch_utils_data.__spec__ = importlib.machinery.ModuleSpec("torch.utils.data", None)

# Bind submodule attributes on parent modules
mock_torch.utils = mock_torch_utils
mock_torch_utils.data = mock_torch_utils_data

sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = mock_torch.nn
sys.modules["torch.utils"] = mock_torch_utils
sys.modules["torch.utils.data"] = mock_torch_utils_data

import argparse
import numpy as np
import tiktoken
from datasets import load_dataset
from google.cloud import storage
from urllib.parse import urlparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default="sample-10BT")
    parser.add_argument("--num-docs", type=int, default=50000, help="Number of documents to pre-tokenize")
    parser.add_argument("--output-gcs-uri", type=str, required=True, help="Target GCS path, e.g. gs://bucket-name/dataset/train.bin")
    parser.add_argument("--local-dataset-path", type=str, default=None, help="Local directory path to load dataset from")
    parser.add_argument("--tfds-dataset", type=str, default=None, help="Name of TFDS dataset to load (e.g. huggingfacefw__fineweb_edu)")
    parser.add_argument("--tfds-data-dir", type=str, default=None, help="GCS/local data directory for TFDS custom datasets")
    parser.add_argument("--tfds-split", type=str, default="train", help="Dataset split to load from TFDS (e.g. train, all)")
    args = parser.parse_args()

    is_tfds = False
    if args.local_dataset_path:
        print(f"Loading local dataset from {args.local_dataset_path}...")
        from datasets import load_from_disk
        # Try loading as a saved HF dataset on disk first
        try:
            dataset = load_from_disk(args.local_dataset_path)
        except Exception:
            # Fallback: try loading as local parquet/json folder if it contains files
            import glob
            parquet_files = glob.glob(os.path.join(args.local_dataset_path, "*.parquet"))
            if parquet_files:
                dataset = load_dataset("parquet", data_files=parquet_files, split="train")
            else:
                json_files = glob.glob(os.path.join(args.local_dataset_path, "*.json"))
                if json_files:
                    dataset = load_dataset("json", data_files=json_files, split="train")
                else:
                    raise ValueError(f"Could not load local dataset from {args.local_dataset_path}. Ensure it contains .parquet or .json files.")
    elif args.tfds_dataset:
        print(f"Loading TFDS dataset {args.tfds_dataset} (split={args.tfds_split}, data_dir={args.tfds_data_dir})...")
        import tensorflow_datasets as tfds
        dataset = tfds.load(args.tfds_dataset, split=args.tfds_split, data_dir=args.tfds_data_dir)
        is_tfds = True
    else:
        print(f"Loading Hugging Face dataset fineweb-edu ({args.subset}) in streaming mode...")
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", name=args.subset, split="train", streaming=True)
    
    print("Initializing tiktoken GPT-2 encoder...")
    enc = tiktoken.get_encoding("gpt2")
    eot_token = 50256 # <|endoftext|>
 
    local_filename = "temp_dataset.bin"
    print(f"Tokenizing first {args.num_docs} documents and writing to local file {local_filename}...")
    
    token_count = 0
    with open(local_filename, "wb") as f:
        doc_idx = 0
        for item in dataset:
            if args.num_docs != -1 and doc_idx >= args.num_docs:
                break
            
            if is_tfds:
                text = item["text"].numpy().decode("utf-8")
            else:
                text = item["text"]
            # Encode text, add EOT token
            tokens = enc.encode(text, allowed_special={"<|endoftext|>"}) + [eot_token]
            
            # Convert to uint16
            tokens_np = np.array(tokens, dtype=np.uint16)
            f.write(tokens_np.tobytes())
            
            token_count += len(tokens)
            doc_idx += 1
            if doc_idx % 5000 == 0:
                print(f"Processed {doc_idx}/{args.num_docs} documents. Total tokens: {token_count:,}")

    print(f"Tokenization complete. Total tokens saved: {token_count:,} ({token_count * 2 / (1024*1024):.2f} MB)")

    # Upload to GCS
    print(f"Uploading local file to GCS: {args.output_gcs_uri}...")
    parsed = urlparse(args.output_gcs_uri)
    bucket_name = parsed.netloc
    blob_path = parsed.path.lstrip("/")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    blob.upload_from_filename(local_filename)
    print("Upload completed successfully!")
    
    # Clean up local file
    if os.path.exists(local_filename):
        os.remove(local_filename)
        print("Cleaned up temporary local file.")

if __name__ == "__main__":
    main()
