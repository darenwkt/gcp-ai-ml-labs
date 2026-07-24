import os
import argparse
import pandas as pd
import torch
from urllib.parse import urlparse
from google.cloud import storage
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=str, required=True)
    parser.add_argument("--model-id", type=str, default="gpt2", help="HF model name or ID")
    parser.add_argument("--dataset-name", type=str, default="enzo-joseph/customer-support-tickets")
    parser.add_argument("--dataset-config", type=str, default="en")
    parser.add_argument("--dataset-csv-gcs", type=str, default="", help="Optional GCS URI to custom dataset CSV")
    parser.add_argument("--output-model-gcs", type=str, required=True, help="Target GCS folder to save the merged fine-tuned model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    return parser.parse_args()

def download_gcs_file(project_id, gcs_uri, local_path):
    parsed = urlparse(gcs_uri)
    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip("/")
    
    print(f"Downloading dataset CSV from {gcs_uri}...")
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    print("Download completed.")

def upload_folder_to_gcs(project_id, local_folder, gcs_uri):
    parsed = urlparse(gcs_uri)
    bucket_name = parsed.netloc
    prefix = parsed.path.lstrip("/")
    
    print(f"Uploading merged model artifacts to {gcs_uri}...")
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    
    for root, _, files in os.walk(local_folder):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_folder)
            blob_name = os.path.join(prefix, relative_path)
            
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded: {relative_path} -> gs://{bucket_name}/{blob_name}")

def map_custom_csv_columns(df):
    desc_keys = ["body", "ticket_description", "description", "issue", "ticket_body"]
    res_keys = ["answer", "resolution", "reply", "response", "solution"]
    
    mapped_desc = None
    mapped_res = None
    
    for col in df.columns:
        if col.lower() in desc_keys:
            mapped_desc = col
        if col.lower() in res_keys:
            mapped_res = col
            
    if not mapped_desc or not mapped_res:
        raise ValueError(
            f"Could not automatically map dataset columns. Available: {list(df.columns)}. "
            f"Expected description-like header ({desc_keys}) and resolution-like header ({res_keys})."
        )
        
    print(f"Mapped description -> '{mapped_desc}' and resolution -> '{mapped_res}'")
    return df[[mapped_desc, mapped_res]].rename(columns={mapped_desc: "body", mapped_res: "answer"})

def format_prompts(example):
    text = (
        "Below is an IT support ticket description. Write a response that resolves the ticket.\n\n"
        f"### Ticket:\n{example['body']}\n\n"
        f"### Response:\n{example['answer']}"
    )
    return {"text": text}

def main():
    args = parse_args()
    
    # 1. Load Dataset
    if args.dataset_csv_gcs:
        local_csv = "/tmp/dataset.csv"
        download_gcs_file(args.project_id, args.dataset_csv_gcs, local_csv)
        df = pd.read_csv(local_csv)
        df_cleaned = map_custom_csv_columns(df)
        dataset = Dataset.from_pandas(df_cleaned)
    else:
        print(f"Loading public mirror dataset '{args.dataset_name}' ({args.dataset_config})...")
        raw_dataset = load_dataset(args.dataset_name, args.dataset_config, split="train")
        # Ensure it has body and answer columns
        dataset = raw_dataset.select_columns(["body", "answer"])
        
    # Map examples to prompt template text
    dataset = dataset.map(format_prompts)
    print(f"Dataset formatted. Record count: {len(dataset)}")
    
    # 2. Load Model and Tokenizer
    print(f"Loading tokenizer and model: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    device_map = "auto" if torch.cuda.is_available() else None
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    
    # 3. Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["c_attn"] if "gpt2" in args.model_id.lower() else ["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    # 5. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args
    )
    
    print("Starting SFT fine-tuning training run...")
    trainer.train()
    print("Fine-tuning completed successfully!")
    
    # Save adapter
    adapter_dir = "./adapter"
    trainer.model.save_pretrained(adapter_dir)
    print("Adapter saved locally.")
    
    # 6. Merge Adapter with Base Model
    print("Merging adapter weights with base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None
    )
    model_merged = PeftModel.from_pretrained(base_model, adapter_dir)
    model_final = model_merged.merge_and_unload()
    
    local_output_dir = "./merged_model"
    model_final.save_pretrained(local_output_dir)
    tokenizer.save_pretrained(local_output_dir)
    print("Merged model weights saved locally.")
    
    # 7. Upload to GCS
    upload_folder_to_gcs(args.project_id, local_output_dir, args.output_model_gcs)
    print("Deployment upload sequence complete!")

if __name__ == "__main__":
    main()
