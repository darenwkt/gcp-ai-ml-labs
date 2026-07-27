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
    parser.add_argument("--max-steps", type=int, default=-1, help="Max training steps (-1 to disable)")
    parser.add_argument("--finetuning-type", type=str, default="lora", choices=["fft", "lora", "qlora", "dora"], help="Fine-tuning mode (fft, lora, qlora, dora)")
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
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        torch_dtype = torch.float32
    
    quantization_config = None
    if args.finetuning_type == "qlora":
        print("QLoRA enabled. Configuring 4-bit BitsAndBytesConfig...")
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map=device_map
    )

    # Convert Conv1D to nn.Linear for GPT-2 models to resolve PEFT DoRA shape mismatch bug
    if "gpt2" in args.model_id.lower():
        from transformers.pytorch_utils import Conv1D
        print("Converting Conv1D layers to standard nn.Linear layers for GPT-2 DoRA compatibility...")
        def convert_conv1d_to_linear(module):
            for name, child in module.named_children():
                if isinstance(child, Conv1D):
                    in_features, out_features = child.weight.shape
                    linear = torch.nn.Linear(in_features, out_features)
                    with torch.no_grad():
                        linear.weight.copy_(child.weight.t())
                        if child.bias is not None:
                            linear.bias.copy_(child.bias)
                    setattr(module, name, linear)
                else:
                    convert_conv1d_to_linear(child)
        convert_conv1d_to_linear(model)

    if args.finetuning_type == "qlora":
        from peft import prepare_model_for_kbit_training
        print("Preparing quantized model for training...")
        model = prepare_model_for_kbit_training(model)
    
    # 3. Configure LoRA / DoRA
    lora_config = None
    if args.finetuning_type != "fft":
        use_dora = (args.finetuning_type == "dora")
        print(f"Configuring PEFT Adapter (DoRA: {use_dora})...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["c_attn"] if "gpt2" in args.model_id.lower() else ["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=use_dora
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
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to="none",
        max_steps=args.max_steps
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
    
    local_output_dir = "./merged_model"
    if args.finetuning_type == "fft":
        print("Saving full fine-tuned model...")
        trainer.model.save_pretrained(local_output_dir)
        tokenizer.save_pretrained(local_output_dir)
        print("Full fine-tuned model saved locally.")
    else:
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
        
        model_final.save_pretrained(local_output_dir)
        tokenizer.save_pretrained(local_output_dir)
        print("Merged model weights saved locally.")
    
    # 7. Upload to GCS
    upload_folder_to_gcs(args.project_id, local_output_dir, args.output_model_gcs)
    print("Deployment upload sequence complete!")

if __name__ == "__main__":
    main()
