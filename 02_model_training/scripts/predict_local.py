import os
import torch
import tiktoken
import sys

# Ensure src/prediction directory is in python path
script_dir = os.path.dirname(os.path.abspath(__file__))
prediction_dir = os.path.abspath(os.path.join(script_dir, "..", "src", "prediction"))
sys.path.append(prediction_dir)
from main import GPTModel, GPT_CONFIG_124M, generate_text_simple

def predict(prompt: str, max_new_tokens: int = 50, model_path: str = "model.pth"):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() else 
        "cpu"
    )
    print(f"Using device: {device}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    print(f"Loading model from {model_path}...")
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(prompt)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
    
    print(f"Generating up to {max_new_tokens} tokens...")
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded_tensor,
            max_new_tokens=max_new_tokens,
            context_size=GPT_CONFIG_124M["context_length"]
        )
        
    decoded = tokenizer.decode(token_ids.squeeze(0).tolist())
    return decoded

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Hello, my name is", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Maximum new tokens to generate")
    parser.add_argument("--model-path", type=str, default="model.pth", help="Path to local model.pth")
    args = parser.parse_args()
    
    try:
        result = predict(args.prompt, args.max_new_tokens, args.model_path)
        print("\n--- Generated Output ---")
        print(result)
    except Exception as e:
        print(f"Error running prediction: {e}")
