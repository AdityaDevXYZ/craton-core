import torch
from train import CratonTorchModel
from tokenizer import CratonTokenizer
import os
import glob

def generate_kaggle():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Booting Craton Inference Engine on {device.upper()}...")
    
    model = CratonTorchModel(vocab_size=100277, d_model=1024, n_heads=16, n_layers=12).to(device)
    
    # 1. Search for the trained brain dynamically in Kaggle
    brain_path = '/kaggle/working/craton_megabrain.pth'
    input_brains = glob.glob('/kaggle/input/**/craton_megabrain.pth', recursive=True)
    
    if os.path.exists(brain_path):
        target_brain = brain_path
    elif len(input_brains) > 0:
        target_brain = input_brains[0]
    else:
        print("WARNING: 'craton_megabrain.pth' not found in Kaggle! Did you attach the dataset?")
        return
        
    model.load_state_dict(torch.load(target_brain, map_location=device, weights_only=True))
    print(f"Mega-Brain neural weights successfully loaded from {target_brain}.")
        
    model.eval()
    tokenizer = CratonTokenizer()
    
    prompt = "<|USER|>\nHello! Can you explain quantum computing to me?\n<|ASSISTANT|>\n"
    print(f"\n[PROMPT]: {prompt}")
    print("--- CRATON SYNTHESIS ---")
    
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    max_new_tokens = 300
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -2048:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :] 
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
    generated_text = tokenizer.decode(idx[0].tolist())
    
    # Run through the Constitution
    try:
        from validator import ConstitutionalValidator
        validator = ConstitutionalValidator()
        is_safe, message = validator.validate(generated_text)
        
        if not is_safe:
            print(f"\n[!!!] SYSTEM INTERCEPT [!!!]")
            print(f"ERROR: {message}")
        else:
            print(f"\n[VALIDATOR]: {message}\n")
            print(generated_text)
    except Exception:
        # Fallback if validator fails to load on Kaggle
        print(generated_text)
        
    print("------------------------\n")

if __name__ == '__main__':
    generate_kaggle()
