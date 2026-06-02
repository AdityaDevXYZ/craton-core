import torch
import torch.nn as nn
from torch.nn import functional as F
from train import CratonTorchModel
from tokenizer import CratonTokenizer
import os
import glob

def train_mega_brain_kaggle():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"CRATON KAGGLE CLOUD INITIALIZING ON {device.upper()}...")
    
    # Phase 6: Expanded vocabulary support
    model = CratonTorchModel(vocab_size=100277, d_model=1024, n_heads=16, n_layers=12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) 
    
    # Initialize Mixed Precision Scaler to save massive amounts of VRAM
    scaler = torch.amp.GradScaler('cuda') 
    
    # In Kaggle, we save to /kaggle/working/ so the output is preserved
    brain_path = '/kaggle/working/craton_megabrain.pth'
    
    # Search for the dataset brain dynamically (bypassing exact folder names)
    input_brains = glob.glob('/kaggle/input/**/craton_megabrain.pth', recursive=True)
    
    if os.path.exists(brain_path):
        print("Found active working Mega-Brain! Loading...")
        model.load_state_dict(torch.load(brain_path, map_location=device, weights_only=True))
    elif len(input_brains) > 0:
        print("Found dataset Mega-Brain! Loading previous memories...")
        model.load_state_dict(torch.load(input_brains[0], map_location=device, weights_only=True))
    else:
        print("Starting Mega-Brain from scratch on Kaggle.")
    
    print("Downloading Phase 6 Conversational & Scientific Knowledge...")
    import datasets
    # 1. Load Conversational Data (OpenAssistant)
    chat_dataset = datasets.load_dataset("OpenAssistant/oasst1", split="train[:1%]")
    
    # 2. Load Scientific Research Data (arXiv abstracts)
    arxiv_dataset = datasets.load_dataset("gfissore/arxiv-abstracts-2021", split="train[:1%]")
    
    tokenizer = CratonTokenizer()
    
    # Format the conversations with <|USER|> and <|ASSISTANT|>
    all_text = ""
    for row in chat_dataset:
        role = "<|USER|>" if row['role'] == 'prompter' else "<|ASSISTANT|>"
        all_text += f"{role}\n{row['text']}\n"
        
    # Append scientific research to the brain's knowledge pool
    for row in arxiv_dataset:
        all_text += f"<|SYSTEM|>\nResearch Abstract: {row['title']}\n{row['abstract']}\n"
        
    ids = tokenizer.encode(all_text)
    train_data = torch.tensor(ids, dtype=torch.long)
    
    def get_batch(batch_size=4, block_size=1024): 
        ix = torch.randint(len(train_data) - block_size, (batch_size,))
        x = torch.stack([train_data[i:i+block_size] for i in ix])
        y = torch.stack([train_data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y
        
    print("Initiating Deep Neural Weight Optimization...")
    model.train()
    
    # Kaggle allows up to 12 hours of headless execution!
    for step in range(5000):
        X, Y = get_batch()
        
        optimizer.zero_grad(set_to_none=True)
        
        # Use Automatic Mixed Precision (AMP) to cut VRAM usage in half
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if step % 100 == 0:
            print(f"Epoch {step} | Mega-Brain Error (Loss): {loss.item():.4f}")
            
    print("Saving Mega-Brain to Kaggle Working Directory...")
    torch.save(model.state_dict(), brain_path)
    print("Training Pulse Complete.")

if __name__ == "__main__":
    train_mega_brain_kaggle()
