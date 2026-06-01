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
    
    model = CratonTorchModel(d_model=1024, n_heads=16, n_layers=12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) 
    
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
    
    # Search for the chunk dynamically
    chunks = glob.glob('/kaggle/input/**/knowledge_chunk_01.c', recursive=True)
    if len(chunks) == 0:
        print("ERROR: Cannot find knowledge_chunk_01.c inside /kaggle/input/. Did you click 'Add Input' on the right panel?")
        return
        
    chunk_path = chunks[0]
        
    print("Absorbing Knowledge Chunk...")
    with open(chunk_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    tokenizer = CratonTokenizer()
    ids = tokenizer.encode(text)
    train_data = torch.tensor(ids, dtype=torch.long)
    
    def get_batch(batch_size=8, block_size=1024): 
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
        logits = model(X)
        loss = F.cross_entropy(logits.view(-1, 100), Y.view(-1))
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Epoch {step} | Mega-Brain Error (Loss): {loss.item():.4f}")
            
    print("Saving Mega-Brain to Kaggle Working Directory...")
    torch.save(model.state_dict(), brain_path)
    print("Training Pulse Complete.")

if __name__ == "__main__":
    train_mega_brain_kaggle()
