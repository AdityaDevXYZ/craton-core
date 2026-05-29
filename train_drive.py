import torch
import torch.nn as nn
from torch.nn import functional as F
from train import CratonTorchModel
from tokenizer import CratonTokenizer
import os

def train_mega_brain():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"CRATON MEGA-BRAIN INITIALIZING ON {device.upper()}...")
    
    # 1. Boot up the massive architecture
    model = CratonTorchModel(d_model=1024, n_heads=16, n_layers=12).to(device)
    
    # Lower learning rate because massive models are highly sensitive during training
    model = CratonTorchModel(d_model=1024, n_heads=16, n_layers=12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) 
    
    brain_path = '/content/drive/MyDrive/Craton_Knowledge/craton_megabrain.pth'
    if os.path.exists(brain_path):
        print("Found existing Mega-Brain! Loading previous memories to resume training...")
        model.load_state_dict(torch.load(brain_path, map_location=device, weights_only=True))
    else:
        print("Starting Mega-Brain from scratch.")
    
    # 2. Read the knowledge chunks directly from Google Drive
    chunk_path = '/content/drive/MyDrive/Craton_Knowledge/knowledge_chunk_01.c'
    if not os.path.exists(chunk_path):
        print(f"ERROR: Cannot find chunk at {chunk_path}")
        print("Ensure you uploaded it to Google Drive in the exact folder 'Craton_Knowledge'.")
        return
        
    print("Absorbing Knowledge Chunk 01 directly from Cloud Storage...")
    with open(chunk_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    tokenizer = CratonTokenizer()
    ids = tokenizer.encode(text)
    train_data = torch.tensor(ids, dtype=torch.long)
    
    print(f"Knowledge absorbed: {len(train_data)} neural tokens.")
    
    # We increase block size to 1024 to give Craton a massive context window
    def get_batch(batch_size=8, block_size=1024): 
        ix = torch.randint(len(train_data) - block_size, (batch_size,))
        x = torch.stack([train_data[i:i+block_size] for i in ix])
        y = torch.stack([train_data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y
        
    print("Initiating Deep Neural Weight Optimization...")
    model.train()
    
    # We run in chunks of 1000 so the GPU doesn't get killed before saving.
    # You can just run the script over and over to stack the training!
    for step in range(1000):
        X, Y = get_batch()
        
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = F.cross_entropy(logits.view(-1, 100), Y.view(-1))
        loss.backward()
        optimizer.step()
        
        # Log progress every 100 steps so we don't spam the console
        if step % 100 == 0:
            print(f"Epoch {step} | Mega-Brain Error (Loss): {loss.item():.4f}")
            
    print("Saving Mega-Brain directly to Google Drive...")
    # Notice we save it straight to Google Drive so it doesn't get lost when Colab shuts down!
    torch.save(model.state_dict(), '/content/drive/MyDrive/Craton_Knowledge/craton_megabrain.pth')
    print("Training Complete. Craton has leveled up.")

if __name__ == "__main__":
    train_mega_brain()
