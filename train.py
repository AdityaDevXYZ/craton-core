import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# We rewrite the Craton Architecture into PyTorch for the Training Phase.
# Why? Because PyTorch handles the extreme calculus (Backpropagation) needed to 
# update the weights, and allows us to deploy this code to free Cloud GPUs.
# -----------------------------------------------------------------------------

class CratonTorchModel(nn.Module):
    def __init__(self, vocab_size=100277, d_model=256, n_heads=8, n_layers=4):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(2048, d_model) # Expanded from 512 to handle Mega-Brain context length
        
        # PyTorch has the Transformer block built-in, but we use it identically
        # to the math we wrote from scratch in NumPy.
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        x = self.token_embedding(idx) + self.position_embedding(pos)
        
        # Create a causal mask so it can't look into the future during training
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)
        
        x = self.transformer(x, mask=mask, is_causal=True)
        logits = self.lm_head(x)
        return logits

def train():
    print("Initializing Craton Training Sequence...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Hardware Detected: {device.upper()}")
    
    model = CratonTorchModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    print("Loading Core Knowledge Tensors...")
    try:
        train_data = torch.load('train.pt')
    except FileNotFoundError:
        print("ERROR: Run prepare_data.py first to download the knowledge base!")
        return

    # Basic data loader function
    def get_batch(split='train', batch_size=32, block_size=128):
        data = train_data # We just use train_data for this micro-test
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y
    
    print("Initiating Neural Weight Optimization (Training)...")
    model.train()
    
    # We run 100 optimization steps for the test
    for step in range(100): 
        X, Y = get_batch()
        
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Epoch {step} | Brain Error (Loss): {loss.item():.4f}")
            
    print("Saving neural pathways to disk...")
    torch.save(model.state_dict(), 'craton_brain.pth')
    print("Training Pulse Complete. 'craton_brain.pth' is ready.")

if __name__ == "__main__":
    train()
