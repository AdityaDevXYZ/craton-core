import urllib.request
import os
import torch
from tokenizer import CratonTokenizer

def prepare_knowledge_base():
    print("Initiating Craton Knowledge Acquisition Pipeline...")
    
    # We are starting Craton's brain with the highest-tier of OS engineering:
    # The core CPU scheduling logic from the Linux Kernel.
    # This aligns perfectly with your "Meta-Engineer" system genesis skill.
    url = "https://raw.githubusercontent.com/torvalds/linux/master/kernel/sched/core.c"
    file_path = "craton_seed_knowledge.c"
    
    if not os.path.exists(file_path):
        print("Downloading Linux Kernel Core Scheduler...")
        urllib.request.urlretrieve(url, file_path)
    else:
        print("Knowledge base already exists locally.")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    print(f"Successfully absorbed {len(text)} bytes of pure C systems logic.")
    
    tokenizer = CratonTokenizer()
    print("Tokenizing data into neural-compatible vectors...")
    ids = tokenizer.encode(text)
    
    data = torch.tensor(ids, dtype=torch.long)
    
    # Split into 90% Training Data, 10% Validation (Testing) Data
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # Save the tensors to disk
    torch.save(train_data, 'train.pt')
    torch.save(val_data, 'val.pt')
    
    print("Knowledge completely compressed into tensors.")
    print("Craton is ready for full-scale GPU training.")

if __name__ == '__main__':
    prepare_knowledge_base()
