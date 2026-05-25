import torch
from train import CratonTorchModel
from tokenizer import CratonTokenizer

def generate():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Booting Craton Inference Engine on {device.upper()}...")
    
    model = CratonTorchModel().to(device)
    
    # Load the trained brain weights
    try:
        model.load_state_dict(torch.load('craton_brain.pth', map_location=device, weights_only=True))
        print("Neural weights successfully loaded. Craton is online.")
    except FileNotFoundError:
        print("WARNING: 'craton_brain.pth' not found! You must run the updated train.py first.")
        return
        
    model.eval()
    tokenizer = CratonTokenizer()
    
    # We feed Craton a starting prompt to see how it continues the C code
    prompt = "struct sched_entity {"
    print(f"\n[PROMPT]: {prompt}")
    print("--- CRATON SYNTHESIS ---")
    
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    max_new_tokens = 300
    for _ in range(max_new_tokens):
        # Keep context within the max sequence length of the brain (512)
        idx_cond = idx[:, -512:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus on the very last predicted token
        logits = logits[:, -1, :] 
        
        # Convert to probabilities and sample the next character
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)
        
    # Translate the math vectors back into readable text
    generated_text = tokenizer.decode(idx[0].tolist())
    
    # INTERCEPT: Run it through the Constitutional Validator before showing the user
    from validator import ConstitutionalValidator
    validator = ConstitutionalValidator()
    is_safe, message = validator.validate(generated_text)
    
    if not is_safe:
        print(f"\n[!!!] SYSTEM INTERCEPT [!!!]")
        print(f"ERROR: {message}")
        print("Craton's output was destroyed before it could reach the host.")
    else:
        print(f"\n[VALIDATOR]: {message}\n")
        print(generated_text)
        
    print("------------------------\n")

if __name__ == '__main__':
    generate()
