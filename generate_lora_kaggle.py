import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os
import glob

def generate_lora_kaggle():
    print("Booting Craton Phase 7 Inference Engine on CUDA...")
    
    model_id = "Qwen/Qwen2.5-7B"
    
    # 1. Compress and load the base brain
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # 2. Dynamically search for your trained LoRA adapter
    adapter_path = '/kaggle/working/craton_lora_adapter'
    input_adapters = glob.glob('/kaggle/input/**/craton_lora_adapter', recursive=True)
    
    if os.path.exists(adapter_path):
        target_adapter = adapter_path
    elif len(input_adapters) > 0:
        target_adapter = input_adapters[0]
    else:
        print("WARNING: 'craton_lora_adapter' not found in Kaggle! Did you attach the dataset?")
        return
        
    print(f"Loading Neural Adapter from {target_adapter}...")
    model = PeftModel.from_pretrained(base_model, target_adapter)
    
    # 3. Generate text
    prompt = "<|USER|>\nHello! Can you explain quantum computing to me?\n<|ASSISTANT|>\n"
    print(f"\n[PROMPT]: {prompt}")
    print("--- CRATON SYNTHESIS ---")
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7)
        
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Pass through validator
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
        print(generated_text)

if __name__ == "__main__":
    generate_lora_kaggle()
