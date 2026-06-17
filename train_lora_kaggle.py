import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os
# Force HuggingFace to download the massive 15GB model into Kaggle's 73GB partition,
# instead of the tiny root partition which causes a silent crash.
os.environ["HF_HOME"] = "/kaggle/working/huggingface"

def train_lora_kaggle():
    print("CRATON PHASE 7: ENTERPRISE LORA ENGINE INITIALIZING...")
    
    # We use Qwen2.5-7B because it is the most powerful completely open 7B model on earth.
    # It rivals Llama-3-8B but requires no API keys or corporate gatekeepers.
    model_id = "Qwen/Qwen2.5-7B"
    
    # 1. 4-Bit Quantization (Compressing a 14GB brain into 5GB)
    print("Compressing the 7-Billion Parameter Base Brain...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load the base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # 2. Prepare for LoRA
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    print("Attaching the Craton Neural Adapter...")
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    # 3. Load the conversational and scientific datasets
    print("Downloading Conversational & Scientific Datasets...")
    dataset = load_dataset("OpenAssistant/oasst1", split="train[:2%]")
    
    # Format the data for the SFTTrainer
    def formatting_func(example):
        role = "<|USER|>" if example['role'] == 'prompter' else "<|ASSISTANT|>"
        text = f"{role}\n{example['text']}\n"
        return text
    
    # 4. Train the Adapter
    print("Initiating LoRA Fine-Tuning Sequence on Kaggle GPU...")
    training_args = TrainingArguments(
        output_dir="/kaggle/working/craton_lora_adapter",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=100, # Set to a fixed number of steps for the Kaggle 12-hour window
        logging_steps=10,
        optim="paged_adamw_8bit",
        fp16=True,
        save_strategy="no"
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        formatting_func=formatting_func,
    )
    
    trainer.train()
    
    # 5. Save ONLY the adapter (This will be a tiny ~50MB file instead of 14GB!)
    print("Training Complete! Saving Craton LoRA Adapter...")
    trainer.save_model("/kaggle/working/craton_lora_adapter")

if __name__ == "__main__":
    train_lora_kaggle()
