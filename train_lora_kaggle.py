import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import os
# Force HuggingFace to download the massive 15GB model into Kaggle's 73GB partition,
# instead of the tiny root partition which causes a silent crash.
os.environ["HF_HOME"] = "/kaggle/working/huggingface"

def train_lora_kaggle():
    try:
        print("CRATON PHASE 7: ENTERPRISE LORA ENGINE INITIALIZING...")
        
        model_id = "Qwen/Qwen2.5-7B"
        
        print("Compressing the 7-Billion Parameter Base Brain...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16 # Fixed for Kaggle T4 architecture compatibility
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Force model entirely onto GPU 0 to prevent CPU-offload crashes
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": 0}
        )
        
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
        
        print("Downloading Conversational Datasets...")
        dataset = load_dataset("OpenAssistant/oasst1", split="train[:2%]")
        
        # Format safely without relying on SFTTrainer's internal mapper
        def format_row(x):
            role = "<|USER|>" if x['role'] == 'prompter' else "<|ASSISTANT|>"
            x['text'] = f"{role}\n{x['text']}\n"
            return x
            
        dataset = dataset.map(format_row)
        
        # VERY IMPORTANT: OpenAssistant dataset has a column called 'labels' containing metadata.
        # PyTorch sees the word 'labels' and tries to use it for Neural Network math, causing a catastrophic crash.
        # We must strip all columns except our formatted 'text' column.
        columns_to_remove = [col for col in dataset.column_names if col != 'text']
        dataset = dataset.remove_columns(columns_to_remove)
        
        print("Initiating LoRA Fine-Tuning Sequence on Kaggle GPU...")
        training_args = SFTConfig(
            output_dir="/kaggle/working/craton_lora_adapter",
            per_device_train_batch_size=2, # Aggressively lowered to prevent OOM
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            max_steps=60, # Keep it extremely short just to verify pipeline completion
            logging_steps=10,
            optim="paged_adamw_8bit",
            fp16=True,
            save_strategy="no",
            max_length=512, # Renamed from max_seq_length in newest trl versions
            dataset_text_field="text" # Ensure it reads our custom formatted column
        )
        
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
        )
        
        trainer.train()
        
        print("Training Complete! Saving Craton LoRA Adapter...")
        trainer.save_model("/kaggle/working/craton_lora_adapter")
        
    except Exception as e:
        # If Kaggle crashes, write the exact error to a text file so it gets saved to the output!
        with open("/kaggle/working/CRASH_LOG.txt", "w") as f:
            import traceback
            f.write(traceback.format_exc())
        print(f"CRITICAL ERROR: {e}")

if __name__ == "__main__":
    train_lora_kaggle()
