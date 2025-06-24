#!/usr/bin/env python3
"""
🚀 Original Parameters SFT Training Script
Fine-tune TinyLlama-1.1B-Chat-v1.0 with the original specified parameters.
"""

import os
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import logging

# Force CPU usage for stability (avoid 4GB GPU limitation)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(6)  # Increased for full training

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(file_path="dataset.jsonl"):
    """Load the complete dataset"""
    logger.info(f"📁 Loading FULL dataset from {file_path}")
    
    conversations = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                if "messages" in data:
                    conversations.append(data["messages"])
    
    logger.info(f"✅ Loaded {len(conversations)} conversations")
    return conversations

def prepare_data(conversations, tokenizer, max_length=512):  # ORIGINAL: 512 tokens
    """Convert conversations to tokenized format with original max_length"""
    logger.info("🔄 Converting conversations to training format...")
    
    formatted_texts = []
    
    for conversation in conversations:
        formatted_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        formatted_texts.append(formatted_text)
    
    # Tokenize with original max_length
    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None
    )
    
    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    })
    
    logger.info(f"✅ Prepared {len(dataset)} training examples")
    return dataset

def setup_model_and_tokenizer(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Load model and tokenizer with ORIGINAL LoRA configuration"""
    logger.info(f"🤖 Loading model: {model_name}")
    logger.info(f"🔧 Using device: CPU (GPU disabled for stability)")
    logger.info(f"💻 CPU threads: {torch.get_num_threads()}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model optimized for CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=True,
        use_cache=False
    )
    
    model = model.to("cpu")
    
    # ORIGINAL LoRA Configuration
    lora_config = LoraConfig(
        r=16,                # ORIGINAL: 16
        lora_alpha=32,       # ORIGINAL: 32
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # ORIGINAL: All 4 modules
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"📊 Trainable parameters: {trainable_params:,}")
    logger.info(f"📊 Total parameters: {total_params:,}")
    logger.info(f"📊 Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    logger.info(f"🎯 Using ORIGINAL LoRA settings: r=16, alpha=32, 4 target modules")
    
    return model, tokenizer

def train_model(model, tokenizer, train_dataset, output_dir="./trained_model"):
    """Train the model with ORIGINAL parameters"""
    logger.info("🚀 Starting training with ORIGINAL parameters...")
    
    # ORIGINAL Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,              # ORIGINAL: 3 epochs
        per_device_train_batch_size=2,   # ORIGINAL: 2
        gradient_accumulation_steps=2,   # ORIGINAL: 2
        learning_rate=5e-5,              # ORIGINAL: 5e-5
        warmup_steps=50,                 # ORIGINAL: 50
        logging_steps=10,                # ORIGINAL: 10
        save_steps=100,                  # ORIGINAL: 100
        save_total_limit=2,              # ORIGINAL: 2
        remove_unused_columns=False,
        dataloader_drop_last=True,
        report_to=None,
        fp16=False,                      # CPU training
        dataloader_num_workers=0,
        optim="adamw_torch",
        no_cuda=True,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    
    # Train with original settings!
    logger.info("🎯 Training starting with ORIGINAL parameters...")
    trainer.train()
    
    # Save the model
    logger.info(f"💾 Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    logger.info("✅ Original parameters training completed successfully!")
    return trainer

def test_model(model, tokenizer):
    """Test the trained model with multiple examples"""
    logger.info("🧪 Testing the trained model...")
    
    test_cases = [
        "What is Python?",
        "Please help me write a thank you email.",
        "Could you please explain how to make coffee?",
        "Tell me about dogs."
    ]
    
    for i, question in enumerate(test_cases, 1):
        test_messages = [{"role": "user", "content": question}]
        
        formatted_input = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(formatted_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,  # ORIGINAL: 100 tokens
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_start = full_response.find(formatted_input) + len(formatted_input)
        new_response = full_response[response_start:].strip()
        
        logger.info(f"📝 Test {i}:")
        logger.info(f"   Input: {question}")
        logger.info(f"   Output: {new_response}")
        logger.info("")

def main():
    """Main training pipeline with ORIGINAL parameters"""
    print("🎯 Original Parameters SFT Training Pipeline")
    print("📋 Using the COMPLETE original specification")
    print("💻 CPU-optimized to avoid 4GB GPU limitation")
    print("=" * 60)
    
    print("\n📊 ORIGINAL PARAMETERS:")
    print("   📁 Dataset: ALL 26 samples")
    print("   📏 Max length: 512 tokens")
    print("   🔄 Epochs: 3")
    print("   📦 Batch size: 2")
    print("   📈 Learning rate: 5e-5")
    print("   🎯 LoRA: r=16, alpha=32, 4 modules")
    print("   ⏰ Expected time: 15-30 minutes")
    print("-" * 60)
    
    try:
        # 1. Load full dataset
        conversations = load_dataset("dataset.jsonl")
        
        # 2. Setup model and tokenizer with original LoRA
        model, tokenizer = setup_model_and_tokenizer()
        
        # 3. Prepare training data with original max_length
        train_dataset = prepare_data(conversations, tokenizer)
        
        # 4. Train with original parameters
        trainer = train_model(model, tokenizer, train_dataset)
        
        # 5. Test the model
        test_model(model, tokenizer)
        
        print("\n🎉 Original parameters training completed successfully!")
        print(f"📁 Model saved in: ./trained_model")
        print("🏆 This is your FULL QUALITY SFT model!")
        print("📈 Expect significantly better responses than lightweight version")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()

# ORIGINAL VS LIGHTWEIGHT COMPARISON:
"""
🔄 PARAMETER CHANGES FROM LIGHTWEIGHT → ORIGINAL:

DATASET:
- Samples: 10 → 26 (2.6x more data)
- Max length: 256 → 512 tokens (2x longer sequences)

TRAINING:
- Epochs: 1 → 3 (3x more training)
- Batch size: 1 → 2 (2x larger batches)
- Learning rate: 2e-5 → 5e-5 (2.5x higher)
- Warmup: 5 → 50 steps (10x more warmup)

LORA:
- Rank: 8 → 16 (2x more parameters)
- Alpha: 16 → 32 (2x scaling)
- Modules: 2 → 4 (2x more target modules)

🎯 EXPECTED QUALITY IMPROVEMENTS:
- Much better instruction following
- More coherent and complete responses
- Better context understanding
- More diverse vocabulary usage
- Improved politeness and helpfulness

⚠️ TRADE-OFFS:
- Longer training time (15-30 minutes vs 3-4 minutes)
- Higher CPU usage during training
- More memory usage
- Better final model quality
""" 