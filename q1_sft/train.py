#!/usr/bin/env python3
"""
🚀 Tiny SFT Training Script
Fine-tune TinyLlama-1.1B-Chat-v1.0 to be more polite and helpful.
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(file_path="dataset.jsonl"):
    """Load and prepare the dataset"""
    logger.info(f"📁 Loading dataset from {file_path}")
    
    conversations = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                if "messages" in data:
                    conversations.append(data["messages"])
    
    logger.info(f"✅ Loaded {len(conversations)} conversations")
    return conversations

def prepare_data(conversations, tokenizer, max_length=512):
    """Convert conversations to tokenized format"""
    logger.info("🔄 Converting conversations to training format...")
    
    formatted_texts = []
    
    for conversation in conversations:
        # Use the tokenizer's chat template
        formatted_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False  # We want the full conversation
        )
        formatted_texts.append(formatted_text)
    
    # Tokenize all texts
    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None
    )
    
    # Create dataset
    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    })
    
    logger.info(f"✅ Prepared {len(dataset)} training examples")
    return dataset

def setup_model_and_tokenizer(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Load model and tokenizer with LoRA configuration"""
    logger.info(f"🤖 Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add pad token if missing (required for training)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else "cpu"
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,                # Rank
        lora_alpha=32,      # Alpha scaling
        lora_dropout=0.1,   # Dropout
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # Attention layers
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"📊 Trainable parameters: {trainable_params:,}")
    logger.info(f"📊 Total parameters: {total_params:,}")
    logger.info(f"📊 Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    return model, tokenizer

def train_model(model, tokenizer, train_dataset, output_dir="./trained_model"):
    """Train the model with specified parameters"""
    logger.info("🚀 Starting training...")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        report_to=None,  # Disable wandb/tensorboard for simplicity
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train!
    logger.info("🎯 Training starting...")
    trainer.train()
    
    # Save the model
    logger.info(f"💾 Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    logger.info("✅ Training completed successfully!")
    return trainer

def test_model(model, tokenizer):
    """Quick test of the trained model"""
    logger.info("🧪 Testing the trained model...")
    
    # Test conversation
    test_messages = [
        {"role": "user", "content": "Please explain what Python is."}
    ]
    
    # Format using chat template
    formatted_input = tokenizer.apply_chat_template(
        test_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(formatted_input, return_tensors="pt")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the new part (after the input)
    response_start = full_response.find(formatted_input) + len(formatted_input)
    new_response = full_response[response_start:].strip()
    
    logger.info("📝 Sample generation:")
    logger.info(f"Input: {test_messages[0]['content']}")
    logger.info(f"Output: {new_response}")

def main():
    """Main training pipeline"""
    print("🎯 Starting Tiny SFT Training Pipeline")
    print("=" * 50)
    
    try:
        # 1. Load dataset
        conversations = load_dataset("dataset.jsonl")
        
        # 2. Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer()
        
        # 3. Prepare training data
        train_dataset = prepare_data(conversations, tokenizer)
        
        # 4. Train the model
        trainer = train_model(model, tokenizer, train_dataset)
        
        # 5. Test the model
        test_model(model, tokenizer)
        
        print("\n🎉 Training pipeline completed successfully!")
        print(f"📁 Model saved in: ./trained_model")
        print("🔄 You can now run evaluation tests!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 