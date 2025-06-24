#!/usr/bin/env python3
"""
💬 Chat with Original (Untrained) TinyLlama Model
Compare responses with your SFT trained model.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_original_model():
    """Load the original (untrained) TinyLlama model"""
    logger.info("🤖 Loading ORIGINAL (untrained) TinyLlama model...")
    
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load original model (no SFT training)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=True
    ).to("cpu")
    
    logger.info("✅ Original model loaded successfully!")
    return model, tokenizer

def chat_with_original(model, tokenizer, question):
    """Ask the original (untrained) model a question"""
    
    messages = [{"role": "user", "content": question}]
    
    # Format using chat template
    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(formatted_input, return_tensors="pt")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    response_start = full_response.find(formatted_input) + len(formatted_input)
    assistant_response = full_response[response_start:].strip()
    
    return assistant_response

def main():
    """Interactive chat with original (untrained) model"""
    print("💬 Chat with Original (Untrained) TinyLlama")
    print("🔍 This is the base model WITHOUT your SFT training")
    print("📊 Compare responses with your SFT model using chat_with_sft.py")
    print("=" * 70)
    
    # Load original model
    model, tokenizer = load_original_model()
    
    print("\n✅ Ready to chat! Type your questions (or 'quit' to exit)")
    print("-" * 70)
    
    while True:
        try:
            # Get user input
            user_input = input("\n❓ Your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                print("Please enter a question!")
                continue
            
            print("🤔 Original model thinking...")
            
            # Get response from original model
            response = chat_with_original(model, tokenizer, user_input)
            
            print(f"🤖 Original Model: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 