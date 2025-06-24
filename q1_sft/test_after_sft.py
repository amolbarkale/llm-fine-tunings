#!/usr/bin/env python3
"""
💬 Chat with your SFT Trained Model
Load and interact with your fine-tuned TinyLlama model.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sft_model():
    """Load your SFT trained model"""
    logger.info("🤖 Loading your SFT trained model...")
    
    # Load base model
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=True
    )
    
    # Load your SFT adapters on top of base model
    model = PeftModel.from_pretrained(base_model, "./trained_model")
    model = model.to("cpu")
    
    logger.info("✅ SFT model loaded successfully!")
    return model, tokenizer

def chat_with_model(model, tokenizer, question):
    """Ask your SFT model a question"""
    
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
    print('full_response:', full_response)

    try:
        # Method 1: Look for assistant tag
        if "<|assistant|>" in full_response:
            parts = full_response.split("<|assistant|>")
            if len(parts) > 1:
                assistant_response = parts[-1].strip()
            else:
                assistant_response = full_response.split(formatted_input)[-1].strip()
        else:
            # Method 2: Use input length to extract new content
            input_tokens = len(tokenizer.encode(formatted_input))
            response_tokens = outputs[0][input_tokens:]  # Get only new tokens
            assistant_response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        
        # Fallback: if extraction failed, show full response
        if not assistant_response or len(assistant_response) < 5:
            print("🔍 Debug - Full response:", full_response)
            assistant_response = full_response.split(formatted_input)[-1].strip()
            
    except Exception as e:
        print(f"⚠️ Extraction error: {e}")
        # Emergency fallback - show everything after input
        assistant_response = full_response.replace(formatted_input, "").strip()
    
    return assistant_response

def main():
    """Interactive chat with your SFT model"""
    print("💬 Chat with Your SFT Trained Model")
    print("🏆 This is your fine-tuned TinyLlama with improved behavior")
    print("=" * 60)
    
    # Load your SFT model
    model, tokenizer = load_sft_model()
    
    print("\n✅ Ready to chat! Type your questions (or 'quit' to exit)")
    print("-" * 60)
    
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
            
            print("🤔 Thinking...")
            
            # Get response from your SFT model
            response = chat_with_model(model, tokenizer, user_input)
            
            print(f"🤖 SFT Model: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 