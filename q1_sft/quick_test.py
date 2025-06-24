#!/usr/bin/env python3
"""
Simple manual testing script for TinyLlama
Just run this and ask questions one by one
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model():
    """Load TinyLlama model"""
    print("🤖 Loading TinyLlama model...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else "cpu"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model loaded!")
    return model, tokenizer

def ask_question(model, tokenizer, question):
    """Ask the model a question"""
    print(f"\n❓ Question: {question}")
    
    messages = [{"role": "user", "content": question}]
    formatted_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_input, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response part
    try:
        assistant_start = response.find("<|assistant|>")
        if assistant_start != -1:
            answer = response[assistant_start + len("<|assistant|>"):].strip()
        else:
            answer = response[len(formatted_input):].strip()
    except:
        answer = response
    
    print(f"🤖 Answer: {answer}")
    return answer

def main():
    """Main function - ask questions manually"""
    model, tokenizer = load_model()
    
    print("\n" + "="*50)
    print("🎯 TinyLlama Manual Testing")
    print("Type your questions (or 'quit' to exit)")
    print("="*50)
    
    while True:
        question = input("\n❓ Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if question:
            ask_question(model, tokenizer, question)
        else:
            print("Please enter a question!")

if __name__ == "__main__":
    main() 