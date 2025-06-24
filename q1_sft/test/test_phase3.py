#!/usr/bin/env python3
"""
🧪 Phase 3 Test: Training Script Setup Validation
This test ensures the training script loads models, processes data, and sets up LoRA correctly.
"""

import os
import json
import torch

def test_model_loading():
    """Test if we can load TinyLlama model and tokenizer"""
    print("🤖 Testing TinyLlama model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        # Load tokenizer first (lighter)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✅ Tokenizer loaded: {tokenizer.name_or_path}")
        
        # Load model (this will take longer)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        print(f"✅ Model loaded: {model.config.model_type}")
        print(f"📊 Model parameters: ~{model.num_parameters()//1000000}M")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_chat_template():
    """Test if tokenizer has proper chat template"""
    print("\n💬 Testing chat template...")
    
    try:
        from transformers import AutoTokenizer
        
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Test chat template exists
        if tokenizer.chat_template is None:
            print("❌ No chat template found!")
            return False
        
        # Test sample conversation
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you?"}
        ]
        
        # Apply chat template
        formatted_chat = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        print("✅ Chat template working!")
        print(f"📝 Sample format:\n{formatted_chat[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Chat template test failed: {e}")
        return False

def test_dataset_compatibility():
    """Test if our dataset can be converted to TinyLlama format"""
    print("\n📁 Testing dataset compatibility...")
    
    try:
        if not os.path.exists("dataset.jsonl"):
            print("❌ dataset.jsonl not found!")
            return False
        
        from transformers import AutoTokenizer
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load and test first few examples
        converted_count = 0
        with open("dataset.jsonl", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 3:  # Test first 3 examples
                    break
                    
                data = json.loads(line.strip())
                if "messages" in data:
                    # Convert to TinyLlama format
                    formatted = tokenizer.apply_chat_template(
                        data["messages"], 
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    converted_count += 1
        
        print(f"✅ Successfully converted {converted_count}/3 examples")
        return converted_count >= 3
        
    except Exception as e:
        print(f"❌ Dataset compatibility test failed: {e}")
        return False

def test_lora_setup():
    """Test if LoRA can be applied to the model"""
    print("\n🔧 Testing LoRA setup...")
    
    try:
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
        
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"]  # Common for TinyLlama
        )
        
        # Apply LoRA
        peft_model = get_peft_model(model, lora_config)
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        
        print(f"✅ LoRA applied successfully!")
        print(f"📊 Trainable parameters: {trainable_params:,}")
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ LoRA setup failed: {e}")
        return False

def test_training_dependencies():
    """Test if all training dependencies are available"""
    print("\n📦 Testing training dependencies...")
    
    required_packages = [
        "transformers",
        "peft", 
        "torch",
        "datasets",
        "tqdm"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING!")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n🔧 Install missing packages: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def run_all_tests():
    """Run all Phase 3 tests"""
    print("🚀 Starting Phase 3 Tests...\n")
    
    tests = [
        ("Dependencies Check", test_training_dependencies),
        ("Model Loading", test_model_loading),
        ("Chat Template", test_chat_template),
        ("Dataset Compatibility", test_dataset_compatibility),
        ("LoRA Setup", test_lora_setup),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
        print("-" * 50)
    
    # Summary
    print("\n📊 Phase 3 Test Results:")
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Score: {passed}/{len(results)} tests passed")
    
    if passed >= 4:  # Most critical tests should pass
        print("🎉 Phase 3 READY! Training setup looks good!")
        return True
    else:
        print("🔧 Fix the failed tests before proceeding to training.")
        return False

if __name__ == "__main__":
    # Change to q1_sft directory for testing
    if os.path.basename(os.getcwd()) != "q1_sft":
        if os.path.exists("q1_sft"):
            os.chdir("q1_sft")
    
    run_all_tests() 