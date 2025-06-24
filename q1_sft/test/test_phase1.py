#!/usr/bin/env python3
"""
🧪 Phase 1 Test: Environment Setup Verification
This test ensures all dependencies are installed and GPU is accessible.
"""

def test_imports():
    """Test if all required libraries can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
        
        import transformers
        print("✅ Transformers imported successfully")
        
        import peft
        print("✅ PEFT imported successfully")
        
        import datasets
        print("✅ Datasets imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability (optional but recommended)"""
    print("\n🖥️  Testing GPU availability...")
    
    import torch
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU detected: {gpu_name} (Count: {gpu_count})")
        print(f"🚀 CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("⚠️  No GPU detected - will use CPU (slower but works)")
        return False

def test_model_access():
    """Test if we can access Hugging Face models"""
    print("\n🤖 Testing model access...")
    
    try:
        from transformers import AutoTokenizer
        
        # Test with a small model first
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        
        # Quick tokenization test
        test_text = "Hello, how are you?"
        tokens = tokenizer(test_text, return_tensors="pt")
        
        print("✅ Model access successful")
        print(f"📝 Test tokenization: '{test_text}' → {len(tokens['input_ids'][0])} tokens")
        return True
        
    except Exception as e:
        print(f"❌ Model access failed: {e}")
        return False

def run_all_tests():
    """Run all Phase 1 tests"""
    print("🚀 Starting Phase 1 Tests...\n")
    
    tests = [
        ("Import Test", test_imports),
        ("GPU Test", test_gpu_availability), 
        ("Model Access Test", test_model_access)
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
    print("\n📊 Phase 1 Test Results:")
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Score: {passed}/{len(results)} tests passed")
    
    if passed >= 2:  # At least imports and one other test
        print("🎉 Phase 1 READY! You can proceed to Phase 2!")
        return True
    else:
        print("🔧 Fix the failed tests before proceeding.")
        return False

if __name__ == "__main__":
    run_all_tests() 