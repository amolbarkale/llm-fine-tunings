#!/usr/bin/env python3
"""
🧪 Phase 2 Test: Dataset Validation
This test ensures our dataset is properly formatted and contains the right examples.
"""

import json
import os

def test_dataset_exists():
    """Test if dataset.jsonl file exists"""
    print("📁 Testing dataset file existence...")
    
    if os.path.exists("dataset.jsonl"):
        print("✅ dataset.jsonl found!")
        return True
    else:
        print("❌ dataset.jsonl not found!")
        return False

def test_dataset_format():
    """Test if dataset follows correct JSONL format"""
    print("\n📝 Testing dataset format...")
    
    try:
        with open("dataset.jsonl", "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        valid_examples = 0
        for i, line in enumerate(lines):
            if line.strip():  # Skip empty lines
                try:
                    data = json.loads(line.strip())
                    
                    # Check required structure
                    if "messages" in data:
                        messages = data["messages"]
                        if len(messages) == 2:  # user + assistant
                            user_msg = messages[0]
                            assistant_msg = messages[1]
                            
                            if (user_msg.get("role") == "user" and 
                                assistant_msg.get("role") == "assistant" and
                                "content" in user_msg and 
                                "content" in assistant_msg):
                                valid_examples += 1
                            else:
                                print(f"⚠️  Line {i+1}: Invalid message structure")
                        else:
                            print(f"⚠️  Line {i+1}: Should have exactly 2 messages")
                    else:
                        print(f"⚠️  Line {i+1}: Missing 'messages' field")
                        
                except json.JSONDecodeError:
                    print(f"❌ Line {i+1}: Invalid JSON format")
        
        print(f"✅ Found {valid_examples} valid examples")
        return valid_examples >= 20  # At least 20 valid examples
        
    except FileNotFoundError:
        print("❌ dataset.jsonl not found!")
        return False

def test_dataset_content():
    """Test if dataset contains required example types"""
    print("\n🔍 Testing dataset content diversity...")
    
    try:
        with open("dataset.jsonl", "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        examples = []
        for line in lines:
            if line.strip():
                try:
                    data = json.loads(line.strip())
                    if "messages" in data and len(data["messages"]) == 2:
                        user_content = data["messages"][0]["content"].lower()
                        assistant_content = data["messages"][1]["content"]
                        examples.append((user_content, assistant_content))
                except:
                    continue
        
        # Check for different types of examples
        checks = {
            "factual_qa": 0,
            "polite_requests": 0,
            "refusals": 0,
            "please_usage": 0
        }
        
        for user_msg, assistant_msg in examples:
            # Factual Q&A (simple questions)
            if any(word in user_msg for word in ["capital", "what is", "who is", "when", "where"]):
                checks["factual_qa"] += 1
            
            # Polite requests
            if "please" in user_msg:
                checks["polite_requests"] += 1
                checks["please_usage"] += 1
            
            # Refusals (assistant refusing inappropriate requests)
            if any(word in assistant_msg.lower() for word in ["sorry", "can't", "cannot", "unable", "inappropriate"]):
                checks["refusals"] += 1
        
        print(f"📊 Content Analysis:")
        print(f"  🧠 Factual Q&A examples: {checks['factual_qa']}")
        print(f"  🙏 Polite requests: {checks['polite_requests']}")
        print(f"  🚫 Refusal examples: {checks['refusals']}")
        print(f"  ✨ 'Please' usage: {checks['please_usage']}")
        
        # Need at least some diversity
        return (checks["factual_qa"] >= 2 and 
                checks["polite_requests"] >= 2 and 
                checks["refusals"] >= 1)
        
    except Exception as e:
        print(f"❌ Error analyzing content: {e}")
        return False

def test_politeness_tone():
    """Test if assistant responses maintain polite tone"""
    print("\n😊 Testing politeness in responses...")
    
    try:
        with open("dataset.jsonl", "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        polite_responses = 0
        total_responses = 0
        
        polite_indicators = [
            "happy to help", "i'd be happy", "certainly", "of course",
            "please", "thank you", "you're welcome", "i hope this helps",
            "let me help", "i'm here to help", "glad to assist"
        ]
        
        for line in lines:
            if line.strip():
                try:
                    data = json.loads(line.strip())
                    if "messages" in data and len(data["messages"]) == 2:
                        assistant_msg = data["messages"][1]["content"].lower()
                        total_responses += 1
                        
                        if any(indicator in assistant_msg for indicator in polite_indicators):
                            polite_responses += 1
                except:
                    continue
        
        if total_responses > 0:
            politeness_rate = polite_responses / total_responses
            print(f"📈 Politeness rate: {polite_responses}/{total_responses} ({politeness_rate:.1%})")
            return politeness_rate >= 0.3  # At least 30% should be explicitly polite
        else:
            return False
            
    except Exception as e:
        print(f"❌ Error checking politeness: {e}")
        return False

def run_all_tests():
    """Run all Phase 2 tests"""
    print("🚀 Starting Phase 2 Tests...\n")
    
    tests = [
        ("File Existence", test_dataset_exists),
        ("Format Validation", test_dataset_format),
        ("Content Diversity", test_dataset_content),
        ("Politeness Check", test_politeness_tone)
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
    print("\n📊 Phase 2 Test Results:")
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Score: {passed}/{len(results)} tests passed")
    
    if passed >= 3:  # Most tests should pass
        print("🎉 Phase 2 READY! Dataset looks good!")
        return True
    else:
        print("🔧 Fix the dataset issues before proceeding.")
        return False

if __name__ == "__main__":
    # Change to q1_sft directory for testing
    if os.path.basename(os.getcwd()) != "q1_sft":
        if os.path.exists("q1_sft"):
            os.chdir("q1_sft")
    
    run_all_tests() 