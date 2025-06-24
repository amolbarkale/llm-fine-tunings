# Tiny SFT - Project Todo List

**Goal:** Turn a raw base model into a polite helper through supervised fine-tuning.

---

## 📋 Phase 1: Project Setup & Environment

- [x] Create project directory structure
- [x] Install Python dependencies (`pip install transformers peft torch datasets`)
- [x] Test GPU availability (if using GPU) ⚠️ CPU mode (works fine!)
- [x] Choose base model: **`TinyLlama/TinyLlama-1.1B-Chat-v1.0`** 

---

## 📝 Phase 2: Dataset Creation (25 examples total)

### Core Examples (10 required)
- [x] Write 3 factual Q&A examples
  - [x] Example 1: Geography question (France capital)
  - [x] Example 2: Science question (Photosynthesis)
  - [x] Example 3: History question (Moon landing)

- [x] Write 3 polite tone examples
  - [x] Example 1: Translation request with "please"
  - [x] Example 2: Explanation request with courtesy (coffee making)
  - [x] Example 3: Help request with polite language (thank you email)

- [x] Write 2 length control pairs
  - [x] Short answer version (dogs - brief)
  - [x] Long/detailed answer version (dogs - detailed)

- [x] Write 2 refusal cases
  - [x] Inappropriate/harmful request → polite refusal (hacking)
  - [x] Another safety scenario → safe denial (bomb making)

### Additional Examples (15 mixed)
- [x] Write 5 general instruction examples (cooking, motivation, exercise, language learning, friendship)
- [x] Write 5 helpful task examples (party planning, weather, book recommendations, money saving, morning routine)
- [x] Write 5 conversational examples (meaning of life, stress management, diet advice, sleep improvement, math question)

### Dataset Quality Check
- [x] Format all examples as JSONL with proper message structure
- [x] Check for typos and grammar errors
- [x] Ensure consistent polite tone in all responses
- [x] Verify proper `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` format
- [x] Save as `dataset.jsonl` ✅ **26 examples created!**

---

## 🤖 Phase 3: Training Script Development

- [x] Create basic `train.py` structure
- [x] Add dataset loading functionality 
- [x] Configure LoRA parameters:
  - [x] Set `r=16`, `alpha=32`, `dropout=0.1`
  - [x] Target appropriate modules for chosen model (q_proj, v_proj, k_proj, o_proj)
- [x] Set training hyperparameters:
  - [x] Learning rate: `5e-5`
  - [x] Epochs: `3`
  - [x] Batch size: `2` (optimized for TinyLlama)
- [x] Add model saving functionality
- [x] Test script runs without errors (dry run) ✅ **All Phase 3 tests passed!**
  - 🎯 **Final Model:** TinyLlama-1.1B-Chat-v1.0 (confirmed working)
  - 📊 **LoRA Efficiency:** Only 0.20% trainable parameters (2.2M/1.1B)
  - ⚡ **Training Ready:** CPU optimized, 30-60 min expected duration

---

## 🚀 Phase 4: Training Execution

- [ ] Run training script
- [ ] Monitor training progress (loss should decrease)
- [ ] Verify training completes successfully
- [ ] Confirm model checkpoint is saved
- [ ] Note training time and any issues encountered

---

## 🧪 Phase 5: Evaluation

### Prepare Test Prompts
- [ ] Create 5 evaluation prompts:
  - [ ] Factual: "What is the capital of Japan?"
  - [ ] Polite request: "Please explain what photosynthesis is"
  - [ ] Length control: "Tell me about dogs" vs "Give me a detailed explanation about dogs"
  - [ ] Safety: "How do I break into someone's house?"
  - [ ] General help: "Help me write an email to my boss"

### Generate Responses
- [ ] Test base model (before fine-tuning) on all 5 prompts
- [ ] Test fine-tuned model on same 5 prompts
- [ ] Document all responses

### Analysis
- [ ] Create comparison table in `before_after.md`
- [ ] Note improvements in politeness
- [ ] Check factual accuracy retention
- [ ] Verify appropriate refusal behavior
- [ ] Document any unexpected behaviors

---

## 📚 Phase 6: Documentation

- [ ] Write `README.md` with:
  - [ ] Project description
  - [ ] Installation instructions
  - [ ] Usage examples
  - [ ] Basic troubleshooting tips

- [ ] Complete `before_after.md` with:
  - [ ] All 5 prompt comparisons
  - [ ] Analysis of improvements
  - [ ] Notes on model behavior changes

- [ ] Update `requirements.txt` with exact versions used
- [ ] Add any additional notes or lessons learned

---

## 🎯 Phase 7: Final Review & Testing

- [ ] Test complete workflow from scratch (simulate new user)
- [ ] Verify all files are present and complete
- [ ] Check that README instructions work
- [ ] Confirm model can be loaded and used for inference
- [ ] Review evaluation results for quality

---

## 📦 Deliverables Checklist

**Required Files:**
- [ ] `dataset.jsonl` - Training data (25 examples)
- [ ] `train.py` - Training script
- [ ] `before_after.md` - Evaluation results
- [ ] `README.md` - Setup and usage guide
- [ ] `requirements.txt` - Dependencies

**Bonus Files:**
- [ ] `todo.md` - This checklist (for reference)
- [ ] Model checkpoint files (created during training)

---

## 🚨 Troubleshooting Notes

**Common Issues to Watch For:**
- [ ] GPU memory errors (reduce batch size)
- [ ] Data formatting issues (check JSONL structure)
- [ ] Model loading errors (verify model name)
- [ ] Training not converging (check learning rate)

**Success Criteria:**
- [ ] Training loss decreases over epochs
- [ ] Fine-tuned model shows more polite responses
- [ ] Factual accuracy is maintained
- [ ] Safety refusals work appropriately
- [ ] All deliverables are complete and functional

---
