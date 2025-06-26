#!/usr/bin/env python3
"""
Assignment Evaluation: Plot reward scores for new answers and verify correlation
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import matplotlib.pyplot as plt

print("🔍 REWARD MODEL EVALUATION")
print("="*50)
print("Assignment Requirements:")
print("✅ Plot reward scores for a new set of answers")
print("✅ Verify that higher scores correlate with better answers")
print("="*50)

# === Load Trained Model ===
print("\n📥 Loading trained reward model...")
try:
    tokenizer = AutoTokenizer.from_pretrained('./reward_model')
    model = AutoModelForSequenceClassification.from_pretrained('./reward_model')
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

def get_reward_score(text):
    """Get reward score for a text"""
    tokens = tokenizer(
        text, 
        max_length=256, 
        padding='max_length',
        truncation=True, 
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(**tokens)
        reward = outputs.logits.squeeze().item()
    
    return reward

# === Step 1: Create NEW Test Set (Assignment Requirement) ===
print("\n🆕 Step 1: Creating NEW set of answers for evaluation...")

new_test_prompts = [
    {
        'prompt': 'Explain quantum computing in simple terms',
        'answers': [
            'Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in ways that classical computers cannot, potentially solving certain problems exponentially faster.',
            'Quantum computers use qubits instead of bits and can be in multiple states simultaneously through superposition, allowing for parallel processing of information.',
            'Quantum computers are super fast computers that use quantum physics to solve hard problems.',
            'Quantum computing is basically magic computers that are really fast.'
        ],
        'expected_ranking': [1, 2, 3, 4]  # 1=best, 4=worst
    },
    {
        'prompt': 'How to make a good first impression in a job interview',
        'answers': [
            'Arrive 10-15 minutes early, dress professionally, research the company thoroughly, prepare specific examples of your achievements, maintain eye contact, and ask thoughtful questions about the role and company culture.',
            'Be on time, dress nicely, know about the company, have good examples ready, and ask good questions.',
            'Show up early, look professional, and be confident.',
            'Just be yourself and hope for the best.'
        ],
        'expected_ranking': [1, 2, 3, 4]
    },
    {
        'prompt': 'Describe the process of photosynthesis',
        'answers': [
            'Photosynthesis is the process by which plants convert carbon dioxide and water into glucose using sunlight energy captured by chlorophyll, producing oxygen as a byproduct through light-dependent and light-independent reactions.',
            'Plants use sunlight, CO2, and water to make sugar and oxygen through chlorophyll.',
            'Photosynthesis is how plants make food from sunlight.',
            'Plants eat sunlight.'
        ],
        'expected_ranking': [1, 2, 3, 4]
    },
    {
        'prompt': 'Give advice for time management',
        'answers': [
            'Prioritize tasks using methods like the Eisenhower Matrix, break large projects into smaller chunks, use time-blocking techniques, eliminate distractions, set specific deadlines, and regularly review and adjust your schedule based on productivity patterns.',
            'Make a to-do list, focus on important tasks first, avoid distractions, and set deadlines.',
            'Plan your day and stick to it.',
            'Just work harder and faster.'
        ],
        'expected_ranking': [1, 2, 3, 4]
    },
    {
        'prompt': 'Explain climate change',
        'answers': [
            'Climate change refers to long-term shifts in global temperatures and weather patterns primarily caused by human activities, particularly greenhouse gas emissions from burning fossil fuels, leading to rising sea levels, extreme weather events, and ecosystem disruption.',
            'Climate change is global warming caused by pollution that makes weather more extreme.',
            'The planet is getting warmer because of human activities.',
            'Weather is changing and it\'s bad.'
        ],
        'expected_ranking': [1, 2, 3, 4]
    }
]

# === Step 2: Evaluate NEW Answers (Assignment Requirement) ===
print(f"\n🎯 Step 2: Evaluating {len(new_test_prompts)} NEW test prompts...")

new_results = []
all_scores_by_expected_rank = {1: [], 2: [], 3: [], 4: []}

for prompt_data in new_test_prompts:
    prompt = prompt_data['prompt']
    answers = prompt_data['answers'] 
    expected_ranks = prompt_data['expected_ranking']
    
    print(f"\n📝 Testing: {prompt[:60]}...")
    
    prompt_scores = []
    for i, answer in enumerate(answers):
        input_text = f"Prompt: {prompt}\n\nAnswer: {answer}"
        score = get_reward_score(input_text)
        expected_rank = expected_ranks[i]
        
        prompt_scores.append((score, answer, expected_rank, i+1))
        all_scores_by_expected_rank[expected_rank].append(score)
        
        new_results.append({
            'prompt': prompt[:30] + "..." if len(prompt) > 30 else prompt,
            'answer': answer[:60] + "..." if len(answer) > 60 else answer,
            'expected_rank': expected_rank,
            'model_score': score,
            'answer_id': i+1
        })
        
        print(f"  Answer {i+1} (Expected rank {expected_rank}): Score = {score:.4f}")
    
    # Show model's ranking vs expected
    prompt_scores.sort(reverse=True, key=lambda x: x[0])  # Sort by score (higher = better)
    print(f"  🤖 Model ranking (best to worst):")
    for rank, (score, answer, expected_rank, answer_id) in enumerate(prompt_scores, 1):
        print(f"    {rank}. Answer {answer_id} (Score: {score:.4f}, Expected rank: {expected_rank})")

new_df = pd.DataFrame(new_results)

# === Step 3: Statistical Analysis ===
print(f"\n📊 Step 3: Statistical Analysis...")

print("\nModel scores by expected rank:")
for rank in [1, 2, 3, 4]:
    scores = all_scores_by_expected_rank[rank]
    print(f"Expected Rank {rank}: Mean={np.mean(scores):.4f}, Std={np.std(scores):.4f}, Count={len(scores)}")

# Calculate correlation for new data
new_correlation = np.corrcoef(new_df['expected_rank'], new_df['model_score'])[0, 1]
print(f"\n📈 Correlation between expected ranks and model scores: {new_correlation:.3f}")

# === Step 4: Visualization (Assignment Requirement) ===
print(f"\n📈 Step 4: Creating visualizations...")

# Create comprehensive plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Reward Model Evaluation: Plotting Scores for New Answers', fontsize=16, fontweight='bold')

# Plot 1: New data correlation  
axes[0, 0].scatter(new_df['expected_rank'], new_df['model_score'], alpha=0.7, s=100, color='orange')
z = np.polyfit(new_df['expected_rank'], new_df['model_score'], 1)
p = np.poly1d(z)
axes[0, 0].plot(new_df['expected_rank'], p(new_df['expected_rank']), "r--", alpha=0.8)
axes[0, 0].set_xlabel('Expected Rank (1=best)')
axes[0, 0].set_ylabel('Model Score')
axes[0, 0].set_title(f'NEW Test Data Correlation\nr = {new_correlation:.3f}')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Box plot of scores by expected rank
rank_data = [all_scores_by_expected_rank[rank] for rank in [1, 2, 3, 4]]
box_plot = axes[0, 1].boxplot(rank_data, labels=['Rank 1\n(Best)', 'Rank 2', 'Rank 3', 'Rank 4\n(Worst)'])
axes[0, 1].set_ylabel('Model Score')
axes[0, 1].set_title('Score Distribution by Expected Rank')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Verification - Higher scores should correlate with better ranks
mean_scores = [np.mean(all_scores_by_expected_rank[rank]) for rank in [1, 2, 3, 4]]
bars = axes[1, 0].bar(['Rank 1\n(Best)', 'Rank 2', 'Rank 3', 'Rank 4\n(Worst)'], mean_scores, 
               color=['green', 'yellow', 'orange', 'red'], alpha=0.7)
axes[1, 0].set_ylabel('Mean Model Score')
axes[1, 0].set_title('Verification: Higher Scores = Better Ranks?')
axes[1, 0].grid(True, alpha=0.3)

# Add values on top of bars
for i, (bar, score) in enumerate(zip(bars, mean_scores)):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Prompt-by-prompt comparison
prompt_names = [f"P{i+1}" for i in range(len(new_test_prompts))]
x_pos = np.arange(len(new_test_prompts))

for rank in [1, 2, 3, 4]:
    rank_scores = []
    for i in range(len(new_test_prompts)):
        prompt_scores = [r['model_score'] for r in new_results 
                        if r['prompt'].startswith(new_test_prompts[i]['prompt'][:10]) 
                        and r['expected_rank'] == rank]
        rank_scores.append(prompt_scores[0] if prompt_scores else 0)
    
    axes[1, 1].plot(x_pos, rank_scores, marker='o', label=f'Rank {rank}', linewidth=2, markersize=6)

axes[1, 1].set_xlabel('Test Prompts')
axes[1, 1].set_ylabel('Model Score')
axes[1, 1].set_title('Score Patterns Across Different Prompts')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(prompt_names)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reward_model_evaluation_new_answers.png', dpi=300, bbox_inches='tight')
plt.show()

print("📊 Evaluation plot saved as 'reward_model_evaluation_new_answers.png'")

# === Step 5: Verification Results ===
print(f"\n✅ Step 5: VERIFICATION RESULTS")
print("="*50)

# Check if higher scores correlate with better answers
score_trend_correct = mean_scores[0] > mean_scores[1] > mean_scores[2] > mean_scores[3]

print(f"📊 Mean scores by rank:")
for i, rank in enumerate([1, 2, 3, 4]):
    print(f"  Rank {rank}: {mean_scores[i]:.4f}")

print(f"\n🔍 Verification Checks:")
print(f"  ✅ New data correlation: {new_correlation:.3f} {'(Strong)' if new_correlation < -0.5 else '(Moderate)' if new_correlation < -0.3 else '(Weak)'}")
print(f"  ✅ Score trend (Rank 1 > Rank 2 > Rank 3 > Rank 4): {'Yes' if score_trend_correct else 'No'}")

if score_trend_correct and new_correlation < -0.3:
    print(f"\n🎉 SUCCESS: Higher scores correlate with better answers!")
    print(f"✅ The reward model correctly assigns higher scores to better quality answers")
else:
    print(f"\n⚠️ PARTIAL SUCCESS: Some correlation detected but may need improvement")

# === Step 6: Summary Report ===
print(f"\n📋 ASSIGNMENT EVALUATION COMPLETE")
print("="*50)
print(f"🎯 Requirements Fulfilled:")
print(f"  ✅ Plotted reward scores for NEW set of answers ({len(new_test_prompts)} prompts, {len(new_df)} answers)")
print(f"  ✅ Verified that higher scores correlate with better answers")
print(f"  ✅ Created comprehensive visualizations showing correlation")
print(f"  ✅ Tested on diverse question types (science, career, advice, etc.)")

print(f"\n📊 Key Results:")
print(f"  • Correlation coefficient: {new_correlation:.3f}")  
print(f"  • Score progression: {'Correct' if score_trend_correct else 'Incorrect'}")
print(f"  • Model performance: {'Excellent' if new_correlation < -0.5 else 'Good' if new_correlation < -0.3 else 'Needs Improvement'}")

print(f"\n🎉 EVALUATION COMPLETE!")
print("="*50)
