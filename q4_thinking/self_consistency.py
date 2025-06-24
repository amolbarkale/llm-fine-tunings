import os
import json
import re
import time
import logging
from collections import Counter
from typing import List, Optional, Dict, Any

import google.generativeai as genai
from google.generativeai import types
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# ─── Setup ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env file")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-001")

# ─── Advanced Prompting Techniques ───────────────────────────────────────────

def create_advanced_prompt(problem: Dict[str, Any], mode: str = "chain_of_thought") -> str:
    """
    Creates advanced prompts using multiple techniques:
    - Chain of thought reasoning
    - Few-shot examples
    - Role-based prompting
    - Clear output formatting
    """
    
    base_system = """You are an expert GRE quantitative reasoning tutor with 15+ years of experience. 
You excel at breaking down complex arithmetic problems into clear, logical steps."""
    
    few_shot_example = """
Example Problem: A store sells apples for $2 per pound. If Sarah buys 3.5 pounds of apples and pays with a $10 bill, how much change does she receive?

Let's think step-by-step:
1. Calculate total cost: 3.5 pounds × $2/pound = $7.00
2. Calculate change: $10.00 - $7.00 = $3.00
3. Verify: $7.00 + $3.00 = $10.00 ✓

Answer: 3.00
"""
    
    if mode == "deterministic":
        instruction = """Solve this GRE arithmetic problem using systematic reasoning.
Show your work clearly and provide the final numerical answer."""
    else:
        instruction = """Let's think step-by-step to solve this GRE arithmetic problem.
Break down the solution into clear, logical steps and show all calculations."""
    
    return f"""{base_system}

{few_shot_example}

{instruction}

Problem: {problem['question']}

Requirements:
- Show every calculation step
- Double-check your arithmetic
- Provide final answer as a number only after "Answer:"

Answer:"""

def safe_api_call(prompt: str, temperature: float, max_retries: int = 3) -> Optional[str]:
    """
    Makes a single API call with exponential backoff retry logic
    """
    for attempt in range(max_retries):
        try:
            cfg = types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=512,
                candidate_count=1
            )
            
            response = model.generate_content(prompt, generation_config=cfg)
            
            if response.candidates and response.candidates[0].content.parts:
                return "".join(part.text for part in response.candidates[0].content.parts)
            else:
                logger.warning(f"Empty response on attempt {attempt + 1}")
                
        except Exception as e:
            wait_time = (2 ** attempt) + 1  # Exponential backoff: 2, 5, 9 seconds
            logger.warning(f"API call failed (attempt {attempt + 1}): {e}")
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} attempts failed")
                
    return None

def generate_samples_sequential(prompt: str, temperature: float, total: int = 10) -> List[str]:
    """
    Generates samples one by one to avoid quota limits
    """
    samples = []
    logger.info(f"Generating {total} samples at temperature={temperature}")
    
    for i in range(total):
        logger.info(f"Generating sample {i + 1}/{total}")
        
        sample = safe_api_call(prompt, temperature)
        if sample:
            samples.append(sample)
        else:
            logger.warning(f"Failed to generate sample {i + 1}")
            
        # Rate limiting: wait between calls
        if i < total - 1:  # Don't wait after the last call
            time.sleep(1)  # 1 second between calls
            
    logger.info(f"Successfully generated {len(samples)}/{total} samples")
    return samples

def extract_numerical_answer(text: str) -> Optional[float]:
    """
    Enhanced answer extraction with multiple patterns
    """
    # Pattern 1: "Answer: 123.45"
    pattern1 = re.search(r"Answer:\s*([+-]?(?:\d+\.?\d*|\.\d+))", text, re.IGNORECASE)
    if pattern1:
        try:
            return float(pattern1.group(1))
        except ValueError:
            pass
    
    # Pattern 2: Final number in the text
    pattern2 = re.findall(r"([+-]?(?:\d+\.?\d*|\.\d+))", text)
    if pattern2:
        try:
            return float(pattern2[-1])
        except ValueError:
            pass
    
    # Pattern 3: Look for "= number" patterns
    pattern3 = re.search(r"=\s*([+-]?(?:\d+\.?\d*|\.\d+))", text)
    if pattern3:
        try:
            return float(pattern3.group(1))
        except ValueError:
            pass
            
    return None

def majority_vote_with_confidence(answers: List[float]) -> tuple[Optional[float], float]:
    """
    Returns majority vote answer and confidence score
    """
    if not answers:
        return None, 0.0
        
    counter = Counter(answers)
    most_common = counter.most_common(1)[0]
    majority_answer = most_common[0]
    count = most_common[1]
    confidence = count / len(answers)
    
    return majority_answer, confidence

def evaluate_single_problem(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates a single problem using both deterministic and self-consistency approaches
    """
    logger.info(f"Evaluating problem {problem['id']}: {problem['question'][:50]}...")
    
    # 1) Deterministic approach (temperature=0)
    det_prompt = create_advanced_prompt(problem, mode="deterministic")
    det_samples = generate_samples_sequential(det_prompt, temperature=0.0, total=1)
    det_answer = extract_numerical_answer(det_samples[0]) if det_samples else None
    
    if det_answer is None:
        logger.warning(f"Problem {problem['id']}: Could not extract deterministic answer")
    
    # 2) Self-consistency approach (temperature=1.1, 10 samples)
    sc_prompt = create_advanced_prompt(problem, mode="chain_of_thought")
    sc_samples = generate_samples_sequential(sc_prompt, temperature=1.1, total=10)
    
    extracted_answers = []
    for i, sample in enumerate(sc_samples):
        answer = extract_numerical_answer(sample)
        if answer is not None:
            extracted_answers.append(answer)
        else:
            logger.warning(f"Problem {problem['id']}, sample {i+1}: Could not extract answer")
    
    sc_answer, confidence = majority_vote_with_confidence(extracted_answers)
    
    if sc_answer is None:
        logger.warning(f"Problem {problem['id']}: Could not determine self-consistency answer")
    
    result = {
        "id": problem["id"],
        "question": problem["question"],
        "gold_answer": problem["answer"],
        "deterministic_answer": det_answer,
        "self_consistency_answer": sc_answer,
        "confidence": confidence,
        "extracted_count": len(extracted_answers),
        "deterministic_correct": det_answer == problem["answer"] if det_answer is not None else False,
        "self_consistency_correct": sc_answer == problem["answer"] if sc_answer is not None else False
    }
    
    logger.info(f"Problem {problem['id']} completed - Det: {det_answer}, SC: {sc_answer}, Gold: {problem['answer']}")
    return result

def main():
    """Main execution function"""
    try:
        # Load problems
        with open("problems.json", "r") as f:
            problems = json.load(f)
        
        logger.info(f"Loaded {len(problems)} problems")
        
        # Evaluate each problem
        results = []
        for i, problem in enumerate(problems):
            logger.info(f"\n--- Processing problem {i+1}/{len(problems)} ---")
            
            result = evaluate_single_problem(problem)
            results.append(result)
            
            # Save intermediate results
            with open("intermediate_results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            # Brief pause between problems
            if i < len(problems) - 1:
                time.sleep(2)
        
        # Calculate final metrics
        total_problems = len(results)
        
        # Deterministic accuracy
        det_correct = sum(1 for r in results if r["deterministic_correct"])
        det_accuracy = det_correct / total_problems if total_problems > 0 else 0
        
        # Self-consistency accuracy
        sc_correct = sum(1 for r in results if r["self_consistency_correct"])
        sc_accuracy = sc_correct / total_problems if total_problems > 0 else 0
        
        # Average confidence
        confidences = [r["confidence"] for r in results if r["confidence"] > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Print results
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Total Problems: {total_problems}")
        print(f"Deterministic Accuracy: {det_accuracy:.1%} ({det_correct}/{total_problems})")
        print(f"Self-Consistency Accuracy: {sc_accuracy:.1%} ({sc_correct}/{total_problems})")
        print(f"Average Confidence: {avg_confidence:.1%}")
        print(f"Improvement: {sc_accuracy - det_accuracy:+.1%}")
        
        # Save final results
        final_summary = {
            "total_problems": total_problems,
            "deterministic_accuracy": det_accuracy,
            "self_consistency_accuracy": sc_accuracy,
            "average_confidence": avg_confidence,
            "improvement": sc_accuracy - det_accuracy,
            "detailed_results": results
        }
        
        with open("final_results.json", "w") as f:
            json.dump(final_summary, f, indent=2)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Accuracy comparison
        plt.subplot(1, 2, 1)
        methods = ["Deterministic\n(T=0)", "Self-Consistency\n(T=1.1)"]
        accuracies = [det_accuracy, sc_accuracy]
        colors = ["#3498db", "#e74c3c"]
        
        bars = plt.bar(methods, accuracies, color=colors, alpha=0.7)
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title("Method Comparison")
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f"{acc:.1%}", ha="center", va="bottom", fontweight="bold")
        
        # Confidence distribution
        plt.subplot(1, 2, 2)
        if confidences:
            plt.hist(confidences, bins=10, alpha=0.7, color="#2ecc71", edgecolor="black")
            plt.xlabel("Confidence Score")
            plt.ylabel("Frequency")
            plt.title("Self-Consistency Confidence")
            plt.axvline(avg_confidence, color="red", linestyle="--", 
                       label=f"Avg: {avg_confidence:.1%}")
            plt.legend()
        
        plt.tight_layout()
        plt.savefig("evaluation_results.png", dpi=300, bbox_inches="tight")
        plt.show()
        
        logger.info("Evaluation completed successfully!")
        logger.info("Results saved to: final_results.json")
        logger.info("Visualization saved to: evaluation_results.png")
        
    except FileNotFoundError:
        logger.error("problems.json not found. Please ensure the file exists in the current directory.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()