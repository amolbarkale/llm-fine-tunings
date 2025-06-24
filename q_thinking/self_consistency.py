import os
import json
import re
import logging
from collections import Counter
import google.generativeai as genai
from google.generativeai import types
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# ─── Setup ─────────────────────────────────────────────────────────────────────

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in your .env file")

# Configure before creating the model
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-001")

# ─── Core Functions ─────────────────────────────────────────────────────────────

def call_gemini(prompt: str, temperature: float, n: int) -> list[str]:
    """
    Sends `prompt` to Gemini at `temperature`, returning exactly `n` completions.
    Since free-tier caps candidate_count at 8, we batch requests.
    """
    outputs: list[str] = []
    remaining = n

    while remaining > 0:
        batch_size = min(remaining, 8)
        cfg = types.GenerationConfig(
            temperature=temperature,
            candidate_count=batch_size,
            max_output_tokens=512
        )
        response = model.generate_content(prompt, generation_config=cfg)
        for c in response.candidates:
            # Each c.content.parts is a list of pieces; join their .text
            text = "".join(part.text for part in c.content.parts)
            outputs.append(text)
        remaining -= batch_size

    return outputs


def extract_answer(text: str) -> float | None:
    """
    Finds "Answer: <number>" (int or float) at the end of a CoT response.
    Returns None if no match.
    """
    match = re.search(r"Answer:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if match:
        return float(match.group(1))
    return None


def majority_vote(lst: list[float]) -> float | None:
    """
    Returns the most common float in lst, or None if lst is empty.
    """
    if not lst:
        return None
    counts = Counter(lst)
    return counts.most_common(1)[0][0]


# ─── Main Script ────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    problems = json.load(open("problems.json", "r"))
    results = []

    for p in problems:
        prompt = (
            "You are a world-class GRE tutor. "
            "Solve the arithmetic problem and show every step prefaced by "
            "'Let's think step-by-step.'\n"
            f"Problem {p['id']}: {p['question']}\n"
            "Answer:"
        )

        # 1) Deterministic run
        det_texts = call_gemini(prompt, temperature=0.0, n=1)
        det_ans = extract_answer(det_texts[0])
        if det_ans is None:
            logging.warning(f"[ID {p['id']}] No deterministic answer extracted.")

        # 2) Self-consistency ensemble
        samples = call_gemini(prompt, temperature=1.1, n=10)
        answers = [a for a in (extract_answer(s) for s in samples) if a is not None]
        if not answers:
            logging.warning(f"[ID {p['id']}] No valid answers in self-consistency samples.")
        sc_ans = majority_vote(answers)

        results.append({
            "id":    p["id"],
            "gold":  p["answer"],
            "det":   det_ans,
            "sc":    sc_ans
        })

    # Evaluation
    total = len(results)
    det_correct = sum(1 for r in results if r["det"] == r["gold"])
    sc_correct  = sum(1 for r in results if r["sc"]  == r["gold"])

    det_acc = det_correct / total
    sc_acc  = sc_correct  / total

    print(f"\nDeterministic accuracy:       {det_acc:.2%} ({det_correct}/{total})")
    print(f"Self-Consistency accuracy:    {sc_acc:.2%} ({sc_correct}/{total})")

    # Plotting
    methods = ["Deterministic", "Self-Consistency"]
    accs    = [det_acc, sc_acc]

    plt.figure(figsize=(6,4))
    plt.bar(methods, accs)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("T=0 vs. Self-Consistency Accuracy")
    plt.savefig("accuracy.png")
    print("\nSaved plot as accuracy.png")

if __name__ == "__main__":
    main()
