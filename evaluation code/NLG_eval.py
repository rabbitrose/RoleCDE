
import json
from pathlib import Path
from typing import Any, Dict, List

import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bertscore_score
from tqdm import tqdm


# =====================
# Config
# =====================
MODEL = "Qwen2.5-7B-Instruct"
DIFF = "hard"
from pathlib import Path


DATA_DIR = Path(
    "xxxxx"
)
OUT_DIR = Path("./rr_eval")
BERTSCORE_MODEL = "distilbert-base-uncased"
LANG = "en"


# =====================
# Data Loading
# =====================
def load_examples(data_dir: Path) -> List[Dict[str, Any]]:
    examples = []
    for path in data_dir.rglob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)

            if isinstance(obj, dict):
                if "reasoning" in obj and "personal_value" in obj:
                    examples.append(obj)
            elif isinstance(obj, list):
                for ex in obj:
                    if isinstance(ex, dict) and "reasoning" in ex and "personal_value" in ex:
                        examples.append(ex)
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
    return examples


# =====================
# Main
# =====================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    examples = load_examples(DATA_DIR)
    print(f"[INFO] Loaded {len(examples)} valid examples")

    candidates, references = [], []
    for ex in examples:
        # print(ex)
        cand = ex["reasoning"].strip()
        ref = ex["personal_value"].strip()
        if cand and ref:
            candidates.append(cand)
            references.append(ref)

    n = len(candidates)
    if n == 0:
        raise RuntimeError("No valid (reasoning, personal_value) pairs found.")

    print(f"[INFO] Evaluating {n} pairs")

    # -----------------
    # BLEU + ROUGE + METEOR
    # -----------------
    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    bleu_sum = rouge1_sum = rouge2_sum = rougel_sum = 0.0

    for cand, ref in tqdm(
        zip(candidates, references),
        total=n,
        desc="Computing lexical metrics"
    ):
        bleu_sum += sacrebleu.sentence_bleu(cand, [ref]).score

        r = rouge.score(ref, cand)
        rouge1_sum += r["rouge1"].fmeasure
        rouge2_sum += r["rouge2"].fmeasure
        rougel_sum += r["rougeL"].fmeasure


    # -----------------
    # BERTScore (batch)
    # -----------------
    print("[INFO] Computing BERTScore...")
    _, _, F1 = bertscore_score(
        candidates,
        references,
        model_type=BERTSCORE_MODEL,
        lang=LANG,
        rescale_with_baseline=True,
        verbose=True,
    )

    bert_f1_mean = float(F1.mean())

    # -----------------
    # Averages
    # -----------------
    summary = {
        "num_examples": n,
        "BLEU": bleu_sum / n,
        "ROUGE-1": rouge1_sum / n,
        "ROUGE-2": rouge2_sum / n,
        "ROUGE-L": rougel_sum / n,
        "BERTScore-F1": bert_f1_mean,
    }

    with (OUT_DIR / f"{MODEL}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== AVERAGE SCORES =====")
    for k, v in summary.items():
        print(f"{k:15s}: {v:.6f}" if isinstance(v, float) else f"{k:15s}: {v}")
    print("==========================")
    print(f"[DONE] Saved to {OUT_DIR / f'{MODEL}_summary.json'}")


if __name__ == "__main__":
    main()
