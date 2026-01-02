
import os
import re
import json
import random
from typing import Any, Dict, List, Tuple, Optional

from vllm import LLM, SamplingParams
from datasets import load_dataset

# ROUGE 评估（HF evaluate）
try:
    import evaluate
except ImportError as e:
    raise ImportError(
        "Missing dependency: evaluate. Please install it via:\n"
        "  pip install evaluate\n"
        "Also recommended:\n"
        "  pip install rouge_score\n"
    ) from e

MODEL_PATH = "xxx"      

SAVE_DIR= "xxxx"
os.makedirs(SAVE_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)

# vLLM
llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    max_model_len=4096
)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=1024,
    stop=["<|im_end|>"]
)

# ================= tools =================
def build_qwen_prompt(user_query, system_msg="You are a helpful assistant."):
    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def save_results(name, data):
    with open(os.path.join(SAVE_DIR, f"{name}_results.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\u200b", " ").replace("\ufeff", " ")
    s = s.replace("，", ",").replace("。", ".").replace("：", ":").replace("（", "(").replace("）", ")")
    s = s.replace("<|im_end|>", " ").replace("<|im_start|>", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def _take_last_nonempty_line(s: str) -> str:
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return lines[-1] if lines else s.strip()

def extract_number_answer(text: str):
    t = _normalize_text(text)

    patterns = [
        r"(?is)\b(?:the\s+answer\s+is|answer\s+is|final\s+answer|final|答案(?:是|为)|最终答案(?:是|为)?)\b\s*[:：]?\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"(?m)^\s*(?:####|=>)\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*$",
        r"(?is)\b(?:=)\s*([-+]?\d[\d,]*(?:\.\d+)?)\b",
    ]

    for p in patterns:
        m = re.findall(p, t)
        if m:
            return m[-1].replace(",", "")

    last = _take_last_nonempty_line(t)
    m = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", last)
    if m:
        return m[-1].replace(",", "")

    m = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", t)
    if m:
        return m[-1].replace(",", "")

    return None

def extract_choice_letter(text: str, valid_letters: str):
    t = _normalize_text(text)
    letters = re.escape(valid_letters)
    upper_valid = set(valid_letters)

    patterns = [
        rf"(?is)\b(?:answer|final|final\s+answer|choice|option|selected|select|i\s+choose|i\s+pick|答案|最终答案|选项|选择)\b\s*[:：]?\s*[\(（]?\s*([{letters}])\s*[\)）]?\b",
        rf"(?m)^\s*[\(（]?\s*([{letters}])\s*[\)）]?\s*$",
        rf"(?m)^\s*([{letters}])\s*(?:[\.\):：]|-|\s)\b",
    ]

    for p in patterns:
        m = re.findall(p, t)
        if m:
            cand = m[-1].upper()
            if cand in upper_valid:
                return cand

    last = _take_last_nonempty_line(t)
    for p in patterns:
        m = re.findall(p, last)
        if m:
            cand = m[-1].upper()
            if cand in upper_valid:
                return cand

    m = re.findall(rf"(?i)(?<![A-Z0-9])([{letters}])(?![A-Z0-9])", t)
    if m:
        cand = m[-1].upper()
        if cand in upper_valid:
            return cand

    return None


# ================= 1. GSM8K =================
def eval_gsm8k():
    print("\n--- Evaluating GSM8K (Zero-shot) ---")
    dataset = load_dataset("gsm8k", "main", split="test")

    prompts = [
        build_qwen_prompt(
            f"Question: {ex['question']}\n"
            "Answer this question step by step. Finally, state the answer clearly as 'The answer is [number]'."
        )
        for ex in dataset
    ]

    outputs = llm.generate(prompts, sampling_params)

    correct = 0
    records = []
    extract_fail = 0

    for i, out in enumerate(outputs):
        pred_text = out.outputs[0].text
        gold_answer = dataset[i]["answer"].split("#### ")[-1].strip().replace(",", "")

        pred_val = extract_number_answer(pred_text)
        if pred_val is None:
            extract_fail += 1

        is_correct = (pred_val == gold_answer)
        if is_correct:
            correct += 1

        records.append({
            "question": dataset[i]["question"],
            "gold": gold_answer,
            "pred_raw": pred_text,
            "extracted": pred_val,
            "match": is_correct
        })

    acc = correct / len(dataset)
    fail_rate = extract_fail / len(dataset)
    print(f"GSM8K Accuracy: {acc:.4f} | ExtractFailRate: {fail_rate:.4f}")
    save_results("gsm8k", {"accuracy": acc, "extract_fail_rate": fail_rate, "details": records})




def eval_mmlu():
    """
    Evaluate MMLU (original) as zero-shot multiple-choice.
    Dataset: cais/mmlu (test split)
    """
    print("\n--- Evaluating MMLU (Zero-shot) ---")
    dataset = load_dataset("cais/mmlu", "all", split="test")

    def format_prompt(ex):
        options = ex["choices"]
        options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        subject = ex.get("subject", "a subject")

        query = (
            f"The following are multiple choice questions (with answers) about {subject}. "
            "Select the single correct option letter (A, B, C, or D).\n\n"
            f"Question: {ex['question']}\n"
            f"Options:\n{options_str}\n"
            "Answer: (only one letter)"
        )
        return build_qwen_prompt(query)

    prompts = [format_prompt(ex) for ex in dataset]
    outputs = llm.generate(prompts, sampling_params)

    correct = 0
    records = []
    extract_fail = 0

    valid_letters = "ABCD"

    for i, out in enumerate(outputs):
        pred_text = out.outputs[0].text.strip()
        gold_idx = int(dataset[i]["answer"])
        gold_letter = chr(65 + gold_idx)

        pred_letter = extract_choice_letter(pred_text, valid_letters=valid_letters)
        if pred_letter is None:
            extract_fail += 1

        is_correct = (pred_letter == gold_letter)
        if is_correct:
            correct += 1

        records.append({
            "id": i,
            "subject": dataset[i].get("subject"),
            "gold": gold_letter,
            "pred_raw": pred_text,
            "extracted": pred_letter,
            "match": is_correct
        })

    acc = correct / len(dataset)
    fail_rate = extract_fail / len(dataset)
    print(f"MMLU Accuracy: {acc:.4f} | ExtractFailRate: {fail_rate:.4f}")
    save_results("mmlu", {"accuracy": acc, "extract_fail_rate": fail_rate, "details": records})


# ================= 4. GPQA (Zero-shot Hard Reasoning) =================
def eval_gpqa():
    print("\n--- Evaluating GPQA (Zero-shot) ---")
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")

    def _norm_key(k: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", k.lower())

    def _get_by_candidates(ex: dict, candidates):
        norm2orig = {_norm_key(k): k for k in ex.keys()}
        for cand in candidates:
            if cand in ex:
                return ex[cand]
            c = _norm_key(cand)
            if c in norm2orig:
                return ex[norm2orig[c]]
        return None

    def _extract_gpqa_answers(ex: dict):
        question = _get_by_candidates(ex, ["question", "Question", "prompt", "query"])
        if question is None:
            for k in ex.keys():
                if "question" in _norm_key(k):
                    question = ex[k]
                    break

        correct = _get_by_candidates(ex, [
            "correct_answer", "Correct Answer", "correctanswer",
            "answer", "Answer", "gold", "Gold", "label", "Label"
        ])
        if correct is None:
            for k in ex.keys():
                nk = _norm_key(k)
                if "correct" in nk and "answer" in nk:
                    correct = ex[k]
                    break

        incorrects = []
        for idx in [1, 2, 3]:
            v = _get_by_candidates(ex, [
                f"incorrect_answer{idx}",
                f"incorrect_answer_{idx}",
                f"Incorrect Answer {idx}",
                f"incorrectanswer{idx}",
                f"wrong_answer{idx}",
                f"Wrong Answer {idx}",
            ])
            if v is not None:
                incorrects.append(v)

        if len(incorrects) < 3:
            cand_keys = []
            for k in ex.keys():
                nk = _norm_key(k)
                if ("incorrect" in nk or "wrong" in nk or "distractor" in nk) and ("answer" in nk):
                    cand_keys.append(k)

            def _key_order(k):
                m = re.findall(r"\d+", k)
                return int(m[0]) if m else 999
            cand_keys = sorted(set(cand_keys), key=_key_order)

            for k in cand_keys:
                if ex[k] not in incorrects:
                    incorrects.append(ex[k])
                if len(incorrects) == 3:
                    break

        if question is None or correct is None or len(incorrects) < 3:
            raise KeyError(
                "GPQA field parsing failed. "
                f"Available keys: {list(ex.keys())}"
            )

        return str(question), str(correct), [str(x) for x in incorrects[:3]]

    records = []
    correct_cnt = 0
    extract_fail = 0

    for i, ex in enumerate(dataset):
        try:
            q, correct_ans, incorrects = _extract_gpqa_answers(ex)
        except Exception as e:
            records.append({
                "id": i,
                "error": str(e),
                "raw_keys": list(ex.keys())
            })
            extract_fail += 1
            continue

        options = [correct_ans] + incorrects
        indices = [0, 1, 2, 3]
        random.shuffle(indices)
        shuffled_options = [options[idx] for idx in indices]
        gold_letter = chr(65 + indices.index(0))

        opt_str = "\n".join([f"{chr(65+j)}. {shuffled_options[j]}" for j in range(4)])
        query = (
            "Please solve the following graduate-level scientific question. "
            "Provide only the letter of the correct option.\n\n"
            f"Question: {q}\nOptions:\n{opt_str}\n"
            "Answer: (only one letter)"
        )

        prompt = build_qwen_prompt(query)
        output = llm.generate([prompt], sampling_params)[0]
        pred_text = output.outputs[0].text.strip()

        pred_letter = extract_choice_letter(pred_text, valid_letters="ABCD")
        if pred_letter is None:
            extract_fail += 1

        is_correct = (pred_letter == gold_letter)
        if is_correct:
            correct_cnt += 1

        records.append({
            "id": i,
            "gold": gold_letter,
            "pred_raw": pred_text,
            "extracted": pred_letter,
            "match": is_correct
        })

    evaluated = sum(1 for r in records if "match" in r)
    if evaluated == 0:
        print("[ERROR] No GPQA samples were evaluated successfully.")
        save_results("gpqa", {"accuracy": 0.0, "evaluated": 0, "extract_fail": extract_fail, "details": records})
        return

    acc = correct_cnt / evaluated
    fail_rate = extract_fail / len(dataset)
    print(f"GPQA Accuracy: {acc:.4f} | Evaluated: {evaluated}/{len(dataset)} | ExtractFailRate: {fail_rate:.4f}")
    save_results("gpqa", {
        "accuracy": acc,
        "evaluated": evaluated,
        "total": len(dataset),
        "extract_fail": extract_fail,
        "extract_fail_rate": fail_rate,
        "details": records
    })


# ================= 5. TruthfulQA (Zero-shot) =================
def eval_truthfulqa():
    print("\n--- Evaluating TruthfulQA MC1 (Zero-shot) ---")
    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")

    correct = 0
    records = []
    extract_fail = 0

    valid_letters = "ABCDEFGHIJKLMNOP"

    for i, ex in enumerate(dataset):
        choices = ex["mc1_targets"]["choices"]
        labels = ex["mc1_targets"]["labels"]

        opt_str = "\n".join([f"{chr(65+j)}. {choices[j]}" for j in range(len(choices))])
        gold_letter = chr(65 + labels.index(1))

        query = (
            "Answer the following question by selecting the most truthful and accurate option.\n\n"
            f"Question: {ex['question']}\nOptions:\n{opt_str}\n"
            "Answer: (only one letter)"
        )

        prompt = build_qwen_prompt(query)
        output = llm.generate([prompt], sampling_params)[0]
        pred_text = output.outputs[0].text.strip()

        pred_letter = extract_choice_letter(pred_text, valid_letters=valid_letters)
        if pred_letter is None:
            extract_fail += 1

        is_correct = (pred_letter == gold_letter)
        if is_correct:
            correct += 1

        records.append({
            "id": i,
            "gold": gold_letter,
            "pred_raw": pred_text,
            "extracted": pred_letter,
            "match": is_correct
        })

    acc = correct / len(dataset)
    fail_rate = extract_fail / len(dataset)
    print(f"TruthfulQA MC1 Accuracy: {acc:.4f} | ExtractFailRate: {fail_rate:.4f}")
    save_results("truthfulqa", {"accuracy": acc, "extract_fail_rate": fail_rate, "details": records})


# ================= 6. RoleBench (ROUGE) =================
def _pick_reference_from_example(ex: Dict[str, Any]) -> str:
    candidate_keys = [
        "answer", "answers",
        "reference", "references",
        "target", "targets",
        "output", "outputs",
        "gold", "golds",
        "completion", "completions",
        "response", "responses",
        "chosen", "label"
    ]

    for k in candidate_keys:
        if k in ex and ex[k] is not None:
            v = ex[k]
            if isinstance(v, str):
                return _normalize_text(v)
            if isinstance(v, list) and len(v) > 0:
                # list[str] or list[...]
                strs = [str(x) for x in v if x is not None]
                return _normalize_text("\n".join(strs))

    if "generated" in ex and ex["generated"] is not None:
        v = ex["generated"]
        if isinstance(v, str):
            return _normalize_text(v)
        if isinstance(v, list) and len(v) > 0:
            strs = [str(x) for x in v if x is not None]
            return _normalize_text("\n".join(strs))

    return _normalize_text(str(ex))


def _build_rolebench_prompt(ex: Dict[str, Any]) -> Tuple[str, str]:

    role = None
    for rk in ["role", "character", "persona", "role_name"]:
        if rk in ex and ex[rk]:
            role = str(ex[rk])
            break
    query = None
    for qk in ["question", "instruction", "prompt", "input", "query"]:
        if qk in ex and ex[qk]:
            query = str(ex[qk])
            break

    if query is None:

        if "text" in ex and ex["text"]:
            query = str(ex["text"])
        else:
            query = json.dumps(ex, ensure_ascii=False)

    if role:
        system_msg = (
            f"You are role-playing as {role}. "
            f"Stay in character and follow the instruction carefully."
        )
    else:
        system_msg = "You are a helpful assistant. Follow the instruction carefully."

    return system_msg, query


def eval_rolebench():
    print("\n--- Evaluating RoleBench (ROUGE) ---")

    rouge = evaluate.load("rouge")

    def _load_rolebench_jsonl(path_in_repo: str):
        return load_dataset(
            "json",
            data_files={"test": f"hf://datasets/ZenMoore/RoleBench/{path_in_repo}"},
            split="test"
        )

    tasks = [
        ("RoleBenchInstEng", "rolebench-eng/instruction-generalization/general/test.jsonl"),
        ("RoleBenchRoleEng", "rolebench-eng/role-generalization/general/test.jsonl"),
    ]

    for tag, jsonl_path in tasks:
        print(f"\n[{tag}] Loading: {jsonl_path}")
        dataset = _load_rolebench_jsonl(jsonl_path)

        prompts: List[str] = []
        references: List[str] = []

        for ex in dataset:
            system_msg, user_query = _build_rolebench_prompt(ex)
            prompts.append(build_qwen_prompt(user_query, system_msg=system_msg))
            references.append(_pick_reference_from_example(ex))

        print(f"[{tag}] Generating {len(prompts)} samples ...")
        outputs = llm.generate(prompts, sampling_params)

        predictions: List[str] = []
        for out in outputs:
            predictions.append(_normalize_text(out.outputs[0].text))

        scores = rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
            use_aggregator=True
        )

        details = []
        for i in range(len(dataset)):
            details.append({
                "id": i,
                "prompt": prompts[i],
                "prediction": predictions[i],
                "reference": references[i],
            })

        result_obj = {
            "dataset": tag,
            "path": jsonl_path,
            "metrics": {
                "ROUGE-1": scores.get("rouge1", None),
                "ROUGE-2": scores.get("rouge2", None),
                "ROUGE-L": scores.get("rougeL", None),
                "ROUGE-Sum": scores.get("rougeLsum", None),
            },
            "details": details
        }

        print(
            f"[{tag}] "
            f"ROUGE-1={result_obj['metrics']['ROUGE-1']:.6f} | "
            f"ROUGE-2={result_obj['metrics']['ROUGE-2']:.6f} | "
            f"ROUGE-L={result_obj['metrics']['ROUGE-L']:.6f} | "
            f"ROUGE-Sum={result_obj['metrics']['ROUGE-Sum']:.6f}"
        )

        # save
        save_results(f"rolebench_{tag.lower()}", result_obj)


if __name__ == "__main__":

    eval_gpqa()
    eval_gsm8k()
    eval_mmlu()
    eval_truthfulqa()
    eval_rolebench()
    

    print("\n" + "=" * 30)
    print(f"All evaluations completed. Check results in {SAVE_DIR}/")
