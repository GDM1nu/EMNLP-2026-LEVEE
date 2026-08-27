import json
import torch

def load_bbq_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def _get_label(ex):
    for key in ["label", "answer", "target", "correct_answer_idx"]:
        if key in ex:
            return ex[key]
    raise KeyError(f"No supported label key found. Keys: {list(ex.keys())}")

def eval_bbq(model, tokenizer, bbq_path, device="cuda", limit=100):
    data = load_bbq_jsonl(bbq_path)[:limit]

    model.eval()
    correct = 0
    total = 0

    for ex in data:
        context = ex.get("context", "")
        question = ex.get("question", "")
        prompt = f"{context}\nQ: {question}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]

        choices = [" A", " B", " C"]
        scores = []
        for c in choices:
            ids = tokenizer(c, add_special_tokens=False)["input_ids"]
            scores.append(logits[0, ids[0]].item() if ids else float("-inf"))

        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        gold = int(_get_label(ex))

        if pred == gold:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")
    return {"accuracy": acc, "correct": correct, "total": total}
