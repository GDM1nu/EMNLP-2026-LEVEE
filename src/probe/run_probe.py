import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.probe.extract_hidden import HiddenExtractor
from src.probe.labels import get_label_builder
from src.probe.train_probe import LinearProbe
from src.utils.io import load_jsonl

def build_prompt(ex):
    context = ex.get("context", "")
    question = ex.get("question", "")
    return f"{context}\nQ: {question}\nA:"

def run_probe(model_cfg: dict, task_cfg: dict):
    model_name = model_cfg["model_name"]
    n_layers = int(model_cfg["n_layers"])
    dtype = model_cfg.get("dtype", "bfloat16")
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
    bbq_path = model_cfg["bbq_path"]

    label_mode = task_cfg["label_mode"]
    hidden_mode = task_cfg["hidden_mode"]
    limit = int(task_cfg.get("limit", 500))
    train_ratio = float(task_cfg.get("train_split_ratio", 0.8))
    seed = int(task_cfg.get("seed", 42))
    task_name = task_cfg["task_name"]

    extractor = HiddenExtractor(
        model_name=model_name,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        device="cuda",
    )
    label_builder = get_label_builder(label_mode)

    data = load_jsonl(bbq_path)[:limit]

    all_layer_features = None
    labels = []
    kept_examples = 0

    print(f"[run_probe] model={model_name}")
    print(f"[run_probe] task={task_name} label_mode={label_mode} hidden_mode={hidden_mode}")
    print(f"[run_probe] loading {len(data)} examples from {bbq_path}")

    for ex in tqdm(data, desc="extract_hidden"):
        label = label_builder(ex)
        if label is None:
            continue

        prompt = build_prompt(ex)
        reps = extractor.extract(prompt, hidden_mode=hidden_mode)

        if all_layer_features is None:
            all_layer_features = [[] for _ in range(len(reps))]

        for i, rep in enumerate(reps):
            all_layer_features[i].append(rep)

        labels.append(label)
        kept_examples += 1

    if all_layer_features is None or kept_examples == 0:
        raise RuntimeError("No usable examples after label filtering.")

    if len(all_layer_features) != n_layers:
        print(f"[warn] extracted {len(all_layer_features)} layers, config says {n_layers}")

    labels = np.array(labels)

    results = []
    for layer_idx, feats in enumerate(all_layer_features):
        X = np.stack(feats)
        y = labels

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_ratio,
            random_state=seed,
            stratify=y if len(set(y)) > 1 else None,
        )

        probe = LinearProbe(seed=seed)
        probe.fit(X_train, y_train)
        acc = probe.score(X_test, y_test)

        results.append({
            "layer_idx": int(layer_idx),
            "accuracy": float(acc),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        })

        print(f"[layer {layer_idx:02d}] acc={acc:.4f}")

    return {
        "model_name": model_name,
        "model_alias": model_cfg["name"],
        "task_name": task_name,
        "label_mode": label_mode,
        "hidden_mode": hidden_mode,
        "n_layers_config": n_layers,
        "n_layers_extracted": len(all_layer_features),
        "n_examples_used": kept_examples,
        "results": results,
    }
