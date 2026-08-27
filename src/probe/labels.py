def build_correct_answer_label(ex) -> int:
    for key in ["label", "answer", "target", "correct_answer_idx"]:
        if key in ex:
            return int(ex[key])
    raise KeyError(f"No supported label key found. Keys: {list(ex.keys())}")

def build_bias_binary_label(ex):
    """
    Returns the index (0/1/2) of the stereotyped answer for ambig examples.
    Returns None (skip) if:
      - context_condition != "ambig"
      - no answer tag matches stereotyped_groups (e.g. only "unknown" candidates)
    """
    if ex.get("context_condition") != "ambig":
        return None

    stereotyped = set(ex.get("additional_metadata", {}).get("stereotyped_groups", []))
    if not stereotyped:
        return None

    answer_info = ex.get("answer_info", {})
    for idx, key in enumerate(["ans0", "ans1", "ans2"]):
        info = answer_info.get(key, [])
        tag = info[1] if len(info) >= 2 else None
        if tag in stereotyped:
            return idx

    return None

def get_label_builder(label_mode: str):
    if label_mode == "correct_answer_3class":
        return build_correct_answer_label
    if label_mode == "bias_binary":
        return build_bias_binary_label
    raise ValueError(f"Unknown label_mode: {label_mode}")
