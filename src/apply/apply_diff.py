import torch

def apply_diff_to_model(model, diff, target_layers):
    """
    model: Hugging Face causal LM
    diff: dict[name -> tensor]
    target_layers: list[int]
    """
    for name, param in model.named_parameters():
        if ".layers." in name:
            try:
                layer_id = int(name.split(".layers.")[1].split(".")[0])
            except Exception:
                continue
            if layer_id not in target_layers:
                continue
        else:
            continue

        if name in diff:
            delta = diff[name].to(device=param.device, dtype=param.data.dtype)
            param.data.add_(delta)

    return model
