import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

class HiddenExtractor:
    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device
        torch_dtype = _DTYPE_MAP[dtype]

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map={"": 0} if device == "cuda" else None,
            trust_remote_code=trust_remote_code,
            output_hidden_states=True,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def extract(self, text: str, hidden_mode: str = "final_token"):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        hidden_states = outputs.hidden_states
        layers = hidden_states[1:]  # drop embedding state

        reps = []
        for h in layers:
            if hidden_mode == "final_token":
                vec = h[:, -1, :]
            elif hidden_mode == "mean_pool":
                vec = h.mean(dim=1)
            else:
                raise ValueError(f"Unknown hidden_mode: {hidden_mode}")
            reps.append(vec.squeeze(0).float().cpu().numpy())

        return reps
