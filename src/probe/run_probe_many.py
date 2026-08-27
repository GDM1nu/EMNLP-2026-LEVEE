import argparse
from pathlib import Path

from src.probe.run_probe import run_probe
from src.utils.io import load_yaml, save_json
from src.utils.paths import ensure_dir
from src.utils.seed import set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True, help="Path to models.yaml")
    parser.add_argument("--task", required=True, help="Path to probe_task.yaml")
    parser.add_argument("--only", default=None, help="Run only one model alias")
    args = parser.parse_args()

    model_bundle = load_yaml(args.models)
    task_cfg = load_yaml(args.task)

    seed = int(task_cfg.get("seed", 42))
    set_seed(seed)

    out_dir = ensure_dir("results/probing")

    for model_cfg in model_bundle["models"]:
        alias = model_cfg["name"]
        if args.only is not None and alias != args.only:
            continue

        try:
            result = run_probe(model_cfg, task_cfg)
            out_path = Path(out_dir) / f"{alias}_{task_cfg['task_name']}.json"
            save_json(result, out_path)
            print(f"[saved] {out_path}")
        except Exception as e:
            print(f"[failed] {alias}: {repr(e)}")

if __name__ == "__main__":
    main()
