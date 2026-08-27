#!/usr/bin/env python3
"""Negate a diff vector: saves {k: -v for k, v in diff.items()} to output path."""
import argparse
import torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="source diff .pt")
    p.add_argument("--output", required=True, help="destination .pt for negated diff")
    args = p.parse_args()

    diff = torch.load(args.input, map_location="cpu", weights_only=False)
    neg  = {k: -v for k, v in diff.items()}
    torch.save(neg, args.output)
    print(f"saved negated diff ({len(neg)} keys) → {args.output}")

if __name__ == "__main__":
    main()
