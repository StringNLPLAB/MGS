"""Prepare WildChat-style training and validation datasets for VERL trainer.

This script processes JSON datasets containing conversation examples.

Each input example is expected to be a dict with at least one of:
- "instruction": string for single-turn prompts
- "messages": list of {"role": str, "content": str}

We construct chat-style rows consumable by VERL with the following schema:
- data_source: configurable data source name (default: "wildchat")
- prompt: list of { role, content } with any trailing assistant message(s) removed
- ability: "chat"
- reward_model.ground_truth: the original example for reference
- extra_info: { split, index, id, source, domain }

The script shuffles examples, selects up to N, reserves K for validation, and saves
to Parquet files in the provided output directory.
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional

import pandas as pd


def build_prompt(example: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """Convert an input example to a list of chat messages.

    Rules:
    - If "instruction" is present and non-empty, return a single user message.
    - Else if "messages" is present, return the conversation with any trailing
      assistant messages removed. Keep only {role, content} fields and only
      strings with non-empty content. If no user/system messages remain, return None.
    """
    instruction = example.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        return [{"role": "user", "content": instruction.strip()}]

    messages = example.get("messages")
    if isinstance(messages, list) and messages:
        # Normalize messages to simple {role, content} pairs
        normalized: List[Dict[str, str]] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if isinstance(role, str) and isinstance(content, str) and content.strip():
                normalized.append({"role": role, "content": content})

        if not normalized:
            return None

        # Drop trailing assistant message(s) to avoid leaking answers
        end = len(normalized)
        while end > 0 and normalized[end - 1]["role"] == "assistant":
            end -= 1
        normalized = normalized[:end]

        if not normalized:
            return None
        return normalized

    return None


def process_fn(data_source: str, example: Dict[str, Any], idx: int, split: str) -> Optional[Dict[str, Any]]:
    prompt = build_prompt(example)
    if not prompt:
        return None

    row = {
        "data_source": data_source, 
        "prompt": prompt,
        "ability": "chat",
        "reward_model": {
            "style": "model",
            "ground_truth": example,
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "id": example.get("id", idx),
            "source": example.get("source", "none"),
            "domain": example.get("domain", "none"),
        },
    }
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare WildChat datasets for VERL training")
    parser.add_argument(
        "--input_json",
        required=True,
        help="Path to input JSON file containing the dataset",
    )
    parser.add_argument(
        "--data_source",
        default="wildchat",
        help="Name of the data source to use in output files",
    )
    parser.add_argument(
        "--local_dir",
        default="data",
        help="Local directory to save processed datasets",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=200000,
        help="Target number of examples (after filtering); fewer will be used if unavailable",
    )
    parser.add_argument(
        "--num_validation_examples",
        type=int,
        default=1000,
        help="Number of validation examples to reserve",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    random.seed(args.seed)

    os.makedirs(args.local_dir, exist_ok=True)

    # ---------------------- Load input JSON ----------------------
    if not os.path.exists(args.input_json):
        raise FileNotFoundError(f"Input JSON not found: {args.input_json}")

    with open(args.input_json, "r", encoding="utf-8") as f:
        try:
            raw: List[Dict[str, Any]] = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON: {e}")

    print(f"Loaded {len(raw)} raw examples from {args.input_json}")

    # ---------------------- Filter/normalize to prompts ----------------------
    normalized: List[Dict[str, Any]] = []
    for ex in raw:
        try:
            if (row := process_fn(args.data_source, ex, idx=0, split="temp")) is not None:
                normalized.append(ex)
        except Exception:
            continue

    print(f"Usable examples after prompt construction: {len(normalized)}")

    # ---------------------- Sample selection ----------------------
    random.shuffle(normalized)
    target_total = min(args.num_examples, len(normalized))
    selected = normalized[:target_total]
    print(f"Selected {len(selected)} examples (target {target_total}).")

    # ---------------------- Build rows and split ----------------------
    random.shuffle(selected)

    val_n = min(args.num_validation_examples, len(selected))
    train_examples = selected[:-val_n] if val_n > 0 else selected
    val_examples = selected[-val_n:] if val_n > 0 else []

    train_rows = [r for idx, ex in enumerate(train_examples) if (r := process_fn(args.data_source, ex, idx, "train")) is not None]
    val_rows = [r for idx, ex in enumerate(val_examples) if (r := process_fn(args.data_source, ex, idx, "val")) is not None]

    print(f"train data size: {len(train_rows)}")
    train_df = pd.DataFrame(train_rows)
    train_out = os.path.join(args.local_dir, f"{args.data_source}_train.parquet")
    train_df.to_parquet(train_out)

    print(f"val data size: {len(val_rows)}")
    val_df = pd.DataFrame(val_rows)
    val_out = os.path.join(args.local_dir, f"{args.data_source}_val.parquet")
    val_df.to_parquet(val_out)

    print(f"Saved train to {train_out}")
    print(f"Saved val to {val_out}")

