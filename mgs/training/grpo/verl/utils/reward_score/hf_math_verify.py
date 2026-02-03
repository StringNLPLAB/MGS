# reward_math_safe.py
from __future__ import annotations
import re
from typing import Tuple, Optional

from verl.utils.reward_score.qwen_math_eval_toolkit.parser import extract_answer as qwen_extract_answer
from math_verify import parse, verify

_STOP_WORDS = ("</s>", "<|im_end|>", "<|endoftext|>", "</response>")

def _trim_to_assistant(text: str) -> str:
    m = re.search(r'<\|im_start\|>assistant', text)
    return text[m.start():] if m else text

def _extract_last_boxed(text: str) -> Optional[str]:
    # Match \boxed{...} with nested braces tolerance.
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = list(re.finditer(pattern, text))
    return matches[-1].group(0) if matches else None

def _extract_solution(solution_str: str) -> Tuple[str, bool]:
    """
    Returns (target_boxed, had_boxed_in_output)
    """
    model_output = _trim_to_assistant(solution_str)
    for sw in _STOP_WORDS:
        if sw in model_output:
            model_output = model_output.split(sw, 1)[0].strip()
    pred = qwen_extract_answer(model_output, data_name="math")
    boxed = _extract_last_boxed(model_output)
    if boxed is not None:
        return boxed, True
    return f"\\boxed{{{pred}}}", False

def _verify_safe(gold_boxed: str, target_boxed: str) -> bool:
    try:
        parsed_gold = parse(gold_boxed)
        parsed_target = parse(target_boxed)
        return bool(verify(gold=parsed_gold, target=parsed_target))
    except Exception:
        return False

def compute_score(data_source, solution_str: str, ground_truth: str, extra_info=None, method: str = "strict") -> float:
    """
    VERL-compatible reward function (single-process).
    Returns 1.0 if verified equal, else 0.0.
    """
    print("Calculating math score")
    # tags = ["<think>", "</think>", "<response>", "</response>"]
    # if any(tag not in solution_str for tag in tags):
    #     return 0
    target_boxed, _ = _extract_solution(solution_str.split("<response>")[-1])

    gold_boxed = ground_truth if "\\boxed" in ground_truth else f"\\boxed{{{ground_truth}}}"
    correct = _verify_safe(gold_boxed, target_boxed)
    return 1.0 if correct else 0.0
