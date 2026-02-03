###################
# MATH-500
###################

import yaml
import json
from typing import Dict, Type, Any, Optional, List, Tuple
from .benchmark_utils import ReasonBenchmark
import numpy as np

import os
import openai 
import random
import re 
import sys
from functools import wraps

## Evaluation helpers

def compute_score(solution_str, ground_truth) -> float:
    retval = 0.
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if is_equiv(answer, ground_truth):
                retval = 1.
    except Exception as e:
        print(e)

    return retval

# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string

def remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string
## Evaluation helpers END

class MATH500(ReasonBenchmark):
    BENCHMARK_NAME = "math_500"
    _cfg = None
    _longcot = False
    _longcot_delimiter = None
    _end_delimiter = None

    @classmethod
    def load_cfg(cls, cfg_file: Optional[str] = None, **kwargs):
        if cls._cfg is not None:
            return
        if cfg_file is None:
            cfg_file = "reason_benchmarks/benchmark_cfgs/math_500.yaml"
        with open(cfg_file, "r") as f:
            cls._cfg = yaml.safe_load(f)

        # Handle longcot configuration
        if kwargs.get("longcot", False):
            cls._longcot = True
            cls._longcot_delimiter = kwargs.get("longcot_delimiter", "</think>")
            cls._longcot_delimiter = cls._longcot_delimiter.replace("\\n", "\n")
            cls._end_delimiter = kwargs.get("end_delimiter", None)

        print(f"Loaded longcot={cls._longcot}, longcot_delimiter={cls._longcot_delimiter}, end_delimiter={cls._end_delimiter}")

    @classmethod
    def _inner_load(cls, cfg_file: Optional[str] = None, *args, **kwargs) -> List[Dict[str, Any]]:
        """Load countdown dataset."""
        cls.load_cfg(cfg_file, **kwargs)

        # Handle longcot configuration in _inner_load as well
        if kwargs.get("longcot", False):
            cls._longcot = True
            cls._longcot_delimiter = kwargs.get(
                "longcot_delimiter",
                cls._longcot_delimiter if cls._longcot_delimiter else "</think>"
            )
            cls._longcot_delimiter = cls._longcot_delimiter.replace("\\n", "\n")
            cls._end_delimiter = kwargs.get("end_delimiter", cls._end_delimiter)

        with open(cls._cfg["test_file"], "r") as f:
            data = json.load(f)

        dataset = []
        for i, d in enumerate(data):
            proc_ex = {}
            inst_id = cls.BENCHMARK_NAME + f"_{i}"
            proc_ex["inst_id"] = inst_id
            
            input_prompt = [
                {
                    "role": "user",
                    "content": f"{d['problem']}\n\nEnclose your final answer in \\boxed{{}}."
                } 
            ]
            
            proc_ex["input_prompt"] = input_prompt
            proc_ex["ground_truth"] = d["answer"]

            dataset.append(proc_ex)
        return dataset

    @classmethod
    def evaluate(
        cls, 
        data_inst: Dict[str, Any], 
        model_output: Dict[str, Any],
        aggregation: Optional[str] = "majority", 
        tiebreak: Optional[str] = "first",
        *args, 
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        return: (metrics, aux_info)
        """
        
        answer = data_inst["ground_truth"]
        
        metrics = []
        results = []
        
        for candidate in model_output["output"]:
            if candidate is None:
                print("Candidate is None")
                candidate = ""
            # Handle longcot: extract the final answer after thinking
            original_candidate = candidate
            if cls._longcot and cls._longcot_delimiter in candidate:
                candidate = candidate.split(cls._longcot_delimiter)[-1].lstrip()
            if cls._longcot and cls._end_delimiter and cls._end_delimiter in candidate:
                candidate = candidate.split(cls._end_delimiter)[0].rstrip()

            result = {
                "candidate": original_candidate,  # Keep original for debugging
                "extracted_answer": candidate,    # Store extracted answer
                "score": compute_score(candidate, answer)
            }
            metric = {
                "score": compute_score(candidate, answer)
            }
            results.append(result)
            metrics.append(metric)
            
        # Instance-level aggregation - ignore the `aggregation` and `tiebreak` arguments
        # They are always averaged for MATH-500
        keys = metrics[0].keys()
        if aggregation.startswith("pass@"):
            k = int(aggregation.split("@")[1])
            metrics = {
                key: float(
                    any(
                        m[key] > 0.5 for m in metrics[:k]
                    )
                ) for key in keys
            }
        elif aggregation == "majority":
            metrics = {
                key: float(np.mean([m[key] for m in metrics]) > 0.5)
                for key in keys
            }
        elif aggregation == "average":
            metrics = {
                key: np.mean([m[key] for m in metrics]) 
                for key in keys
            }
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        return metrics, {"problem": data_inst["input_prompt"], "answer": answer, "results": results}


def test_loading():
    # """Load the benchmark data. requiring "inst_id", "input_prompt", and "ground_truth" keys."""
    data = MATH500.load()
    print(data[0]["ground_truth"])
    print(data[0]["inst_id"])
    print(data[0]["input_prompt"])

if __name__ == "__main__":
    test_loading()

