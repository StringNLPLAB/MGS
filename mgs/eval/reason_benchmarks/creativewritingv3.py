###################
# Creative Writing V3
###################

import yaml
import json
from typing import Dict, Type, Any, Optional, List, Tuple
from .benchmark_utils import ReasonBenchmark, extract_with_tag, aggregate_among_valid_extractions
import numpy as np
import time

import os
import openai 
import random
import re 
import sys
import itertools
from functools import wraps


import openai
if openai.__version__ == "0.28.0":
    OPENAI_RATE_LIMIT_ERROR = openai.error.RateLimitError
    OPENAI_API_ERROR = openai.error.APIError
else:
    from openai import OpenAI
    OPENAI_RATE_LIMIT_ERROR = openai.RateLimitError
    OPENAI_API_ERROR = openai.APIError

# Scoring utilities
SCORE_RANGE_MIN = 0
SCORE_RANGE_MAX = 20

def parse_judge_scores_creative(judge_model_response: str) -> Dict[str, float]:
    scores = {}

    # Parse scores using multiple regex patterns
    # Pattern 1: Metric: Score or Metric: Score X
    score_pattern1 = r'(.*?):\s*(?:Score\s+)?(-?\d+(?:\.\d+)?)'
    # Pattern 2: Metric: [Score]
    score_pattern2 = r'(.*?):\s*\[(-?\d+(?:\.\d+)?)\]'
    
    # Combine both patterns
    matches1 = re.findall(score_pattern1, judge_model_response)
    matches2 = re.findall(score_pattern2, judge_model_response)
    
    # Process matches from both patterns
    for matches in [matches1, matches2]:
        for match in matches:
            metric_name = match[0].strip()
            score = float(match[1])
            # Add check to ensure score <= 20
            if score <= SCORE_RANGE_MAX:
                scores[metric_name] = score
            # If score > 20, it's discarded/ignored

    return scores

def invert_if_negative(metric: str, score: float, negative_criteria: List[str]) -> float:
    if metric in negative_criteria:
        return 20.0 - score
    return score

def compute_score(
    task: Dict[str, Any], 
    negative_criteria: List[str]
) -> float:
    piece_scores = []
            
    # Gather valid numeric scores
    local_vals = []
    for metric, val in task.items():
        if isinstance(val, (int, float)):
            new_val = invert_if_negative(metric, val, negative_criteria)
            if new_val <= SCORE_RANGE_MAX:
                local_vals.append(new_val)
    if local_vals:
        score = sum(local_vals) / len(local_vals)  # 0 to 20
    else:
        return None
    
    return score * 5.0 # scale to 0..100

# API utilities
def truncate_islice(text, max_words=3200):
    if text is None:
        return ""
    it = re.finditer(r'\S+', text)
    # get the 3200th match (zero‐based slice at index 3199)
    m = next(itertools.islice(it, max_words-1, None), None)
    return text[: m.end() ] if m else text

def openai_chat_request(
    model: str=None,
    engine: str=None,
    temperature: float=0,
    max_tokens: int=512,
    top_p: float=1.0,
    frequency_penalty: float=0,
    presence_penalty: float=0,
    prompt: str=None,
    n: int=1,
    messages: List[dict]=None,
    stop: List[str]=None,
    json_mode: bool=False,
    **kwargs,
) -> List[str]:
    """
    Request the evaluation prompt from the OpenAI API in chat format.
    Args:
        prompt (str): The encoded prompt.
        messages (List[dict]): The messages.
        model (str): The model to use.
        engine (str): The engine to use.
        temperature (float, optional): The temperature. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 800.
        top_p (float, optional): The top p. Defaults to 0.95.
        frequency_penalty (float, optional): The frequency penalty. Defaults to 0.
        presence_penalty (float, optional): The presence penalty. Defaults to 0.
        stop (List[str], optional): The stop. Defaults to None.
    Returns:
        List[str]: The list of generated evaluation prompts.
    """
    # Call openai api to generate aspects
    assert prompt is not None or messages is not None, "Either prompt or messages should be provided."
    if messages is None:
        messages = [{"role":"system","content":"You are a helpful AI assistant."},
                    {"role":"user","content": prompt}]
    
    if openai.__version__ == "0.28.0":
        response = openai.ChatCompletion.create(
            model=model,
            response_format = {"type": "json_object"} if json_mode else None,
            engine=engine,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            **kwargs,
        )
        contents = []
        for choice in response['choices']:
            # Check if the response is valid
            if choice['finish_reason'] not in ['stop', 'length']:
                raise ValueError(f"OpenAI Finish Reason Error: {choice['finish_reason']}")
            contents.append(choice['message']['content'])
    else:
        # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="http://llms-se.baidu-int.com:8200")
        model = model.split("/")[-1]

        response = client.chat.completions.create(
            model=model, 
            response_format = {"type": "json_object"} if json_mode else None,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            **kwargs,
        )
        contents = []
        for choice in response.choices:
            # Check if the response is valid
            if choice.finish_reason not in ['stop', 'length']:
                if 'content_filter' in choice.finish_reason:
                    contents.append("Error: content filtered due to OpenAI policy. ")
                else:
                    raise ValueError(f"OpenAI Finish Reason Error: {choice.finish_reason}")
            contents.append(choice.message.content.strip())
    return contents

def anthropic_chat_request(
    model: str = None,
    temperature: float = 0,
    max_tokens: int = 512,
    prompt: str = None,
    n: int = 1,
    messages: list = None,
    stop: list = None,
    json_mode: bool = False,
    api_key: str = None,
    **kwargs,
) -> list:
    """
    Request the evaluation prompt from the Anthropic API in chat format (Claude models).
    Args:
        prompt (str): The encoded prompt (if messages is None).
        messages (List[dict]): The messages (Anthropic format: role/content).
        model (str): The Claude model to use.
        temperature (float, optional): The temperature. Defaults to 0.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 512.
        stop (List[str], optional): The stop sequences. Defaults to None.
        json_mode (bool, optional): If True, requests a JSON-formatted response. Defaults to False.
        api_key (str, optional): Anthropic API key. Defaults to environment variable.
        n (int, optional): Number of completions (Anthropic only supports n=1 for now).
    Returns:
        List[str]: The list of generated evaluation prompts.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("Please install the anthropic package: pip install anthropic")
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    if messages is None:
        if prompt is None:
            raise ValueError("Either prompt or messages must be provided.")
        messages = [
            {"role": "user", "content": prompt}
        ]
    # Anthropic only supports n=1 for now
    if n != 1:
        raise ValueError("Anthropic API only supports n=1 completions per request.")
    # Claude v3 supports JSON mode via system prompt
    extra_args = {}
    if json_mode:
        extra_args["system"] = "Respond only with a valid JSON object."
    # Call the API
    response = client.messages.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stop_sequences=stop,
        **extra_args,
        **kwargs,
    )
    # Anthropic returns a list of content blocks; join all text blocks
    content = "".join(
        block["text"] if isinstance(block, dict) and block.get("type") == "text" else str(block)
        for block in response.content
    ) if hasattr(response, "content") else str(response)
    return [content]

def retry_handler(retry_limit=3):
    """
        This is an error handler for requests to OpenAI API.
        If will retry for the request for `retry_limit` times if the error is not a rate limit error.
        Otherwise, it will wait for the time specified in the error message and constantly retry.
        You can add specific processing logic for different types of errors here.

        Args:
            retry_limit (int, optional): The number of times to retry. Defaults to 3.

        Usage:
            @retry_handler(retry_limit=3)
            def call_openai_api():
                pass
    """
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retried = 0
            while True:
                try:
                    sys.stdout.flush()
                    return func(*args, **kwargs)
                except Exception as e:
                    # if rate limit error, wait 2 seconds and retry
                    if isinstance(e, OPENAI_RATE_LIMIT_ERROR):
                        words = str(e).split(' ')
                        try:
                            time_to_wait = int(words[words.index('after') + 1])
                        except ValueError:
                            time_to_wait = 5
                        # print("Rate limit error, waiting for {} seconds for another try..".format(time_to_wait))
                        time.sleep(time_to_wait) # wait 30 seconds
                        # print("Finished waiting for {} seconds. Start another try".format(time_to_wait))
                    elif isinstance(e, OPENAI_API_ERROR):
                        # this is because the prompt contains content that is filtered by OpenAI API
                        if retried < retry_limit:
                            print("API error:", str(e))
                            if "invalid" in str(e).lower():
                                print("Invalid request, returning.")
                                retried = retry_limit
                                raise e
                            print(f"Retrying for the {retried + 1} time..")
                        else:
                            err_msg = str(e)
                            if '504 Gateway Time-out' in err_msg:
                                print ('Yi issue!')
                                return ['']
                            else:
                                raise e # to prevent infinite loop
                        retried += 1
                    else:
                        err_msg = str(e)
                        print(e.__class__.__name__+":", err_msg)
                        if retried < retry_limit:
                            if 'blocked' in err_msg:
                                print ('blocked output issue!')
                                return ['Error: this query is blocked by APIs.']
                            print(f"Retrying for the {retried + 1} time..")
                        else:
                            print("Retry limit reached. Saving the error message and returning.")
                            print(kwargs["prompt"])
                            raise e
                        retried += 1
        return wrapper
    return decorate

def eval_one_example_api_call(
    example, 
    negative_criteria,
    model=None,
    engine=None,
    temperature=0,
    max_tokens=2048,
):
    # If already evaluated, just re-parse and return.
    if example.get("result", "N/A") != "N/A" and example.get("error", "N/A") == "N/A" and "parsed_result" in example:
        example["parsed_result"] = parse_judge_scores_creative(result)
        example["parsed"] = example["parsed_result"] is not None
        example["error"] = "N/A"
        example["score"] = compute_score(example["parsed_result"], negative_criteria)
        assert example["score"] is not None, "Score is None"
        return example

    if model.startswith("anthropic/"):
        model = model.split("/")[-1]
        api = anthropic_chat_request
    elif model.startswith("openai/"):
        model = model.split("/")[-1]
        api = openai_chat_request
    else:
        raise ValueError(f"Invalid model: {model}")

    args = {
        "prompt": example["prompt"],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": []
    }
    if model:
        args["model"] = model
    if engine:
        args["engine"] = engine


    try:
        result = api(**args)[0]
        example["result"] = result
        example["parsed_result"] = parse_judge_scores_creative(result)
        example["parsed"] = example["parsed_result"] is not None
        example["error"] = "N/A"
        example["score"] = compute_score(example["parsed_result"], negative_criteria)
    except Exception as e:
        raise e
        example["error"] = str(e)
        example["result"] = result if "result" in locals() else "N/A"
        example["parsed_result"] = {}
        
    return example

def eval_one_example(
    candidate: str,
    base_prompt: str,
    judge_prompt: str,
    creative_writing_criteria: List[str],
    negative_criteria: List[str],
    **kwargs,
):
    prompt = judge_prompt.format(
        writing_prompt=base_prompt,
        test_model_response=candidate,
        creative_writing_criteria="\n".join(["- " + c for c in creative_writing_criteria]),
        lower_is_better_criteria=", ".join(negative_criteria),
    )

    return eval_one_example_api_call(
        {"prompt": prompt},
        negative_criteria,
        **kwargs,
    )

class CreativeWritingV3(ReasonBenchmark):
    BENCHMARK_NAME = "creativewritingv3"
    _cfg = None
    _eval_templates = None
    _eval_kwargs = {}
    _ref_model = None
    _longcot = False
    _longcot_delimiter = None
    _end_delimiter = None

    @classmethod
    def load_cfg(cls, cfg_file: Optional[str] = None, **kwargs):
        if cls._cfg is not None:
            return
        if cfg_file is None:
            cfg_file = "reason_benchmarks/benchmark_cfgs/creativewritingv3.yaml"
        with open(cfg_file, "r") as f:
            cls._cfg = yaml.safe_load(f)
            
        if kwargs.get("longcot", False):
            cls._longcot = True
            cls._longcot_delimiter = kwargs.get("longcot_delimiter", "</think>")
            cls._longcot_delimiter = cls._longcot_delimiter.replace("\\n", "\n")
            cls._end_delimiter = kwargs.get("end_delimiter", None)

    @classmethod
    def _inner_load(cls, cfg_file: Optional[str] = None, *args, **kwargs) -> List[Dict[str, Any]]:
        """Load countdown dataset."""
        cls.load_cfg(cfg_file, **kwargs)

        with open(cls._cfg["creative_criteria_file"], "r", encoding="utf-8") as f:
            cls.creative_writing_criteria = [line.strip() for line in f if line.strip()]
        with open(cls._cfg["negative_criteria_file"], "r", encoding="utf-8") as f:
            cls.negative_writing_criteria = [line.strip() for line in f if line.strip()]
        with open(cls._cfg["judge_prompt_file"], "r", encoding="utf-8") as f:
            cls.judge_prompt = f.read()
        with open(cls._cfg["creative_prompts_file"], "r", encoding="utf-8") as f:
            creative_prompts = json.load(f)

        dataset = []
        for prompt_key, prompt_obj in creative_prompts.items():
            base_prompt = prompt_obj.get("writing_prompt", "")
            seed_mods = prompt_obj.get("seed_modifiers", [])
            if not seed_mods:
                print(f"No seed modifiers for prompt {prompt_key}; skipping.")
                continue

            for i in range(1, cls._cfg["iterations"]+1):
                iteration_seed = seed_mods[(i-1) % len(seed_mods)]
                final_prompt = base_prompt.replace("<SEED>", iteration_seed)
                input_prompt = [
                    {
                        "role": "user",
                        "content": final_prompt
                    }
                ]
                proc_ex = {
                    "inst_id": cls.BENCHMARK_NAME + f"_{prompt_key}_{i}",
                    "input_prompt": input_prompt,
                    "seed_modifier": iteration_seed,
                    "base_prompt": base_prompt,
                    "ground_truth": {}
                }
                dataset.append(proc_ex)
        
        return dataset

    @classmethod
    def evaluate(
        cls, 
        data_inst: Dict[str, Any], 
        model_output: Dict[str, Any],
        aggregation: Optional[str] = "max", 
        tiebreak: Optional[str] = "first",
        *args, 
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        return: (metrics, aux_info)
        """

        results = []
        metrics = []
        for candidate in model_output["output"]:
            try:
                if cls._longcot and cls._longcot_delimiter in candidate:
                    candidate = candidate.split(cls._longcot_delimiter)[-1].lstrip() # split at the last e.g. <think>
                if cls._longcot and cls._end_delimiter and cls._end_delimiter in candidate:
                    candidate = candidate.split(cls._end_delimiter)[0].rstrip()

                # Truncate responses
                if cls._cfg.get("max_response_words", None) is not None:
                    candidate = truncate_islice(candidate, cls._cfg["max_response_words"])
            except Exception as e:
                print(f"Error truncating candidate: {e}")
                candidate = ""

            result = eval_one_example(
                candidate,
                data_inst["base_prompt"],
                cls.judge_prompt,
                cls.creative_writing_criteria,
                cls.negative_writing_criteria,
                model=cls._cfg["judge_model"],
            )
            result.update({
                "inst_id": data_inst["inst_id"],
                "seed_modifier": data_inst["seed_modifier"],
            })
            results.append(result)

            metric = {
                k: v for k, v in result.items()
                if isinstance(v, (int, float)) 
                and k not in ["parsed"]
            }
            metrics.append(metric)

        # aggregate metrics
        # override everything and do mean
        metrics = {
            key: sum([( 0 if key not in m else m[key]) for m in metrics]) / len(metrics)
            for key in metrics[0].keys()
        }

        return metrics, {"results": results}

def test_loading():
    # """Load the benchmark data. requiring "inst_id", "input_prompt", and "ground_truth" keys."""
    data = CreativeWritingV3.load()
    print(data[0]["inst_id"])
    print(data[0]["input_prompt"])

if __name__ == "__main__":
    test_loading()

