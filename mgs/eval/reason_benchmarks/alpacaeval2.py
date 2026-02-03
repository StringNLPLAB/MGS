import yaml
import json
from typing import Dict, Type, Any, Optional, List, Tuple
from .benchmark_utils import ReasonBenchmark, extract_with_tag, aggregate_among_valid_extractions
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import os
import openai 
import random
import re 
import sys
import time
from functools import wraps

import openai
if openai.__version__ == "0.28.0":
    OPENAI_RATE_LIMIT_ERROR = openai.error.RateLimitError
    OPENAI_API_ERROR = openai.error.APIError
else:
    from openai import OpenAI
    OPENAI_RATE_LIMIT_ERROR = openai.RateLimitError
    OPENAI_API_ERROR = openai.APIError

def shorten(text, K=-1):
    if K > 0 and len(text.split(" ")) > K:
        text = " ".join(text.split(" ")[:K]) + "... (truncated)" 
    return text

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

def parse_result(result_str): 
    try:
        for entry in json.loads(result_str)["output"]:
            model = entry["model"]
            rank = int(entry["rank"])
            if rank == 1 and model in ["A", "B"]:
                return model, "N/A"
        return None, f"Could not find model A or B in the results."
    except Exception as e:
        print(result_str)
        print(e)
        return None, f"Error parsing result: {e}"

def eval_one_example_api_call(
    example, 
    model=None,
    engine=None,
    temperature=0,
    max_tokens=128,
):
    # If already evaluated, just re-parse and return.
    if example.get("result", "N/A") != "N/A" and example.get("error", "N/A") == "N/A" and "parsed_result" in example:
        example["parsed_result"] = parse_result(example["result"])
        example["parsed"] = example["parsed_result"] is not None
        return example

    openai_args = {
        "prompt": example["prompt"],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "json_mode": True,
        "stop": []
    }
    if model:
        openai_args["model"] = model
    if engine:
        openai_args["engine"] = engine

    @retry_handler(retry_limit=10)
    def api(**kwargs):
        if kwargs.get("model", "").startswith("claude"):
            raise NotImplementedError("Claude API is not implemented yet.")
        else:
            result = openai_chat_request(**kwargs)
        return result[0]

    result = api(**openai_args)
    example["result"] = result
    example["parsed_result"], example["error"] = parse_result(result)
    example["parsed"] = example["parsed_result"] is not None
        
    return example

def eval_one_example(
    instruction,
    model_output,
    reference,
    template,
    max_words_to_eval=1000,
    **kwargs,
):
    if random.random() < 0.5:
        A = model_output
        B = reference
        candidate_position = "A"
    else:
        A = reference
        B = model_output
        candidate_position = "B"
    
    instruction = shorten(instruction, max_words_to_eval)
    A = shorten(A, max_words_to_eval)
    B = shorten(B, max_words_to_eval)

    prompt = template.replace(
        "{$instruction}", instruction
    ).replace(
        "{$model_output}", A # randomize the order of A and B
    ).replace(
        "{$reference}", B
    )
    
    result = eval_one_example_api_call(
        {"prompt": prompt},
        **kwargs,
    )  
    
    winner = result["parsed_result"]
    entry = {
        "instruction": instruction,
        "model_output": model_output,
        "reference": reference,
        "candidate_position": candidate_position,
        "judgement": result["result"],
        "preferred": int(winner == candidate_position)
    }                 
    
    return entry

class AlpacaEval2(ReasonBenchmark):
    BENCHMARK_NAME = "alpacaeval2"
    _cfg = None
    _eval_template = None
    _eval_kwargs = {}
    _ref_model = None
    _longcot = False
    _longcot_delimiter = None
    _end_delimiter = None
    _implements_aggregation = True

    @classmethod
    def load_cfg(cls, cfg_file: Optional[str] = None, **kwargs):
        if cls._cfg is not None:
            return
        if cfg_file is None:
            cfg_file = "reason_benchmarks/benchmark_cfgs/alpacaeval2.yaml"
        with open(cfg_file, "r") as f:
            cls._cfg = yaml.safe_load(f)
            
        if kwargs.get("longcot", False):
            cls._longcot = True
            cls._longcot_delimiter = kwargs.get("longcot_delimiter", "</think>")
            cls._longcot_delimiter = cls._longcot_delimiter.replace("\\n", "\n")
            cls._end_delimiter = kwargs.get("end_delimiter", None)

    @classmethod
    def _inner_load(cls, cfg_file: Optional[str] = None, *args, **kwargs) -> List[Dict[str, Any]]:
        cls.load_cfg(cfg_file, **kwargs)

        cls._eval_template = open(cls._cfg["eval_template"], "r").read()
        cls._eval_kwargs = {
            "max_words_to_eval": cls._cfg["max_words_to_eval"],
            "model": cls._cfg["model"],
            "engine": cls._cfg["engine"],
            "temperature": cls._cfg["temperature"],
            "max_tokens": cls._cfg["max_tokens"]
        }
        cls._ref_model = cls._cfg["ref_model"]
        
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
                    "content": d["instruction"]
                } 
            ] # longcot is fine for single turn
            
            proc_ex["input_prompt"] = input_prompt
            proc_ex["baseline"] = d["output"]
            proc_ex["instruction"] = d["instruction"]
            proc_ex["ground_truth"] = None

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
        
        instruction, reference = data_inst["instruction"], data_inst["baseline"]
        
        metrics = []
        results = []
        
        for candidate in model_output["output"]:
            # If we are using long-cot, and the output has a think delimiter, we need to split the output
            if candidate is None:
                print("Candidate is None")
                candidate = ""
            if cls._longcot and cls._longcot_delimiter in candidate:
                candidate = candidate.split(cls._longcot_delimiter)[-1].lstrip()
            if cls._longcot and cls._end_delimiter and cls._end_delimiter in candidate:
                candidate = candidate.split(cls._end_delimiter)[0].rstrip()
            result = eval_one_example(
                instruction,
                candidate,
                reference,
                cls._eval_template,
                **cls._eval_kwargs
            )
            results.append(result)

            # We need two entries - one for the model and one for the reference
            metric = [
                {
                    "model": "model",
                    "delta_len": len(result["model_output"].split()) - len(result["reference"].split()),
                    "win": result["preferred"]
                },
                {
                    "model": "reference",
                    "delta_len": -len(result["model_output"].split()) + len(result["reference"].split()),
                    "win": 1 - result["preferred"]
                }
            ]
            metrics.append(metric)
            
        # Instance-level aggregation 
        # Always choose the first entry
        metrics = metrics[0]
        return metrics, {"results": results}

    @classmethod
    def aggregate_metrics(
        cls,
        metrics: List[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        # Flatten the list of lists and convert it into a dataframe
        df = pd.DataFrame(
            [
                entry for outer_entry in metrics for entry in outer_entry
            ]
        )   # "model", "delta_len", and "win"

        # Fit logistic regression model
        X = pd.get_dummies(df["model"], prefix="model")
        X["delta_len"] = df["delta_len"]
        y = df["win"]
        clf = LogisticRegression(fit_intercept=True, solver="lbfgs")
        clf.fit(X, y)

        # 3. Counterfactual preds
        X_zero = X.copy()
        X_zero["delta_len"] = 0
        df["lc_pred"] = clf.predict_proba(X_zero)[:, 1]

        # 4. Calculate win rates
        win_rate = (df.groupby("model")["win"].mean() * 100).to_dict()
        lc_win_rate = (df.groupby("model")["lc_pred"].mean() * 100).to_dict()
        metrics = {
            "win_rate": win_rate["model"],
            "lc_win_rate": lc_win_rate["model"]
        }

        return metrics

def test_loading():
    # """Load the benchmark data. requiring "inst_id", "input_prompt", and "ground_truth" keys."""
    data = WildBench.load()
    print(data[0]["ground_truth"])
    print(data[0]["inst_id"])
    print(data[0]["input_prompt"])

if __name__ == "__main__":
    test_loading()

