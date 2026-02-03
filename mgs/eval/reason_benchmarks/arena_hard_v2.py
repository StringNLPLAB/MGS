###################
# WildBench 
# TODO:
# 1. Test this code and ensure that we can replicate the leaderboard numbers
# 2. Add functionality to allow long-COT models to think before answering
# 3. Run this with 
#   a. Deepseek-R1 distilled Q1 models (i) without thinking and (ii) with thinking
#   b. Deepseek-R1 (i) without thinking and (ii) with thinking <-- this requires getting outputs from an API
#   c. o3-mini-high with thinking
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
    
    # Normalize model id like "openai/gpt-4.1" -> "gpt-4.1"
    norm_model = model.split("/")[-1] if isinstance(model, str) else model

    if openai.__version__ == "0.28.0":
        response = openai.ChatCompletion.create(
            model=norm_model,
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
        model = norm_model

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

def extract_values_from_json(
    json_string, 
    keys=["score", "strengths", "weaknesses", "choice"], 
    allow_no_quotes=False
):
    extracted_values = {}
    for key in keys:
        if key not in json_string:
            continue
        # Create a regular expression pattern to find the value for the given key
        pattern = f'"{key}"\\s*:\\s*"([^"]*?)"'
        match = re.search(pattern, json_string)
        if match:
            extracted_values[key] = match.group(1)
        else:
            # Handle the case where the value might contain broken quotes
            pattern = f'"{key}"\\s*:\\s*"(.*?)"'
            match = re.search(pattern, json_string, re.DOTALL)
            if match:
                extracted_values[key] = match.group(1)
        if not match and allow_no_quotes:
            # to allow no quotes on the values
            pattern = f'"{key}"\\s*:\\s*([^,\\s]*)'
            match = re.search(pattern, json_string)
            if match:
                extracted_values[key] = match.group(1)
            else:
                # to allow no quotes on the keys
                pattern = f'{key}\\s*:\\s*([^,\\s]*)'
                match = re.search(pattern, json_string)
                if match:
                    extracted_values[key] = match.group(1)
    return extracted_values

def parse_result(result_str, mode="json", eval_mode="pairwise"): 
    assert eval_mode in ["score", "pairwise"]
    result_str = result_str.strip() 
    try: 
        try:
            parsed_result = json.loads(result_str)
        except:
            parsed_result = extract_values_from_json(result_str, keys=["score", "choice"])
    except Exception as e:
        print(result_str)
        print(e)
        parsed_result = {}
    return parsed_result

def eval_one_example_api_call(
    example, 
    model=None,
    engine=None,
    temperature=0,
    max_tokens=1024,
    mode="pairwise",
):
    # If already evaluated, just re-parse and return.
    if example.get("result", "N/A") != "N/A" and example.get("error", "N/A") == "N/A" and "parsed_result" in example:
        example["parsed_result"] = parse_result(example["result"], eval_mode=mode)
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

    try:
        result = api(**openai_args)
        example["result"] = result
        example["parsed_result"] = parse_result(result, eval_mode=mode)
        example["parsed"] = example["parsed_result"] is not None
        example["error"] = "N/A"
    except Exception as e:
        example["error"] = str(e)
        example["result"] = result if "result" in locals() else "N/A"
        example["parsed_result"] = {}
        
    return example

def eval_one_example(
    candidate,
    history,
    checklist,
    templates,
    mode="pairwise",
    reference=None,
    max_words_to_eval=1000,
    **kwargs,
):
    assert isinstance(history, list) and len(history) > 0, "History should be a list of strings."
    assert isinstance(checklist, list) and len(checklist) > 0, "Checklist should be a list of strings."
    
    history_list, last_query = history[:-1], history[-1]["content"]
    history = ""
    for entry in history_list:
        if entry["role"] == "user":
            history += "USER: " + entry["content"] + "\n\n"
        else:
            history += "ASSISTANT: " + entry["content"] + "\n\n"
    
    if mode == "pairwise":
        reference = reference if isinstance(reference, str) else (
            reference[0] if isinstance(reference, list) else reference
        )
        
        if random.random() < 0.5:
            A = candidate
            B = reference
            candidate_position = "A"
        else:
            A = reference
            B = candidate
            candidate_position = "B"
        
        history = shorten(history, max_words_to_eval)
        user_query = shorten(last_query, max_words_to_eval)
        A = shorten(A, max_words_to_eval)
        B = shorten(B, max_words_to_eval)
        checklist_markdown = ""
        for checklist_item in checklist:
            checklist_markdown += f"- {checklist_item}\n"
        
        prompt = templates["pairwise"].replace(
            "{$history}", history
        ).replace(
            "{$user_query}", last_query
        ).replace(
            "{$candidate_A}", A
        ).replace(
            "{$candidate_B}", B
        ).replace(
            "{$checklist}", checklist_markdown
        )
        
        result = eval_one_example_api_call(
            {"prompt": prompt},
            mode="pairwise",
            **kwargs,
        )  
        
        parsed_result = result["parsed_result"]
        multiplier = 1 if candidate_position == "A" else -1
        reward = -100
        if "choice" in parsed_result:
            choice = parsed_result["choice"].replace(" ", "").strip().upper()
            if parsed_result["choice"] == "A++":
                reward = 100 * multiplier
            elif parsed_result["choice"] == "A+":
                reward = 50 * multiplier
            elif parsed_result["choice"] == "A=B":
                reward = 0 * multiplier
            elif parsed_result["choice"] == "B+":
                reward = -50 * multiplier
            elif parsed_result["choice"] == "B++":
                reward = -100 * multiplier
        parsed_result = {"reward": reward}                     
    elif mode == "score":
        history = shorten(history, max_words_to_eval)
        user_query = shorten(last_query, max_words_to_eval)
        model_output = shorten(candidate, max_words_to_eval)
        checklist_markdown = ""
        for checklist_item in checklist:
            checklist_markdown += f"- {checklist_item}\n"
        prompt = templates["score"].replace(
            "{$history}", history
        ).replace(
            "{$user_query}", last_query
        ).replace(
            "{$model_output}", model_output
        ).replace(
            "{$checklist}", checklist_markdown
        )
        
        result = eval_one_example_api_call(
            {"prompt": prompt},
            mode="score",
            **kwargs,
        )
        
        parsed_result = result["parsed_result"]
        score = 1
        if "score" in parsed_result:
            parsed_score = parsed_result["score"].replace(" ", "").strip()
            if parsed_score.isnumeric():
                parsed_score = int(parsed_score)
                if parsed_score >= 1 and parsed_score <= 10:
                    score = parsed_score
        parsed_result = {"score": score}
        result["metrics"] = parsed_result
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    return result, parsed_result

def convert_wildbench_prompt_into_single_conversation(
    prompt: List[Dict[str, str]],
):
    preamble = "Consider the following conversation between a user and an assistant. " + \
        "You should step in as the assistant and provide the best possible response to the user's query. " + \
        "Do not print any extra information or formatting, and do *not* print the 'ASSISTANT:' tag--just the response."
    conversation = ""
    for entry in prompt:
        if entry["role"] == "user":
            conversation += "USER: " + entry["content"] + "\n\n"
        elif entry["role"] == "assistant":
            conversation += "ASSISTANT: " + entry["content"] + "\n\n"
        else:
            raise ValueError(f"Invalid role: {entry['role']}")
    return [
        {
            "role": "user",
            "content": preamble + "\n\n" + conversation
        }
    ]

class ArenaHardV2(ReasonBenchmark):
    BENCHMARK_NAME = "arena_hard_v2"
    _cfg = None

    # Judge configuration defaults (can be overridden in YAML if added later)
    _judge_model_default = "openai/gpt-4.1"
    _judge_temperature = 0.0
    _judge_max_tokens = 3072

    # LongCoT handling (strip think blocks / formatting from model outputs)
    _longcot = False
    _longcot_delimiter = None
    _end_delimiter = None
    _start_think_marker = None

    # Verdict parsing
    _regex_patterns = [r"\[\[([AB<>=]+)\]\]", r"\[([AB<>=]+)\]"]

    # Pairwise prompt template (from arena-hard-auto config)
    _prompt_template = (
        "<|User Prompt|>\n{QUESTION}\n\n"
        "<|The Start of Assistant A's Answer|>\n{ANSWER_A}\n<|The End of Assistant A's Answer|>\n\n"
        "<|The Start of Assistant B's Answer|>\n{ANSWER_B}\n<|The End of Assistant B's Answer|>"
    )

    # System prompts with and without style-control (mirrors judge_utils)
    _OG_ARENA_HARD_PROMPT = (
        "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. "
        "You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\n"
        "Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\n"
        "When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\n"
        "Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. "
        "Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. "
        "Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\n"
        "Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\n"
        "After providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n"
        "1. Assistant A is significantly better: [[A>>B]]\n"
        "2. Assistant A is slightly better: [[A>B]]\n"
        "3. Tie, relatively the same: [[A=B]]\n"
        "4. Assistant B is slightly better: [[B>A]]\n"
        "5. Assistant B is significantly better: [[B>>A]]\n\n"
        "Example output: \"My final verdict is tie: [[A=B]]\"."
    )

    _CREATIVE_PROMPT = (
        "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. "
        "You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\n"
        "When evaluating the assistants' answers, compare both assistants' answers. You must identify and correct any mistakes or inaccurate information.\n\n"
        "Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. "
        "Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. "
        "Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\n"
        "Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\n"
        "After providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n"
        "1. Assistant A is significantly better: [[A>>B]]\n"
        "2. Assistant A is slightly better: [[A>B]]\n"
        "3. Tie, relatively the same: [[A=B]]\n"
        "4. Assistant B is slightly better: [[B>A]]\n"
        "5. Assistant B is significantly better: [[B>>A]]\n\n"
        "Example output: \"My final verdict is tie: [[A=B]]\"."
    )

    @classmethod
    def load_cfg(cls, cfg_file: Optional[str] = None, **kwargs):
        if cls._cfg is not None:
            return
        if cfg_file is None:
            cfg_file = "reason_benchmarks/benchmark_cfgs/arena_hard_v2.yaml"
        with open(cfg_file, "r") as f:
            cls._cfg = yaml.safe_load(f)
            
        # Optional overrides if ever added to YAML
        cls._judge_model_default = cls._cfg.get("judge_model", cls._judge_model_default)

        # LongCoT options via kwargs
        if kwargs.get("longcot", False):
            cls._longcot = True
            cls._longcot_delimiter = kwargs.get("longcot_delimiter", cls._longcot_delimiter or "</think>")
            if isinstance(cls._longcot_delimiter, str):
                cls._longcot_delimiter = cls._longcot_delimiter.replace("\\n", "\n")
            cls._end_delimiter = kwargs.get("end_delimiter", cls._end_delimiter)
            cls._start_think_marker = kwargs.get("start_think_marker", cls._start_think_marker)

    @classmethod
    def _inner_load(cls, cfg_file: Optional[str] = None, *args, **kwargs) -> List[Dict[str, Any]]:
        cls.load_cfg(cfg_file, **kwargs)
        prompts_file = cls._cfg["prompts_file"]

        with open(prompts_file, "r") as f:
            data = json.load(f)

        dataset = []
        for i, d in enumerate(data):
            proc_ex = {}
            proc_ex["inst_id"] = d.get("uid", f"arena_hard_v2_{i}")
            # Simple prompting: just the user prompt
            proc_ex["input_prompt"] = [{"role": "user", "content": d["prompt"]}]
            # Store ground truth/reference and meta we need for judging
            proc_ex["ground_truth"] = {
                "reference": d.get("reference", ""),
                "reference_model": d.get("reference_model", "unknown"),
                "category": d.get("category", "hard_prompt"),
            }
            dataset.append(proc_ex)
        return dataset

    @classmethod
    def _get_system_prompt_for_category(cls, category: str) -> str:
        if category == "creative_writing":
            return cls._CREATIVE_PROMPT
        # Style control ON for non-creative categories (hard prompts)
        return cls._OG_ARENA_HARD_PROMPT

    @classmethod
    def _format_pairwise_prompt(cls, question: str, answer_a: str, answer_b: str) -> str:
        return cls._prompt_template.format(QUESTION=question, ANSWER_A=answer_a, ANSWER_B=answer_b)

    @classmethod
    def _parse_verdict(cls, judgment_text: str) -> Optional[str]:
        import re
        upper = judgment_text.upper()
        for pattern in cls._regex_patterns:
            matches = re.findall(pattern, upper)
            matches = [m for m in matches if m != ""]
            if len(set(matches)) > 0:
                return matches[-1].strip("\n")
        return None

    @classmethod
    def _judge_pairwise(
        cls,
        question: str,
        baseline_answer: str,
        candidate_answer: str,
        category: str,
        judge_model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        if judge_model is None:
            judge_model = cls._judge_model_default

        system_prompt = cls._get_system_prompt_for_category(category)
        user_prompt = cls._format_pairwise_prompt(question, baseline_answer, candidate_answer)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        resp = openai_chat_request(
            model=judge_model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
        )[0]
        verdict = cls._parse_verdict(resp)
        return {"judgment": resp, "verdict": verdict}

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
        ground = data_inst["ground_truth"]
        category = ground.get("category", "hard_prompt")
        question = data_inst["input_prompt"][0]["content"]
        baseline = ground.get("reference", "")

        judge_model = cls._cfg.get("judge_model", cls._judge_model_default)
        temp = cls._judge_temperature
        max_toks = cls._judge_max_tokens

        per_output_metrics = []
        per_output_aux = []
        
        for candidate in model_output["output"]:
            if candidate is None:
                print("Candidate is None")
                candidate = ""
            # Strip long-CoT think content if configured
            if cls._longcot and cls._longcot_delimiter and isinstance(candidate, str) and cls._longcot_delimiter in candidate:
                candidate = candidate.split(cls._longcot_delimiter)[-1].lstrip()
            if cls._longcot and cls._end_delimiter and isinstance(candidate, str) and cls._end_delimiter in candidate:
                candidate = candidate.split(cls._end_delimiter)[0].rstrip()
            # Round 1: A=baseline, B=candidate
            r1 = cls._judge_pairwise(
                question=question,
                baseline_answer=baseline,
                candidate_answer=candidate,
                category=category,
                judge_model=judge_model,
                temperature=temp,
                max_tokens=max_toks,
            )

            # Round 2: A=candidate, B=baseline
            r2 = cls._judge_pairwise(
                question=question,
                baseline_answer=candidate,
                candidate_answer=baseline,
                category=category,
                judge_model=judge_model,
                temperature=temp,
                max_tokens=max_toks,
            )

            # Map verdicts to numeric win/tie/loss for the candidate
            def score_round(verdict: Optional[str], flipped: bool) -> float:
                if verdict is None:
                    return 0.5  # treat unknown as tie
                verdict = verdict.replace(" ", "")
                # When flipped=False, A=baseline, B=candidate
                # When flipped=True, A=candidate, B=baseline
                if not flipped:
                    if verdict in ["B>A", "B>>A"]:
                        return 1.0
                    if verdict == "A=B":
                        return 0.5
                    if verdict in ["A>B", "A>>B"]:
                        return 0.0
                else:
                    if verdict in ["A>B", "A>>B"]:
                        return 1.0
                    if verdict == "A=B":
                        return 0.5
                    if verdict in ["B>A", "B>>A"]:
                        return 0.0
                return 0.5

            s1 = score_round(r1["verdict"], flipped=False)
            s2 = score_round(r2["verdict"], flipped=True)
            win_rate = float((s1 + s2) / 2.0)

            per_output_metrics.append({"win_rate": win_rate})
            per_output_aux.append({
                "round1": r1,
                "round2": r2,
            })

        # Aggregate across multiple sampled outputs for this instance
        if aggregation == "max":
            agg_win = float(max(m["win_rate"] for m in per_output_metrics))
        else:
            agg_win = float(np.mean([m["win_rate"] for m in per_output_metrics]))

        return {"win_rate": agg_win}, {"judgments": per_output_aux}


def test_loading():
    data = ArenaHardV2.load()
    print(data[0]["ground_truth"])
    print(data[0]["inst_id"])
    print(data[0]["input_prompt"])

if __name__ == "__main__":
    test_loading()

