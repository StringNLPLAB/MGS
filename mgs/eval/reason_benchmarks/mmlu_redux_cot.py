###################
# MMLU Redux
###################

import yaml
import json
from typing import Dict, Any, Optional, List, Tuple
from .benchmark_utils import ReasonBenchmark
import re

def eval_one_example(
    response: str,
    ground_truth: str,
    **kwargs,
):
    answer = None
    if "Answer:" in response:
        answer = response.split("Answer:")[1].strip()
    elif "answer:" in response:
        answer = response.split("answer:")[1].strip()
        
    if answer is not None:
        answer_start = answer[:100]
        answer = re.sub(r"[^A-Z]", "", answer)
        if len(answer) > 0:
            answer = answer[0]
            if answer not in ["A", "B", "C", "D"]:
                print(f"Invalid answer start: {answer_start}")

    if answer not in ["A", "B", "C", "D"]:
        extraction = 0.0
        score = 0.0
    else:
        extraction = 1.0
        score = 1.0 if answer.upper() == ground_truth.upper() else 0.0
    return {
        "extraction": extraction,
        "score": score,
    }

class MMLUReduxCoT(ReasonBenchmark):
    BENCHMARK_NAME = "mmlu_redux_cot"
    _cfg = None
    _longcot = False
    _longcot_delimiter = None
    _end_delimiter = None
    
    @classmethod
    def load_cfg(cls, cfg_file: Optional[str] = None, **kwargs):
        if cls._cfg is not None:
            return
        if cfg_file is None:
            cfg_file = "reason_benchmarks/benchmark_cfgs/mmlu_redux.yaml"
        with open(cfg_file, "r") as f:
            cls._cfg = yaml.safe_load(f)

        if kwargs.get("longcot", False):
            cls._longcot = True
            cls._longcot_delimiter = kwargs.get("longcot_delimiter", "</think>")
            cls._longcot_delimiter = cls._longcot_delimiter.replace("\\n", "\n")
            cls._end_delimiter = kwargs.get("end_delimiter", None)

    @classmethod
    def _inner_load(cls, cfg_file: Optional[str] = None, *args, **kwargs) -> List[Dict[str, Any]]:
        """Load mmlu_redux dataset."""
        cls.load_cfg(cfg_file, **kwargs)

        with open(cls._cfg["data_file"], "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = []
        for i, example in enumerate(data):
            prompt = f"Please answer the following question:\n\n{example['question']}\n\nSelect the correct answer from the following options:\n\n"
            for i, option in enumerate(example["choices"]):
                option_str = chr(ord("A") + i) + f". {option}\n"
                prompt += option_str
            # prompt += "\nOutput only the letter of the correct answer in uppercase and nothing else - do not include any other comments or formatting such as quotes."
            prompt += "\nMark your final uppercase letter answer after \"Answer:\""

            answer, error, alt_answer = example["answer"], example["error_type"], example["correct_answer"]
            if error == "ok":
                answer = chr(ord("A") + answer)
            elif error == "wrong_groundtruth" and isinstance(alt_answer, str) and alt_answer.isdigit():
                answer = chr(ord("A") + int(alt_answer))
            else:
                # wrong question
                continue

            example = {
                "inst_id": str(i),
                "input_prompt": [{
                    "role": "user",
                    "content": prompt
                }],
                "ground_truth": answer,
            }
            dataset.append(example)
        
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

        results = []
        metrics = []
        for response in model_output["output"]:
            if response is None:
                print("Response is None")
                response = ""
            if cls._longcot and cls._longcot_delimiter in response:
                response = response.split(cls._longcot_delimiter)[-1].lstrip()
            if cls._longcot and cls._end_delimiter and cls._end_delimiter in response:
                response = response.split(cls._end_delimiter)[0].rstrip()

            result = eval_one_example(
                response,
                data_inst["ground_truth"],
            )
            result.update({
                "inst_id": data_inst["inst_id"],
            })
            results.append(result)

            metrics.append({
                "score": result["score"],
            })

        # aggregate metrics
        if aggregation == "majority":
            num_follows = sum([result["score"] for result in results])
            if 2*num_follows >= len(results):
                metrics = {
                    "score": 1,
                }
            else:
                metrics = {
                    "score": 0,
                }
        else:
            raise ValueError(f"Invalid aggregation: {aggregation}")

        return metrics, {"results": results}

def test_loading():
    # """Load the benchmark data. requiring "inst_id", "input_prompt", and "ground_truth" keys."""
    data = MMLURedux.load()
    print(data[0]["inst_id"])
    print(data[0]["input_prompt"])

if __name__ == "__main__":
    test_loading()

