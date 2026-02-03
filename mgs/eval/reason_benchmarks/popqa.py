###################
# PopQA
###################

import yaml
import json
from typing import Dict, Any, Optional, List, Tuple
from .benchmark_utils import ReasonBenchmark

def eval_one_example(
    response: str,
    ground_truth: List[str],
    **kwargs,
):
    any_word_in_response = 0
    
    ## split response into words
    ## it seems like the popqa code doesn't do this
    # response_words = response.strip().split()

    if any(
        (word in response) or 
        (word.lower() in response) or 
        (word.upper() in response) or 
        (word.capitalize() in response)
        for word in ground_truth
    ):
        any_word_in_response = 1
    
    return {
        "score": any_word_in_response,
    }

class PopQA(ReasonBenchmark):
    BENCHMARK_NAME = "popqa"
    _cfg = None
    _longcot = False
    _longcot_delimiter = None
    _end_delimiter = None
    
    @classmethod
    def load_cfg(cls, cfg_file: Optional[str] = None, **kwargs):
        if cls._cfg is not None:
            return
        if cfg_file is None:
            cfg_file = "reason_benchmarks/benchmark_cfgs/popqa.yaml"
        with open(cfg_file, "r") as f:
            cls._cfg = yaml.safe_load(f)

        if kwargs.get("longcot", False):
            cls._longcot = True
            cls._longcot_delimiter = kwargs.get("longcot_delimiter", "</think>")
            cls._longcot_delimiter = cls._longcot_delimiter.replace("\\n", "\n")
            cls._end_delimiter = kwargs.get("end_delimiter", None)

    @classmethod
    def _inner_load(cls, cfg_file: Optional[str] = None, *args, **kwargs) -> List[Dict[str, Any]]:
        """Load popqa dataset."""
        cls.load_cfg(cfg_file, **kwargs)

        with open(cls._cfg["data_file"], "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = []
        for example in data:
            prompt = f"Please answer the following question by generating a short snippet with a few sentences. You should not exceed 100 words in your response. Do not output any other text or comments.\n\n{example['question']}"
            
            # Parse the possible_answers JSON string into an actual list
            try:
                possible_answers = json.loads(example["possible_answers"])
            except (json.JSONDecodeError, TypeError):
                # Fallback: if it's already a list or parsing fails
                possible_answers = example["possible_answers"]
                if isinstance(possible_answers, str):
                    # Last resort: split by comma and clean up
                    possible_answers = [ans.strip().strip('"').strip("'") for ans in possible_answers.split(",")]
            
            example = {
                "inst_id": example["id"],
                "input_prompt": [{
                    "role": "user",
                    "content": prompt
                }],
                "ground_truth": possible_answers,
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
    data = PopQA.load()
    print(data[0]["inst_id"])
    print(data[0]["input_prompt"])

if __name__ == "__main__":
    test_loading()

