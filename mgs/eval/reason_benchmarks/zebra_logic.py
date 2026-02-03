import yaml
import json
import math
from typing import Dict, Type, Any, Optional, List, Tuple
from .benchmark_utils import ReasonBenchmark, aggregate_among_valid_extractions


_ZEBRA_GRID = """
# Example Puzzle 

There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:
 - Each person has a unique name: `Peter`, `Eric`, `Arnold`.
 - Each person has a unique favorite drink: `tea`, `water`, `milk`

## Clues for the Example Puzzle

1. Peter is in the second house.
2. Arnold is directly left of the one who only drinks water.
3. The one who only drinks water is directly left of the person who likes milk.

## Answer to the Example Puzzle

{
    "reasoning": "Given Clue 1, we know Peter is in House 2. According to Clue 2, Arnold is directly left of the one who only drinks water. The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. Thus, Peter drinks water, and Eric lives in House 3. Then, according to Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.",
    "solution": {
        "House 1": {
            "Name": "Arnold",
            "Drink": "tea"
        },
        "House 2": {
            "Name": "Peter",
            "Drink": "water"
        },
        "House 3": {
            "Name": "Eric",
            "Drink": "milk"
        }
    }
}

# Puzzle to Solve 

{puzzle}


# Instruction

Now please solve the above puzzle. Present your reasoning and solution in the following json format:

{json_template}

"""

_SIZE_RANGES = {
        "small": (0, 10**3),
        "medium": (10**3, 10**6),
        "large": (10**6, 10**10),
        "xlarge": (10**10, float("inf"))
}


def _read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def _apply_lgp_grid_template(item):
    prompt_str = _ZEBRA_GRID[:] 
    prompt_str = prompt_str.replace("{puzzle}", item["puzzle"])
    num_houses = len(item["solution"]["rows"])
    columns = item["solution"]["header"]
    assert columns[0] == "House"
    json_template = {"reasoning": "___", "solution": {}}
    for i in range(num_houses):
        json_template["solution"][f'House {i+1}'] = {columns[j]: "___" for j in range(1, len(columns))}
    json_str = json.dumps(json_template, indent=4)
    prompt_str = prompt_str.replace("{json_template}", json_str)
    return prompt_str

def _search_space_size(size):
    N, M = map(int, size.split("*"))
    return (math.factorial(N))**M


# code from ZebraLogic codebase
def _extract_last_complete_json(s):
    # Stack to keep track of opening and closing braces
    stack = []
    last_json_start = None
    last_json_str = None
    
    for i, char in enumerate(s):
        if char == '{':
            stack.append(i)
            if last_json_start is None:
                last_json_start = i
        elif char == '}':
            if stack:
                start = stack.pop()
                if not stack:
                    # Complete JSON object found
                    last_json_str = s[last_json_start:i+1]
                    last_json_start = None
    
    # Load the last JSON object
    if last_json_str:
        try:
            return json.loads(last_json_str.replace("\n", ""))
        except json.JSONDecodeError:
            pass
    
    return None


class ZebraLogic(ReasonBenchmark):

    BENCHMARK_NAME = "zebra_logic"
    _difficulty = None
    _cfg = None
    _longcot = False
    _longcot_delimiter = None
    _end_delimiter = None

    @classmethod
    def load_cfg(cls, cfg_file: Optional[str] = None, **kwargs):
        if cls._cfg is not None:
            return
        if cfg_file is None:
            cfg_file = "reason_benchmarks/benchmark_cfgs/zebra_logic.yaml"
        with open(cfg_file, "r") as f:
            cls._cfg = yaml.safe_load(f)

        # Handle longcot configuration
        if kwargs.get("longcot", False):
            cls._longcot = True
            cls._longcot_delimiter = kwargs.get("longcot_delimiter", "</think>")
            cls._longcot_delimiter = cls._longcot_delimiter.replace("\\n", "\n")
            cls._end_delimiter = kwargs.get("end_delimiter", None)

        print(f"Loaded longcot={cls._longcot}, longcot_delimiter={cls._longcot_delimiter}, end_delimiter={cls._end_delimiter}")

    # NOTE: testing only for now
    @classmethod
    def _inner_load_by_difficulty(cls, cfg_file: Optional[str] = None, difficulty=None, *args, **kwargs) -> List[Dict[str, Any]]:
        """Load zebra logic dataset."""
        cls.load_cfg(cfg_file, **kwargs)

        data = _read_jsonl(cls._cfg["test_file"])

        for d in data:
            d["search_space_size"] = _search_space_size(d["size"])
        if difficulty is None:
            pass
        elif difficulty in _SIZE_RANGES:
            lb = _SIZE_RANGES[difficulty][0]
            ub = _SIZE_RANGES[difficulty][1]
            data = [d for d in data if lb <= d["search_space_size"] < ub]
        else:
            raise ValueError(f"Invalid difficulty: {difficulty}")

        dataset = []
        for i, d in enumerate(data):
            proc_ex = {}
            proc_ex["inst_id"] = d["id"]
            proc_ex["size"] =  d["size"]
            proc_ex["puzzle"] = d["puzzle"]
            proc_ex["ground_truth"] = d["solution"]
            # proc_ex
            input_prompt = _apply_lgp_grid_template(d)
            proc_ex["input_prompt"] = input_prompt

            dataset.append(proc_ex)
        return dataset

    @classmethod
    def _inner_load(cls, cfg_file = None, *args, **kwargs):
        return cls._inner_load_by_difficulty(cfg_file, difficulty=cls._difficulty, *args, **kwargs)

    @classmethod
    def evaluate(
        cls, data_inst: Dict[str, Any], model_output: Dict[str, Any],
        aggregation: Optional[str] = "majority", tiebreak: Optional[str] = "first",
        *args, **kwargs
    ) -> Tuple[Any, Any]:
        """
        return: (metrics, aux_info)
        """
        def extract_func(x):
            # Handle longcot: extract the final answer after thinking
            candidate = x
            if candidate is None:
                print("Candidate is None")
                candidate = ""
            if cls._longcot and cls._longcot_delimiter in candidate:
                candidate = candidate.split(cls._longcot_delimiter)[-1].lstrip()
            if cls._longcot and cls._end_delimiter and cls._end_delimiter in candidate:
                candidate = candidate.split(cls._end_delimiter)[0].rstrip()

            prediction_table = _extract_last_complete_json(candidate)
            if prediction_table is None or "solution" not in prediction_table:
                return None
            # this has to be hashable
            return json.dumps(prediction_table["solution"])

        solution = aggregate_among_valid_extractions(extract_func, data_inst, model_output, aggregation, tiebreak)
        solution = json.loads(solution) if solution is not None else None

        if solution is None:
            acc = 0.0
            extraction = 0.0
        else:
            extraction = 1.0
            acc = _eval_zebra_logic_grid(data_inst, solution)["accuracy"]

        return {"accuracy": acc, "extraction": extraction}, {"solution": solution}


def _eval_zebra_logic_grid(data_inst, prediction_table):
        solution = data_inst["ground_truth"]

        solution_table = {}
        num_houses = len(solution["rows"])
        columns = solution["header"]
        assert columns[0] == "House"
        solution_table = {}
        this_total_cells = 0
        for i in range(num_houses):
            solution_table[f'House {i+1}'] = {columns[j]: solution["rows"][i][j] for j in range(1, len(columns))}
            this_total_cells += len(columns) - 1

        this_correct_cells = 0 # number in the solution_table
        for house in solution_table:
            for column in solution_table[house]:
                # if prediction_table[house][column] not exist then pass
                try:
                    if house in prediction_table and column in prediction_table[house]:
                        truth_cell = solution_table[house][column].lower().strip()
                        if prediction_table[house][column] is None or len(prediction_table[house][column]) == 0:
                            continue
                        if type(prediction_table[house][column]) == list:
                            predicted_cell = prediction_table[house][column][0].lower().strip()
                        elif type(prediction_table[house][column]) == str:
                            predicted_cell = prediction_table[house][column].lower().strip()
                        else:
                            raise ValueError(f"Unknown type: {type(prediction_table[house][column])}")
                        if truth_cell.lower().strip() == predicted_cell.lower().strip():
                            this_correct_cells += 1
                except:
                    # ignore and pass
                    pass

        # compute puzzle success rate
        if this_correct_cells == this_total_cells:
            acc = 1.0
        else:
            acc = 0.0

        return {"accuracy": acc}


class ZebraLogicSmall(ZebraLogic):
    BENCHMARK_NAME = "zebra_logic_small"
    _difficulty = "small"

class ZebraLogicMedium(ZebraLogic):
    BENCHMARK_NAME = "zebra_logic_medium"
    _difficulty = "medium"

class ZebraLogicLarge(ZebraLogic):
    BENCHMARK_NAME = "zebra_logic_large"
    _difficulty = "large"

class ZebraLogicXLarge(ZebraLogic):
    BENCHMARK_NAME = "zebra_logic_xlarge"
    _difficulty = "xlarge"

def test_loading():
    # """Load the benchmark data. requiring "inst_id", "input_prompt", and "ground_truth" keys."""
    data = ZebraLogicMedium.load()
    print(data[0]["input_prompt"])
    print(data[0]["ground_truth"])
    print(data[0]["inst_id"])
    print(len(data))

if __name__ == "__main__":
    # clean_data()
    test_loading()