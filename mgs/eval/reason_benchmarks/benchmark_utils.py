import yaml
from collections import Counter
from typing import Dict, Type, Any, Optional, List


# aggregation types
AGG_TYPES = ["majority", "longest", "max"]
TIEBREAK_TYPES = ["first", "longest"]

class ReasonBenchmark:
    """base class for all benchmarks."""

    _registry = {}
    _implements_aggregation = False

    def __init_subclass__(cls, **kwargs):
        """Register all children in registry"""
        super().__init_subclass__(**kwargs)
        if cls == ReasonBenchmark:
            return
        print(f"Registering {cls.__name__} as {cls.BENCHMARK_NAME}")
        dataset_name = getattr(cls, "BENCHMARK_NAME", cls.__name__)
        ReasonBenchmark._registry[dataset_name] = cls

    @classmethod
    def load_cfg(cls, cfg_file: Optional[str] = None, **kwargs):
        if cls._cfg is not None:
            return
        if cfg_file is None:
            dataset_name = getattr(cls, "BENCHMARK_NAME", cls.__name__)
            cfg_file = f"reason_benchmarks/benchmark_cfgs/{dataset_name}.yaml"
        with open(cfg_file, "r") as f:
            cls._cfg = yaml.safe_load(f)

    @classmethod
    def load(cls, cfg_file: Optional[str] = None, *args, **kwargs) -> List[Dict[str, Any]]:
        """Load the benchmark data. requiring "inst_id", "input_prompt", and "ground_truth" keys."""
        data = cls._inner_load(cfg_file, *args, **kwargs)
        # exam data instances
        assert isinstance(data, list)
        assert isinstance(data[0], dict)
        assert "inst_id" in data[0]
        assert "input_prompt" in data[0]
        assert "ground_truth" in data[0]
        return data

    @classmethod
    def _inner_load(cls, cfg_file: Optional[str] = None, *args, **kwargs) -> List[Dict[str, Any]]:
        """Inner load function benchmark."""
        raise NotImplementedError

    @classmethod
    def get_benchmark(cls, name, *args, **kwargs):
        """Retrieve dataset class by name and instantiate it."""
        if name not in ReasonBenchmark._registry:
            raise ValueError(f"Dataset '{name}' not found in registry. Available: {list(ReasonBenchmark._registry.keys())}")
        return ReasonBenchmark._registry[name](*args, **kwargs)

    @classmethod
    def list_benchmarks(cls):
        """List all registered datasets."""
        return list(ReasonBenchmark._registry.keys())

    @classmethod
    def evaluate(cls, data_inst: Any, model_output: Any, aggregation: Optional[str] = "majority", *args, **kwargs) -> Any:
        """Evaluate the benchmarks."""
        raise NotImplementedError

    @classmethod
    def implements_aggregation(cls) -> bool:
        """Check if the benchmark implements aggregation."""
        return cls._implements_aggregation

    @classmethod
    def aggregate_metrics(cls, metrics: Any) -> Dict[str, Any]:
        """Aggregate metrics."""
        raise NotImplementedError

# Example usage
def load_reason_benchmark(name: str, **kwargs) -> Optional[Type[ReasonBenchmark]]:
    """Factory function to create Benchmark instances."""
    benchmark_cls = ReasonBenchmark.get_benchmark(name, **kwargs)
    if benchmark_cls is None:
        raise ValueError(f"Benchmark '{name}' not found in registry")
    return benchmark_cls


### some utility functions for evaluation
def extract_with_tag(response: str, tag: str):
    start = response.find(f"<{tag}>")
    end = response.find(f"</{tag}>")
    if start == -1 or end == -1:
        return None
    return response[start+len(tag)+2:end].strip()


#### default evaluator helpers
def aggregate_among_valid_extractions(
        extraction_func: callable, # function for extracting the final parsed output
        data_inst: Dict[str, Any],
        model_output: Dict[str, Any],
        aggregation="majority",
        tiebreak="first",
        *args,
        **kwargs
    ) -> str:
    """
    Aggregate among valid extractions.
    Take a model outputs and aggregate to select a final solution.
    For instance, output is [(A), (B), (A), (C)], return (A)
    """
    assert aggregation in AGG_TYPES and tiebreak in TIEBREAK_TYPES

    extracted_solutions = [(extraction_func(pr_str), pr_str) for pr_str in model_output["output"]]
    extracted_solutions = [(sol, pr_str) for sol, pr_str in extracted_solutions if sol is not None]

    if len(extracted_solutions) == 0:
        return None

    if aggregation == "majority":
        sol_counter = Counter([sol for sol, _ in extracted_solutions])
        max_count = max(sol_counter.values())
        max_cnt_solutions = [(sol, pr_str) for sol, pr_str in extracted_solutions if sol_counter[sol] == max_count]

        if len(max_cnt_solutions) == 1:
            return max_cnt_solutions[0][0]
        elif tiebreak == "first":
            return max_cnt_solutions[0][0]
        elif tiebreak == "longest":
            return max(max_cnt_solutions, key=lambda x: len(x[1]))[0]
    elif aggregation == "longest":
        solution = max(extracted_solutions, key=lambda x: len(x[1]))[0]

    return solution


def default_mcqa_evaluator(
        extraction_func: callable,
        data_inst: Dict[str, Any],
        model_output: Dict[str, Any],
        aggregation="majority",
        tiebreak="first",
        *args,
        **kwargs
    ):
    """Evaluate mcqa benchmarks.
    return: (metrics, aux_info)
    example usage:
    >> exc_func = lambda x: extract_with_tag(x, "choice")
    >> extract_with_tag(data_inst, model_output, extraction_func)
    """

    gt = data_inst["ground_truth"]
    solution = aggregate_among_valid_extractions(extraction_func, data_inst, model_output, aggregation, tiebreak)
    acc = 1.0 if solution == gt else 0.0
    extraction = 1.0 if solution is not None else 0.0
    return {"accuracy": acc, "extraction": extraction}, {"solution": solution}
