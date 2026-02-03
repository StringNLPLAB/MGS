import argparse
import yaml
import os
import numpy as np
import random
import json
import copy

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from cached_query_tool import cached_batch_query
import reason_benchmarks
from reason_benchmarks.benchmark_utils import load_reason_benchmark, AGG_TYPES, TIEBREAK_TYPES
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)

    # general config
    parser.add_argument("--config", type=str, default=None)

    # dataset args
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--benchmark_config", type=str, default=None)
    parser.add_argument("--test_sample_size", type=int, default=None)

    # base query args
    parser.add_argument("--model", type=str, default="vllm/models/Llama-3.2-3B-Instruct", help="Model name or path. Add vllm/ prefix for using VLLM.")
    parser.add_argument("--max_tokens", type=str, default="2048", help="Max generation tokens. It has to be a string, if multiple values are provided, they will be used for different benchmarks.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_model_len", type=int, default=None, help="Maximum length of the model. Just a int.")

    # sampling args
    parser.add_argument("--n", type=str, default="1", help="Number of samples to generate")
    parser.add_argument("--aggregation", type=str, default="majority", choices=AGG_TYPES, help="Aggregation method for multiple samples, default is majority voting.")
    parser.add_argument("--tiebreak", type=str, default="longest", choices=TIEBREAK_TYPES, help="Tiebreak method for multiple samples, default is longest.")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.95, help="GPU memory utilization for VLLM. Default is 0.95.")

    # outputs and caching
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_saving_name", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--skip_eval", action="store_true", default=False) # skip evaluation -- useful if we need internet access for evaluation
    parser.add_argument("--force_overwrite", action="store_true", default=False)

    # longcot options
    parser.add_argument("--longcot_config", type=str, default=None)
    parser.add_argument("--longcot", action="store_true", default=False)
    parser.add_argument("--longcot_delimiter", type=str, default="</think>")
    parser.add_argument("--end_delimiter", type=str, default=None)
    parser.add_argument("--start_think_marker", type=str, default=None)
    parser.add_argument("--prompt_for_longcot", type=str, default=None)
    parser.add_argument("--stop_token", type=str, default=None)

    # only for gpt-oss
    parser.add_argument("--reasoning_effort", type=str, default=None, choices=["low", "medium", "high"])

    # parallel evaluation options
    parser.add_argument("--parallel_eval", action="store_true", default=False, help="Enable parallel evaluation (e.g., OpenAI-based judges).")
    parser.add_argument("--parallel_eval_batch_size", type=int, default=32, help="Batch size for parallel evaluation when --parallel_eval is set.")

    args = parser.parse_args()
    
    if args.longcot_config is not None:
        config = json.load(open(args.longcot_config))
        args.prompt_for_longcot = config.get("prompt_for_longcot", None)        # If set -> we prompt the model to perform longcot
        args.longcot = config.get("longcot", True)
        args.longcot_delimiter = config.get("longcot_delimiter", "</think>")
        args.end_delimiter = config.get("end_delimiter", None)
        args.start_think_marker = config.get("start_think_marker", None)
    else:
        if args.longcot_delimiter is not None and "\\n" in args.longcot_delimiter:
            args.longcot_delimiter = args.longcot_delimiter.replace("\\n", "\n")
        if args.end_delimiter is not None and "\\n" in args.end_delimiter:
            args.end_delimiter = args.end_delimiter.replace("\\n", "\n")
        if args.start_think_marker is not None and "\\n" in args.start_think_marker:
            args.start_think_marker = args.start_think_marker.replace("\\n", "\n")


    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)

    args.model = args.model.rstrip("/")
    if args.model_saving_name is None:
        if args.model.startswith("vllm/"):
            model_base_name = "vllm/" + os.path.basename(args.model)
        else:
            model_base_name = os.path.basename(args.model)
        args.model_saving_name = model_base_name.replace("/", "-")

    if args.output_dir is None:
        args.output_dir = os.path.join("outputs", args.model_saving_name)

    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def output_filename_func(args):
    # save results
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_filename = f"{args.benchmark}_max{args.max_tokens}t{args.temperature}p{args.top_p}n{args.n}_{args.seed}.json"
    return os.path.join(args.output_dir, output_filename)

def eval_benchmark(args):
    # set random seed
    set_seed(args.seed)

    # load benchmark data
    benchmark_cls = load_reason_benchmark(
        args.benchmark,
    )

    data = benchmark_cls.load(
        cfg_file=args.benchmark_config,
        longcot=args.longcot,
        longcot_delimiter=args.longcot_delimiter,
        end_delimiter=args.end_delimiter,
        start_think_marker=args.start_think_marker,
    )

    if args.test_sample_size is not None:
        # sample without replacement
        data = random.sample(data, min(args.test_sample_size, len(data)))

    # seperate the query and evaluation for easy caching and re-computing eval metrics
    cache_file = os.path.join("caches", f"{args.model_saving_name}_{args.benchmark}.sqlite")

    # batch querying
    prompts = []
    for d_inst in data:
        user_prompt = d_inst["input_prompt"]
        if isinstance(user_prompt, list):
            msgs = user_prompt
        elif isinstance(user_prompt, str):
            msgs = [{"role": "user", "content": user_prompt}]
        else:
            raise ValueError(f"Unknown prompt type: {type(user_prompt)}")
        prompts.append(msgs)

    # viewing first 2 prompts
    for p in prompts[:2]:
        print(p[-1]["content"])

    # checkpoint
    output_filename = output_filename_func(args)
    checkpoint_filename = os.path.join(os.path.dirname(output_filename), f"checkpoint_{os.path.basename(output_filename)}")

    print("*" * 50)
    print(f"Checkpoint Filename: {checkpoint_filename}")
    print(f"Cache File: {cache_file}")
    print("*" * 50)
    if os.path.exists(checkpoint_filename) and not args.force_overwrite:
        print(f"!!! Loading checkpoint from {checkpoint_filename}")
        outputs = json.load(open(checkpoint_filename))
    else:
        query_kwargs = {}
        if "gpt-oss" in args.model and args.reasoning_effort is not None:
            query_kwargs["reasoning_effort"] = args.reasoning_effort
        if args.stop_token is not None:
            query_kwargs["stop"] = [args.stop_token]
            # try: adding "User:" to the stop token (hardcoded)
            # query_kwargs["stop"] += ["User:"]
        outputs = cached_batch_query(
            cache_file, 
            prompts, 
            args.model, 
            args.max_tokens, 
            args.temperature, 
            args.top_p, 
            args.max_model_len, 
            args.n,
            start_think_marker=args.start_think_marker,
            aux_kwargs={"force_overwrite": args.force_overwrite, "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization},
            query_kwargs=query_kwargs,
            custom_system_prompt=args.prompt_for_longcot,
        )

        # save checkpoint
        with open(checkpoint_filename, "w") as f:
            json.dump(outputs, f, indent=4)

        # save outputs
        intermediate_outputs = []
        for d_inst, o in zip(data, outputs):
            intermediate_outputs.append({
                "inst_id": d_inst["inst_id"],
                "input_prompt": d_inst["input_prompt"],
                "output": o["output"],
            })
        os.makedirs(args.output_dir, exist_ok=True)
        
        intermediate_filename = output_filename_func(args)
        intermediate_filename = os.path.join(os.path.dirname(intermediate_filename), "model_outputs_" + os.path.basename(intermediate_filename))
        with open(intermediate_filename, "w") as f:
            json.dump(intermediate_outputs, f, indent=4)  

    # view first 2 outputs
    for o in outputs[:2]:
        print(o["output"][0])  

    if args.skip_eval:
        print("*** Skipping evaluation ***")
        return
    else:
        print("*** Evaluating ***")

    results = []
    all_metrics = []

    def eval_one(idx):
        d_inst = data[idx]
        output = outputs[idx]
        metrics, aux_info = benchmark_cls.evaluate(d_inst, output, aggregation=args.aggregation, tiebreak=args.tiebreak)
        out = {
            "inst_id": d_inst["inst_id"],
            "output": {**output, **aux_info},
            "metrics": metrics,
        }
        return idx, metrics, out

    if not args.parallel_eval:
        for i in tqdm(range(len(data))):
            _, metrics, out = eval_one(i)
            all_metrics.append(metrics)
            results.append(out)
    else:
        batch_size = args.parallel_eval_batch_size
        for start in tqdm(range(0, len(data), batch_size)):
            end = min(start + batch_size, len(data))
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {executor.submit(eval_one, i): i for i in range(start, end)}
                for fut in as_completed(futures):
                    idx, metrics, out = fut.result()
                    all_metrics.append(metrics)
                    results.append(out)

    if benchmark_cls.implements_aggregation():
        avg_metrics = benchmark_cls.aggregate_metrics(all_metrics)
    else:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([(0 if key not in m else m[key]) for m in all_metrics])

    # aggregation metrics
    avg_metrics["test_sample_size"] = len(all_metrics)
    if "extraction" in avg_metrics and "accuracy" in avg_metrics:
        avg_metrics["extracted_accuracy"] = avg_metrics["accuracy"] / avg_metrics["extraction"] if avg_metrics["extraction"] > 0 else 0.0

    output_filename = output_filename_func(args)

    output_content = {
        "avg_metrics": avg_metrics,
        "args": args.__dict__,
        "results": results,
    }
    print(f"Saving results to {output_filename}")
    print(f"Average metrics:")
    for k, v in avg_metrics.items():
        print(f"{k}: {v}")

    with open(output_filename, "w") as f:
        json.dump(output_content, f, indent=2)

    with open(output_filename.replace(".json", ".json.score"), "w") as f:
        json.dump(avg_metrics, f, indent=2)


def main():
    args = _parse_args()

    if "," in args.max_tokens:
        max_tokens = [int(x) for x in args.max_tokens.split(",")]
    else:
        max_tokens = [int(args.max_tokens)]

    if "," in args.n:
        ns = [int(x) for x in args.n.split(",")]
    else:
        ns = [int(args.n)]

    if "," in args.benchmark:
        benchmarks = args.benchmark.split(",")
    else:
        benchmarks = [args.benchmark]

    assert len(max_tokens) == 1 or len(max_tokens) == len(benchmarks)
    assert len(ns) == 1 or len(ns) == len(benchmarks)

    if len(max_tokens) == 1:
        max_tokens = max_tokens * len(benchmarks)
    if len(ns) == 1:
        ns = ns * len(benchmarks)

    for benchmark, max_token, n_sample in zip(benchmarks, max_tokens, ns):
        benchmark_args = copy.deepcopy(args)
        benchmark_args.benchmark = benchmark
        benchmark_args.max_tokens = max_token
        benchmark_args.n = n_sample

        eval_benchmark(benchmark_args)
        continue
        try:
            eval_benchmark(benchmark_args)
        except Exception as e:
            print(f"FAILURE Error in benchmark {benchmark}")
            print(e)
            if args.debug:
                raise e

if __name__=="__main__":
    main()
