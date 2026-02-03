# Evaluation Framework

This directory contains code to evaluate models on the suite of benchmarks included in the paper. 
It is designed to be easy to use and extend to other benchmarks.
At this moment, we support models hosted on vllm from local or HuggingFace checkpoints, OpenAI API endpoints, Anthropic API endpoints, Gemini API endpoints, and models hosted on Together AI.
We also support thinking models (used extensively in our paper) automatically, by including a `longcot_config.json` inside the checkpoint directory. We always use the config
```
{
    "longcot": true,
    "longcot_delimiter": "</think>",
    "end_delimiter": null,
    "start_think_marker": "<think>"
}
```
for the warm-started models.

## Setup

Please install the necessary dependencies according the global `requirements.txt` file!

## Example usage

> **NOTE**: Do not forge to activate your conda environment in place of the placeholder line!

We show an example script in `scripts/launch_general_benchmarks_inference.sh` and `scripts/launch_general_benchmarks_prompting.sh`.
This script has only one model (Llama-3.1-8B-Instruct) for illustration, but has all the benchmarks we evaluated on (the ones commented out are only in the Appendices of our paper).
The reason we have two scripts is that the cluster we performed inference on does not have internet access; therefore, we use the flag `--skip_eval` to only perform model inference, and then the `prompting` script uses the (automatically) cached responses to prompt judge models for the benchmarks that require it.
The `--parallel_eval` flag uses multithreading to speed up the prompting stage.
Note how we pass models in via `vllm/models/{MODEL_NAME}`. The script automatically interprets this to point to the local path `models/{MODEL_NAME}`.
In contrast, `openai/gpt-4.1-mini` would be interpreted as the model `gpt-4.1-mini` in the OpenAI API.
Use prefixes `anthropic/`, `gemini/`, and `together/` to similar ends.

Once you have the results for your models, you can call
```
python scripts/collect_all_outputs.py
```
to collect all the scores and save them in `outputs/outfiles/benchmark_scores.txt`.
The model list and the output file are hardcoded at the top of the script, but you can easily modify them to your needs.

## Behind the scenes

The evaluation suite has a simple structure. The file `reason_benchmarks/benchmark_utils.py` implements a base `ReasonBenchmark` class which individual benchmarks (e.g., `reason_benchmarks/alpacaeval2.py`) subclass to prepare prompts.
The `run_benchmark_sampling.py` file calls these files to prepare prompts, and calls the wrapper `cached_query_tool.py` to sample these models and cache the corresponding responses.
The benchmarks also implement evaluators and aggregators used to compile the final scores in `*.json.score` files.
These files are picked up by the `collect_all_outputs.py` script to compile the final scores.

## Adding your own benchmark

Suppose you want to add a benchmark `bench`.
Go ahead and create a directory `reason_benchmarks/data_files/bench` and put any files (prompts, reference answers, etc.) you need in there.
Also create a file `reason_benchmarks/benchmark_cfgs/bench.yaml` with any hyperparameters that control prompt formatting or evaluation.
You can then add a file `reason_benchmarks/bench.py` to implement the benchmark class.
A good reference is `reason_benchmarks/alpacaeval2.py`.
You will primarily need to implement the `_inner_load` method to load the prompts and format them as per the files and config you added above, and also the `evaluate` method to evaluate a single example.