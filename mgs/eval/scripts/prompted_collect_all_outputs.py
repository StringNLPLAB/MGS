#!/usr/bin/env python3
"""
Script to collect all benchmark scores from model outputs and format them for spreadsheet copying.
"""

import os
import json
import glob
from pathlib import Path

# Define models and benchmarks from the shell script
MODELS = [
    # Example model - replace with your own model path
    "prompt-Llama-3.1-8B",
]

BENCHMARKS = [
    "creativewritingv3",
    "alpacaeval2",
    "wildbench",
    "arena_hard_v2",
    # "wildbench_newref"
]

# Define score keys for different benchmarks
SCORE_KEYS = {
    "alpacaeval2": "lc_win_rate",
    "wildbench": "reward",
    "wildbench_newref": "reward",
    "creativewritingv3": "score",
    "ifeval": "score", 
    "mmlu_redux": "score",
    "mmlu_redux_cot": "score",
    "popqa": "score",
    "arena_hard_v2": "win_rate",
    "math_500": "score",
    "zebra_logic": "accuracy",
    "ifbench": "score",
}

def find_score_file(outputs_dir, benchmark, model):
    """Find the score file for a given benchmark and model."""
    # Build the path to the model directory
    model_dir = Path(outputs_dir) / f"{benchmark}-compare" / model
    
    if not model_dir.exists():
        print(f"Model directory {model_dir} does not exist")
        return None
    
    # Look for score files that start with the benchmark name
    score_files = list(model_dir.glob(f"{benchmark}*.json.score"))
    
    if not score_files:
        print(f"No score files found for {model_dir}")
        return None
    
    # Return the first matching score file
    return score_files[0]

def read_score(score_file, benchmark):
    """Read the score from a score file based on the benchmark type."""
    try:
        with open(score_file, 'r') as f:
            data = json.load(f)
        
        score_key = SCORE_KEYS.get(benchmark, "score")
        score = data.get(score_key)
        
        # Multiply by 100 for specific benchmarks before rounding
        if score is not None:
            if benchmark in {"ifeval", "mmlu_redux", "mmlu_redux_cot", "popqa", "arena_hard_v2", "math_500", "zebra_logic", "ifbench"}:
                score = score * 100
            return round(score, 1)  # Round to 1 decimal place
        else:
            print(f"No score found for {score_file} {benchmark}")
            return None
    except Exception as e:
        print(f"Error reading {score_file}: {e}")
        return None

def collect_all_scores(outputs_dir="outputs"):
    """Collect all scores and return as a structured data."""
    results = {}
    
    for model in MODELS:
        results[model] = {}
        for benchmark in BENCHMARKS:
            score_file = find_score_file(outputs_dir, benchmark, model)
            if score_file:
                score = read_score(score_file, benchmark)
                results[model][benchmark] = score
            else:
                results[model][benchmark] = None
    
    return results

def format_for_spreadsheet(results):
    """Format results in a way that's easy to copy to spreadsheet, with fixed-width columns."""
    lines = []
    
    # Column widths
    model_width = 120
    col_width = 5
    
    # Header
    header = (
        "Model".ljust(model_width)[:model_width] +
        " ".join(b.ljust(col_width)[:col_width] for b in BENCHMARKS)
    )
    lines.append(header)
    
    # Data rows
    for model in MODELS:
        model_name = model.ljust(model_width)[:model_width]
        row = [model_name]
        for benchmark in BENCHMARKS:
            score = results[model][benchmark]
            if score is not None:
                score_str = str(score)
            else:
                score_str = "-"
            row.append(score_str.ljust(col_width)[:col_width])
        lines.append(" ".join(row))
    
    return "\n".join(lines)

def main():
    # Change to the eval directory if we're not already there
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    
    print("Collecting scores from all models and benchmarks...")
    results = collect_all_scores()
    
    # Count how many scores we found
    total_scores = sum(1 for model_results in results.values() 
                      for score in model_results.values() 
                      if score is not None)
    
    print(f"Found {total_scores} scores total")
    
    # Format for spreadsheet
    formatted_output = format_for_spreadsheet(results)
    
    # Write to file
    output_file = "outputs/outfiles/prompted_benchmark_scores.txt"
    with open(output_file, 'w') as f:
        f.write(formatted_output)
    
    print(f"Results written to {output_file}")
    print("\nPreview:")
    print(formatted_output[:500] + "..." if len(formatted_output) > 500 else formatted_output)
    
    # Also print some statistics
    print(f"\nStatistics:")
    for benchmark in BENCHMARKS:
        count = sum(1 for model_results in results.values() 
                   if model_results[benchmark] is not None)
        print(f"  {benchmark}: {count}/{len(MODELS)} models have scores")

if __name__ == "__main__":
    main()