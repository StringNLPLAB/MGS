#!/bin/bash -l

MODELS=(
    # Example model - replace with your own model path
    "openai/gpt-4o"
)
BENCHMARKS=(
    "creativewritingv3"
    "ifbench"
    "mmlu_redux_cot"
    "popqa"
    "alpacaeval2"
    "wildbench"
    "arena_hard_v2"
    # "ifeval"
    # "wildbench_newref"
    # "math_500"
    # "zebra_logic"
)

for BENCHMARK in "${BENCHMARKS[@]}"; do
for MODEL in "${MODELS[@]}"; do

MODEL_BASENAME=$(basename $MODEL)

# Check if we have a longcot config file
# First place to look is inside the model directory, but it may not be there because we use soft links
if [ -f "models/${MODEL}/longcot_config.json" ]; then
    LONGCOT_CONFIG="models/${MODEL}/longcot_config.json"
    EXTRA_ARGS="--longcot_config ${LONGCOT_CONFIG}"
    MAX_TOKENS=8192 # allow for longcot
elif [ -f "outputs/model_configs/${MODEL}.json" ]; then
    LONGCOT_CONFIG="outputs/model_configs/${MODEL}.json"
    EXTRA_ARGS="--longcot_config ${LONGCOT_CONFIG}"
    MAX_TOKENS=8192 # allow for longcot
else
    EXTRA_ARGS=""
    MAX_TOKENS=4096
fi

JOB_NAME="${BENCHMARK}-${MODEL_BASENAME}"
TEMPERATURE=0.7
TOP_P=0.95
MAX_MODEL_LEN=8192

# First: if the output directory already has a file that starts with the benchmark then skip
OUTPUT_DIR="outputs/${BENCHMARK}-compare/${MODEL_BASENAME}"
if ls "${OUTPUT_DIR}"/*${BENCHMARK}*.json.score > /dev/null 2>&1; then
    # Skip all models that have already been run
    # echo "Skipping ${MODEL} - already has a *${BENCHMARK}*.json file"
    continue
fi

# >>> set your ENV here!!
if [ -z "$ENV" ]; then
    echo "Error: ENV variable is not set. Please set your conda environment name."
    exit 1
fi

# for wildbench and wildbench_newref, we can try parallel eval
if [ "$BENCHMARK" == "wildbench" ] || [ "$BENCHMARK" == "wildbench_newref" ] || [ "$BENCHMARK" == "arena_hard_v2" ] || [ "$BENCHMARK" == "creativewritingv3" ] || [ "$BENCHMARK" == "alpacaeval2" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --parallel_eval"
fi

# for large bennchmarks, limit to 1000 samples
if [ "$BENCHMARK" == "mmlu_redux_cot" ] || [ "$BENCHMARK" == "popqa" ] || [ "$BENCHMARK" == "ifeval" ] || [ "$BENCHMARK" == "ifbench" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --test_sample_size 1000"
fi

# >>> source /path/to/conda/profile.d/conda.sh
conda activate $ENV      

mkdir -p ${OUTPUT_DIR}

CMD="python run_benchmarks_sampling.py \
    --benchmark ${BENCHMARK} \
    --model $MODEL \
    --output_dir ${OUTPUT_DIR} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --max_tokens ${MAX_TOKENS} \
    --max_model_len ${MAX_MODEL_LEN} \
    ${EXTRA_ARGS}"

echo $CMD
$CMD

done
done