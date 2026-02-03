#!/bin/bash -l

MODELS=(
    # Example model - replace with your own model path
    "Llama-3.1-8B-Instruct"
)
BENCHMARKS=(
    "creativewritingv3"
    "alpacaeval2"
    "wildbench"
    "arena_hard_v2"
    # "wildbench_newref"
)

for BENCHMARK in "${BENCHMARKS[@]}"; do
for MODEL in "${MODELS[@]}"; do

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

JOB_NAME="${BENCHMARK}-${MODEL}"
TEMPERATURE=0.7
TOP_P=0.95
MAX_MODEL_LEN=8192

# First: if the output directory already has a file that starts with the benchmark then skip
OUTPUT_DIR="outputs/${BENCHMARK}-compare/${MODEL}"
if ls "${OUTPUT_DIR}"/*${BENCHMARK}*.json > /dev/null 2>&1; then
    if ls "${OUTPUT_DIR}"/*${BENCHMARK}*.json.score > /dev/null 2>&1; then
        # Skip all models that have already been run
        # echo "Skipping ${MODEL} - already has a *${BENCHMARK}*.json file"
        continue
    else
        echo "!!! Running ${MODEL} for ${BENCHMARK}"
    fi
else
    # No output file exists
    echo "!!! Could not find ${OUTPUT_DIR}/*${BENCHMARK}*.json"
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

# >>> source /path/to/conda/profile.d/conda.sh
conda activate $ENV      

mkdir -p ${OUTPUT_DIR}

CMD="python run_benchmarks_sampling.py \
    --benchmark ${BENCHMARK} \
    --model vllm/models/$MODEL \
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