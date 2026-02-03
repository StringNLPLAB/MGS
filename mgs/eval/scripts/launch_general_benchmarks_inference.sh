#!/bin/bash -l

MODELS=(
    # Example model - replace with your own model path
    "Llama-3.1-8B-Instruct"
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

# First: if the output directory already has a file that starts with the benchmark then skip
OUTPUT_DIR="outputs/${BENCHMARK}-compare/${MODEL}"
if ls "${OUTPUT_DIR}"/*${BENCHMARK}*.json > /dev/null 2>&1; then
    # Skip all models that have already been run
    # echo "Skipping ${MODEL} - already has a *${BENCHMARK}*.json file"
    continue
else
    # echo "!!! Running ${MODEL} for ${BENCHMARK}"
    : # noop
fi
mkdir -p ${OUTPUT_DIR}


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
N_GPUS=1
TIME=4:00:00

# Set max model length
MAX_MODEL_LEN=$((4096 + MAX_TOKENS))

# if benchmark in [ifeval, mmlu_pro, popqa] then don't skip eval, otherwise skip
if [ "$BENCHMARK" == "ifeval" ] || [ "$BENCHMARK" == "mmlu_redux" ] || [ "$BENCHMARK" == "mmlu_redux_cot" ] || [ "$BENCHMARK" == "popqa" ] || [ "$BENCHMARK" == "math_500" ] || [ "$BENCHMARK" == "zebra_logic" ] || [ "$BENCHMARK" == "ifbench" ]; then
    : # noop
else
    EXTRA_ARGS="$EXTRA_ARGS --skip_eval"
fi

# for wildbench and wildbench_newref, we can try parallel eval
if [ "$BENCHMARK" == "wildbench" ] || [ "$BENCHMARK" == "wildbench_newref" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --parallel_eval"
fi

CMD="python run_benchmarks_sampling.py \
    --benchmark ${BENCHMARK} \
    --model vllm/models/$MODEL \
    --output_dir ${OUTPUT_DIR} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --max_tokens ${MAX_TOKENS} \
    --max_model_len ${MAX_MODEL_LEN} \
    ${EXTRA_ARGS} \
    --force_overwrite"

# >>> set your ENV here!!
if [ -z "$ENV" ]; then
    echo "Error: ENV variable is not set. Please set your conda environment name."
    exit 1
fi

# if job is running then skip
if squeue -h --me -n ${JOB_NAME} | grep -q .; then
    echo "!!! skipping ${JOB_NAME} - already running"
    continue
else
    echo "!!! running ${JOB_NAME}"
fi

MEM=40G

sbatch<<EOT
#!/bin/bash -l
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=1
#SBATCH --output=./joblog/%x-%A_%a.out                          
#SBATCH --gres=gpu:${N_GPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME} 

# >>> source /path/to/conda/profile.d/conda.sh
conda activate $ENV      

echo $CMD
$CMD
EOT

done
done