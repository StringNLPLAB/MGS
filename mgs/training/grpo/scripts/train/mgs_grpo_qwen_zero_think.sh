export PROJECT_NAME=rlvr_rlmt_thinking
export WANDB_API_KEY=2de3defdedb87a44ee32a7e5d02a764d56e9d765
export LOG=logs/${RUN_NAME}.log

MODEL_PATH=${1:-"models/Qwen/Qwen-2.5-7B-Prompted"}
RUN_NAME=${2:-"mixed_training_proportion_math_0.3_chat_0.3_if_0.3_layer-wise-pcgrad"}

echo "Running: $RUN_NAME"
echo "Model Path: $MODEL_PATH"

python -m verl.trainer.main_ppo_mixed \
  --config-path mgs/training/grpo/configs \
  --config-name mgs_mixed_grpo_qwen_think \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.project_name=$PROJECT_NAME \
  trainer.experiment_name=$RUN_NAME \
  trainer.n_gpus_per_node=8 \
  actor_rollout_ref.model.path=$MODEL_PATH \
  "$@" |& tee "${LOG}"