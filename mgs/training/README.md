# Training

This half of the codebase contains the code required for training the lanugage models (thinking or non-thinking) with SFT, DPO, PPO, and GRPO.

## GRPO
For PPO and GRPO, we use verl (https://github.com/volcengine/verl).
The corresponding code is provided in `ppo_grpo/`.
For the most part, the code is the same as the verl code in the above link, with the following differences:
- We implement a new Reward Model Worker, the `SequenceRewardModelWorker` in  `ppo_grpo/verl/workers/fsdp_workers.py`. This worker strips thinking traces out based on the longCoT config before scoring with a reward model.
- We allow a `strict` option that treats the response as `null` if the thinking portion is not correctly formatted as per the passed longCoT config. We only use this option for our instruct models as the results for other models seem unaffected by this option.
- We provide data preparation utilities inside `ppo_grpo/scripts/data` to prepare the datasets for training (which will be placed inside `ppo_grpo/data` by default).
- For an explanation of the various hyperparamaters and options, refer to `ppo_grpo/configs/grpo__llamabase__warm-start__think.yaml`'s comments.

Please refer to further documentation provided by verl for details about what the various hyperparameters mean and how to set them.
We provide several configuration files for GRPO in `ppo_grpo/configs/` and corresponding run scripts in `ppo_grpo/scripts/train` to launch them.
These include scripts for Llama-3.1-8B (base and instruct), Qwen2.5-7B (base and instruct), and the prompted/zero versions of these models.
We also provide two example run scripts for PPO in the same directory, along with their corresponding configuration files.
Our runs use FSDPv2 with verl, and are able to fit an 8B model on a single node with 8 H100s.
The provided scripts reflect the hyperparameters used in our experiments, except that they train for 2 epochs (234 steps), whereas we take the checkpoint around step 117 (specifically, step 135) since the model tends to overfit the reward after that.
These runs usually take 14-16 hours to complete with 8 H100s and the 
[Skywork-Reward-Llama-3.1-8B-v0.2](https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2) reward model.