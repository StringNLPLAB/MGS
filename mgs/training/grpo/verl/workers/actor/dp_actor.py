import logging
import os
import random

import torch
import numpy as np
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def _set_grad_tensors(params, grad_tensors):
    """Sets the .grad attribute for each parameter from the provided list of tensors."""
    for p, g in zip(params, grad_tensors):
        if p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.copy_(g)

class LayerWisePCGradUtility:
    """
    Layer-wise PCGrad.
    Applies the projection to each parameter tensor individually.
    Tracks conflicts and cosine similarity for visualization.
    """
    def __init__(self, model):
        # Store parameters AND their names for visualization
        # format: [(name, param), (name, param), ...]
        self.named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.params = [p for n, p in self.named_params]
        self.param_names = [n for n, p in self.named_params]


    def apply_projection(self, task_grad_buffers: dict):
        """
        Args:
            task_grad_buffers: Dict[task_name, List[Tensor]]
        Returns:
            final_grads: List[Tensor]
            metrics: Dict (visualization data)
        """
        task_names = list(task_grad_buffers.keys())
        num_tasks = len(task_names)
        
        # 单任务直接返回
        if num_tasks <= 1:
            final_grads = []
            for i in range(len(self.params)):
                g_sum = torch.zeros_like(self.params[i])
                for t in task_names:
                    if task_grad_buffers[t][i] is not None:
                        g_sum.add_(task_grad_buffers[t][i])
                final_grads.append(g_sum)
            return final_grads, {}

        final_grads = []
        metrics = {}
        
        sorted_task_names = sorted(task_names)
        
        # --- Module-wise 循环 ---
        for p_idx, (param_name, param) in enumerate(self.named_params):
            # 1. Original gradients extraction
            original_grads = {}
            for t in task_names:
                g = task_grad_buffers[t][p_idx]
                if g is None:
                    g = torch.zeros_like(param)
                original_grads[t] = g.detach().clone()

            # 2. analysis
            for i in range(len(sorted_task_names)):
                for j in range(i + 1, len(sorted_task_names)):
                    t1 = sorted_task_names[i]
                    t2 = sorted_task_names[j]
                    
                    g1 = original_grads[t1]
                    g2 = original_grads[t2]
                    
                    # 计算 Cosine Similarity
                    dot_product = torch.sum(g1 * g2)
                    norm_1 = torch.norm(g1)
                    norm_2 = torch.norm(g2)
                    
                    if norm_1 > 1e-8 and norm_2 > 1e-8:
                        cosine = dot_product / (norm_1 * norm_2)
                        metrics[f"layer_cos/{param_name}_{t1}_vs_{t2}"] = cosine.item()
                    else:
                        # 如果梯度为 0，相似度无意义，记为 0
                        metrics[f"layer_cos/{param_name}_{t1}_vs_{t2}"] = 0.0

            # 3. gradient surgery -- PCGrad
            projected_grads = {t: original_grads[t].clone() for t in task_names}
            task_indices = torch.randperm(num_tasks).tolist()
            
            conflict_count = 0 # counter for conflicts

            for i in task_indices:
                task_i = task_names[i]
                g_i = projected_grads[task_i]
                
                other_tasks = [x for x in task_indices if x != i]
                random.shuffle(other_tasks)
                
                for j in other_tasks:
                    task_j = task_names[j]
                    g_j_original = original_grads[task_j] # Note: use original grad for projection
                    
                    dot_product = torch.sum(g_i * g_j_original)
                    
                    if dot_product < 0:
                        conflict_count += 1
                        norm_sq_j = torch.sum(g_j_original * g_j_original)
                        if norm_sq_j > 1e-12:
                            proj = (dot_product / norm_sq_j) * g_j_original
                            g_i.sub_(proj)

            metrics[f"layer_conflict_cnt/{param_name}"] = conflict_count

            # 4. aggregate final gradients
            g_sum = torch.zeros_like(param)
            for t in task_names:
                g_sum.add_(projected_grads[t])
            
            final_grads.append(g_sum)

        return final_grads, metrics

# --- Main Actor Class ---
class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker"""

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        # ... (Existing init code) ...
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        
        # Initialize PCGrad utilities if we are in Actor mode
        if self.actor_optimizer is not None:
            self.pcgrad_util = LayerWisePCGradUtility(self.actor_module)

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    def _apply_pcgrad_and_step(self, task_grad_buffers):
        """
        Apply PCGrad if multiple tasks exist, otherwise standard step.
        Args:
            task_grad_buffers: Dict[task_name, List[Tensor]] - Accumulated gradients
        """
        metrics = {}

        # 1. Single Task Case: Optimization
        if len(task_grad_buffers) <= 1:
            for task_grads in task_grad_buffers.values():
                # Directly set gradients to model parameters
                _set_grad_tensors(self.pcgrad_util.params, task_grads)
            
            grad_norm = self._optimizer_step()
            return grad_norm
        
        # 2. Multiple Tasks Case: Layer-wise PCGrad
        # This now returns Tensors AND a dictionary of layer stats
        final_grad_tensors, pcgrad_metrics = self.pcgrad_util.apply_projection(task_grad_buffers)
        
        # Add PCGrad stats to metrics to be logged
        metrics.update(pcgrad_metrics)
        
        # Set final resolved gradients
        _set_grad_tensors(self.pcgrad_util.params, final_grad_tensors)
        
        # Step
        grad_norm = self._optimizer_step()
        return grad_norm, metrics

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys
                ``input_ids``: tensor of shape [batch_size, sequence_length].
                ``attention_mask``: tensor of shape [batch_size, sequence_length].
                ``position_ids``: tensor of shape [batch_size, sequence_length].
                ``responses``:  tensor of shape [batch_size, response_length].

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        # MUST RETURN A TUPLE
        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]

        # Selection keys setup
        select_keys = [
            "responses", "response_mask", "input_ids", "attention_mask", 
            "position_ids", "old_log_probs", "advantages"
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        non_tensor_select_keys.append("ability")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}

        # --- Helper for Accumulation ---
        def accumulate_grads_to_buffer(buffer_dict, task, current_params):
            """
            Sums current gradients into the buffer.
            buffer_dict[task] is a List[Tensor] matching params.
            """
            if task not in buffer_dict:
                # Initialize buffer with detached clones of current gradients
                buffer_dict[task] = []
                for p in current_params:
                    if p.grad is not None:
                        buffer_dict[task].append(p.grad.detach().clone())
                    else:
                        # Handle case where some params might not have grads (e.g. frozen)
                        buffer_dict[task].append(torch.zeros_like(p))
            else:
                # Accumulate
                for buf, p in zip(buffer_dict[task], current_params):
                    if p.grad is not None:
                        buf.add_(p.grad) 

        for epoch in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                
                # STORAGE FOR PCGRAD
                # Dictionary mapping TaskName -> List[Accumulated Gradient Tensors]
                task_grad_buffer = {} 
                # Dictionary mapping TaskName -> Count of microbatches (for averaging)
                task_counts = {}

                for micro_batch in micro_batches:
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    
                    # Safe task extraction
                    task_type_raw = micro_batch.non_tensor_batch.get("ability")
                    if isinstance(task_type_raw, np.ndarray):
                        task_type = str(task_type_raw.item()) if task_type_raw.size > 0 else "default"
                    else:
                        task_type = str(task_type_raw)
                    
                    # Update count
                    task_counts[task_type] = task_counts.get(task_type, 0) + 1

                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # Forward pass
                    calculate_entropy = (entropy_coeff != 0)
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    loss = policy_loss * loss_scale_factor
                    
                    # --- CRITICAL CHANGE ---
                    # 1. Backward to generate .grad
                    loss.backward()
                    
                    # 2. Accumulate .grad into task_grad_buffer immediately
                    # This prevents storing the computation graph or multiple copies of gradients
                    accumulate_grads_to_buffer(task_grad_buffer, task_type, self.pcgrad_util.params)
                    
                    # 3. Clear .grad immediately to save memory for next microbatch
                    self.actor_optimizer.zero_grad()

                    # Metrics
                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                # --- AVERAGE GRADIENTS ---
                # Before PCGrad, we must average the accumulated gradients by the number of micro-batches
                # otherwise tasks with more micro-batches may have huge gradient magnitudes.
                # Note: The loss is already scaled by loss_scale_factor (approx 1/accum), 
                # but if dynamic batching or uneven tasks occur, we might want explicit averaging.
                # However, loss_scale_factor handles the main global batch scaling. 
                # Since we are using uniform task mixing for now, we can skip this.
                # If you want strict per-task averaging, uncomment below:
                # for t_type in task_grad_buffer:
                #    if task_counts[t_type] > 1:
                #        for g in task_grad_buffer[t_type]:
                #            g.div_(task_counts[t_type]) # Optional: Normalize task vector magnitude
                
                # Apply PCGrad projection and Optimizer Step
                grad_norm, pcgrad_metrics = self._apply_pcgrad_and_step(task_grad_buffer)
                
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                mini_batch_metrics.update(pcgrad_metrics)
                append_to_dict(metrics, mini_batch_metrics)
                
                # Explicit cleanup
                del task_grad_buffer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.actor_optimizer.zero_grad()
        return metrics