"""
PixelPrune Training Script.

Supports two modes (auto-detected from config):
  - Standard CE loss training
  - Knowledge Distillation (KD): alpha * CE + (1 - alpha) * KL

Models: Qwen3-VL / Qwen3.5 (auto-detected via AutoModelForImageTextToText).
"""

import os
import sys
from time import time

import deepspeed
import torch
import torch.distributed as dist

from utils import (
    Logger,
    log_version,
    seed_everything,
    get_args,
    get_dataset,
    get_model,
    get_ref_model,
    fuse,
    sanity_check,
    compute_kd_loss,
    save_hf_checkpoint,
)


def main(hparam):
    fuse()
    sanity_check(hparam)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    seed_everything(hparam["training_config"]["random_seed"] + rank)

    # KD mode: alpha > 0 enables KD (self-distillation if ref_model_path not set)
    kd_config = hparam["training_config"].get("kd_config", {})
    kd_alpha = kd_config.get("alpha", 0)
    kd_temperature = kd_config.get("temperature", 1.0)
    use_kd = kd_alpha > 0

    save_dir = log_version(hparam["training_config"]["save_dir"], rank)
    save_params = {k: v for k, v in hparam.items() if k != "args"}
    save_params["world_size"] = world_size
    log = Logger(logdir=save_dir, hparams=save_params, global_rank=rank)

    log.msg("=" * 50)
    log.msg(f"Log directory: {save_dir}")
    if use_kd:
        log.msg(f"[KD] alpha={kd_alpha}, temperature={kd_temperature}")
    log.msg("=" * 50)

    # ==================== Model Loading ====================
    t0 = time()
    model = get_model(hparam)
    log.msg(f"Model loaded in {time() - t0:.3f}s")

    if use_kd:
        t0 = time()
        ref_model = get_ref_model(hparam)
        log.msg(f"Teacher model loaded in {time() - t0:.3f}s")

    # ==================== Parameter Grouping ====================
    llm_group, mlp_group, vit_group = [], [], []
    for name, param in model.named_parameters():
        if "visual" not in name:
            if not hparam["training_config"]["freeze_llm"]:
                param.requires_grad = True
                llm_group.append(param)
            else:
                param.requires_grad = False
        elif "merger" in name:
            param.requires_grad = True
            mlp_group.append(param)
        else:
            if not hparam["training_config"]["freeze_vit"]:
                param.requires_grad = True
                vit_group.append(param)
            else:
                param.requires_grad = False

    model_parameters = llm_group + vit_group + mlp_group

    # ==================== DeepSpeed Init ====================
    engine, _, _, _ = deepspeed.initialize(
        args=hparam, model=model,
        model_parameters=model_parameters,
        config=hparam["deepspeed_config"],
    )

    if use_kd:
        ref_engine, _, _, _ = deepspeed.initialize(
            args=hparam, model=ref_model,
            config={
                "train_micro_batch_size_per_gpu": hparam["deepspeed_config"]["train_micro_batch_size_per_gpu"],
                "bf16": {"enabled": True},
                "zero_optimization": {"stage": 0},
            },
        )

    # ==================== Training State ====================
    smooth_weight = 0.8
    trained_tokens = 0
    step, update = 0, 0
    log_info = {"smooth_loss": 0}
    if use_kd:
        log_info.update({"smooth_ce_loss": 0, "smooth_kl_loss": 0})

    # ==================== Data Loading ====================
    t0 = time()
    dataloader = get_dataset(hparam)
    log.msg(f"Dataloader loaded in {time() - t0:.3f}s")

    # ==================== Training Loop ====================
    engine.train()
    if use_kd:
        ref_engine.eval()

    mode_str = "KD" if use_kd else "CE"
    log.msg(f"{'=' * 20} start {mode_str} training {'=' * 20}")
    total_num_steps = hparam["deepspeed_config"]["scheduler"]["params"]["total_num_steps"]
    grad_accum_steps = engine.gradient_accumulation_steps()

    acc_data_time = acc_fwd_time = acc_bwd_time = 0.0
    grad_accum_start = training_start = time()
    dataloader_iter = iter(dataloader)

    while True:
        step += 1

        # ---------- Start of gradient accumulation cycle ----------
        if (step - 1) % grad_accum_steps == 0:
            grad_accum_start = time()
            acc_data_time = acc_fwd_time = acc_bwd_time = 0.0
            torch.cuda.reset_peak_memory_stats(device=engine.device)
            acc_loss = acc_tokens = 0
            acc_ce_loss = acc_kl_loss = 0.0
            acc_seq_len = acc_valid_labels = acc_num_examples = 0
            update_trained_tokens = 0

            # Preload batches
            t0 = time()
            accumulated_batches = [next(dataloader_iter) for _ in range(grad_accum_steps)]
            acc_data_time = time() - t0

            # Pre-compute token stats for loss correction
            local_valid_tokens = 0
            for ab in accumulated_batches:
                bs, sl = ab["input_ids"].shape
                acc_seq_len += sl
                vl = ab["labels"].ne(-100).sum().item()
                acc_valid_labels += vl
                local_valid_tokens += vl
                update_trained_tokens += bs * sl

                # Count real examples (exclude padding pseudo-samples)
                pos = ab["position_ids"][0, 0]
                labels = ab["labels"][0]
                starts = pos.eq(0).nonzero(as_tuple=False).view(-1).tolist()
                starts.append(pos.numel())
                acc_num_examples += sum(
                    1 for i in range(len(starts) - 1)
                    if starts[i+1] - starts[i] > 1
                    and labels[starts[i]:starts[i+1]].ne(-100).any().item()
                )

            global_valid_tokens = torch.tensor([local_valid_tokens], device=engine.device, dtype=torch.long)
            dist.all_reduce(global_valid_tokens, op=dist.ReduceOp.SUM)
            global_valid_tokens = global_valid_tokens.item()

        # ---------- Micro-step ----------
        batch = accumulated_batches[(step - 1) % grad_accum_steps]
        batch_size, seq_length = batch["input_ids"].shape
        assert batch_size == 1, "batch size must be 1"
        trained_tokens += batch_size * seq_length

        # Move to device
        model_dtype = next(engine.parameters()).dtype
        batch_dev = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_dev[key] = value.to(engine.device, dtype=model_dtype) if value.dtype.is_floating_point else value.to(engine.device)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                batch_dev[key] = [v.to(engine.device, dtype=model_dtype) if v.dtype.is_floating_point else v.to(engine.device) for v in value]
            else:
                batch_dev[key] = value

        t_fwd = time()

        if use_kd:
            student_batch = {k: v.clone() if hasattr(v, "clone") else v for k, v in batch_dev.items()}
            student_result, student_pos, student_labels = engine(**student_batch, hidden_only=True, no_prune=False)
            teacher_batch = {k: v.clone() if hasattr(v, "clone") else v for k, v in batch_dev.items()}
            teacher_result, teacher_pos, teacher_labels = ref_engine(**teacher_batch, hidden_only=True, no_prune=True)
            loss, ce_loss, kl_loss, num_valid = compute_kd_loss(
                student_hidden_states=student_result,
                teacher_hidden_states=teacher_result,
                student_lm_head_weight=engine.module.lm_head.weight,
                teacher_lm_head_weight=ref_engine.module.lm_head.weight,
                student_labels=student_labels,
                student_position_ids=student_pos,
                teacher_labels=teacher_labels,
                teacher_position_ids=teacher_pos,
                temperature=kd_temperature,
                alpha=kd_alpha,
            )
            acc_ce_loss += ce_loss.detach().item() / num_valid if num_valid > 0 else 0
            acc_kl_loss += kl_loss.detach().item() / num_valid if num_valid > 0 else 0
        else:
            outputs = engine(**batch_dev, return_dict=True, use_cache=False)
            loss = outputs.loss

        t_fwd_end = time()

        micro_valid = batch["labels"].ne(-100).sum().item()
        corrected_loss = loss * (world_size * grad_accum_steps / global_valid_tokens)

        if use_kd:
            acc_loss += loss.detach().item() / num_valid if num_valid > 0 else 0
        else:
            acc_loss += loss.detach().item()
        acc_tokens += micro_valid

        t_bwd = time()
        engine.backward(corrected_loss)
        t_bwd_end = time()
        engine.step()

        acc_fwd_time += t_fwd_end - t_fwd
        acc_bwd_time += t_bwd_end - t_bwd
        del batch_dev

        # ---------- End of gradient accumulation cycle ----------
        if step % grad_accum_steps == 0:
            total_time = time() - grad_accum_start
            update += 1

            if use_kd:
                avg_loss = acc_loss / grad_accum_steps
                avg_ce = acc_ce_loss / grad_accum_steps
                avg_kl = acc_kl_loss / grad_accum_steps
                metric = {"loss": avg_loss, "ce_loss": avg_ce, "kl_loss": avg_kl, "lr": engine.get_lr()[0]}
                log_info["smooth_loss"] = smooth_weight * log_info["smooth_loss"] + (1 - smooth_weight) * avg_loss
                log_info["smooth_ce_loss"] = smooth_weight * log_info["smooth_ce_loss"] + (1 - smooth_weight) * avg_ce
                log_info["smooth_kl_loss"] = smooth_weight * log_info["smooth_kl_loss"] + (1 - smooth_weight) * avg_kl
            else:
                avg_loss = acc_loss / acc_tokens if acc_tokens > 0 else 0.0
                metric = {"loss": avg_loss, "lr": engine.get_lr()[0]}
                log_info["smooth_loss"] = smooth_weight * log_info["smooth_loss"] + (1 - smooth_weight) * avg_loss

            log.metric("train", metric, update)

            # Logging
            if update % hparam["training_config"]["log_step"] == 0:
                r_tokens = torch.tensor([update_trained_tokens], device=engine.device)
                dist.all_reduce(r_tokens, op=dist.ReduceOp.SUM)
                token_speed = r_tokens.item() / total_time
                r_trained = torch.tensor([trained_tokens], device=engine.device)
                dist.all_reduce(r_trained, op=dist.ReduceOp.SUM)
                r_seq = torch.tensor([acc_seq_len], device=engine.device)
                dist.all_reduce(r_seq, op=dist.ReduceOp.AVG)
                r_vlabels = torch.tensor([acc_valid_labels], device=engine.device)
                dist.all_reduce(r_vlabels, op=dist.ReduceOp.AVG)
                r_examples = torch.tensor([acc_num_examples], device=engine.device)
                dist.all_reduce(r_examples, op=dist.ReduceOp.AVG)

                peak_mem = torch.tensor([torch.cuda.max_memory_allocated(device=engine.device)],
                                        device=engine.device, dtype=torch.float64)
                dist.all_reduce(peak_mem, op=dist.ReduceOp.MAX)

                msg = f"| update {update} | speed {token_speed:.0f}tok/s | tokens {r_trained.item():.2E}"
                msg += f" | samples {int(r_examples.item())} | seq {int(r_seq.item())} | vlabels {int(r_vlabels.item())}"
                if use_kd:
                    msg += f" | loss {avg_loss:.4f} | ce {avg_ce:.4f} | kl {avg_kl:.4f}"
                else:
                    msg += f" | loss {avg_loss:.4f}"
                msg += f"\n  data {acc_data_time:.2f}s | fwd {acc_fwd_time:.2f}s | bwd {acc_bwd_time:.2f}s"
                msg += f" | total {total_time:.2f}s | avg {(time() - training_start) / update:.2f}s"
                msg += f" | mem {peak_mem.item() / 1024**3:.1f}GB"
                log.msg(msg)

            # Checkpoint
            if (update % hparam["training_config"]["save_interval"] == 0
                    or update == 10
                    or update == total_num_steps):
                r_loss = torch.tensor([log_info["smooth_loss"]], device=engine.device)
                dist.all_reduce(r_loss, op=dist.ReduceOp.AVG)
                r_trained = torch.tensor([trained_tokens], device=engine.device)
                dist.all_reduce(r_trained, op=dist.ReduceOp.SUM)

                client_id = f"update-{update}-loss-{r_loss.item():.4f}-tokens-{r_trained.item():.2E}"
                save_hf_checkpoint(engine, os.path.join(save_dir, client_id), hparam)

            if update >= total_num_steps:
                dist.barrier()
                log.msg(f"{'=' * 20} {mode_str} training finished {'=' * 20}")
                sys.exit(0)


if __name__ == "__main__":
    hparam = get_args()
    deepspeed.init_distributed(dist_backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    main(hparam)
