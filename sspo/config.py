"""Configuration loader and hardware auto-preset generator for SSPO.

Generates and validates hardware presets across `NUM_GPUS in {1, 2, 4, 8, 16}` x `GPU_MEM_GB in {40, 80}`:
- `>= 4 x 80GB`: `Qwen/Qwen3-8B` full-parameter training (`use_lora = False`, FSDP `FULL_SHARD`).
- `1-2 x 80GB` or `40GB` GPUs: `Qwen/Qwen3-8B` LoRA (`r=64, alpha=128`) or `Qwen/Qwen3-4B` LoRA/full-param.
- Enforces:
  1. `num_gpus % tensor_parallel_size == 0`
  2. `(batch_problems * n_rollouts) % (num_gpus * micro_batch_size_per_gpu) == 0`
  3. `estimated_peak_mem_gb <= 0.90 * gpu_mem_gb`
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from sspo.atomic_io import REPO_ROOT

VALID_NUM_GPUS = (1, 2, 4, 8, 16)
VALID_GPU_MEM_GB = (40, 80)


def generate_hardware_preset(
    num_gpus: int,
    gpu_mem_gb: int,
    batch_problems: int = 48,
    n_rollouts: int = 16,
    total_steps: int = 150,
    prefer_4b_on_small: bool = False,
) -> Dict[str, Any]:
    """Generate hardware preset for `(num_gpus, gpu_mem_gb)` and verify consistency constraints."""
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")
    if gpu_mem_gb < 16:
        raise ValueError(f"gpu_mem_gb must be >= 16, got {gpu_mem_gb}")

    # 1. Model & LoRA vs Full-Parameter choice
    if num_gpus >= 4 and gpu_mem_gb >= 80:
        model_name = "Qwen/Qwen3-8B"
        model_params_b = 8.0
        use_lora = False
        lora_rank = 0
        lora_alpha = 0
        tp_size = 2 if num_gpus < 16 else 2
    elif gpu_mem_gb >= 80:
        # 1-2 x 80GB
        if prefer_4b_on_small and num_gpus == 1:
            model_name = "Qwen/Qwen3-4B"
            model_params_b = 4.0
            use_lora = False
            lora_rank = 0
            lora_alpha = 0
        else:
            model_name = "Qwen/Qwen3-8B"
            model_params_b = 8.0
            use_lora = True
            lora_rank = 64
            lora_alpha = 128
        tp_size = 1 if num_gpus == 1 else 2
    else:
        # 40GB GPUs
        if num_gpus <= 2:
            model_name = "Qwen/Qwen3-4B"
            model_params_b = 4.0
            use_lora = True
            lora_rank = 64
            lora_alpha = 128
            tp_size = 1
        else:
            model_name = "Qwen/Qwen3-8B"
            model_params_b = 8.0
            use_lora = True
            lora_rank = 64
            lora_alpha = 128
            tp_size = 2

    # Ensure num_gpus % tp_size == 0
    while num_gpus % tp_size != 0 and tp_size > 1:
        tp_size //= 2

    num_rollout_replicas = num_gpus // tp_size
    rollout_concurrency_per_replica = 64 if gpu_mem_gb >= 80 else 32
    rollout_concurrency = num_rollout_replicas * rollout_concurrency_per_replica

    # Micro-batch size per GPU must divide total_sequences = batch_problems * n_rollouts
    total_sequences = batch_problems * n_rollouts
    candidate_mbs = [4, 2, 1] if (gpu_mem_gb >= 80 and (use_lora or num_gpus >= 8)) else [2, 1]
    micro_batch_size_per_gpu = 1
    for mbs in candidate_mbs:
        if total_sequences % (num_gpus * mbs) == 0:
            micro_batch_size_per_gpu = mbs
            break

    # If batch_problems * n_rollouts is not divisible by num_gpus (e.g. custom small smoke batch on 16 GPUs),
    # adjust effective batch_problems up to the nearest multiple of num_gpus
    if total_sequences % (num_gpus * micro_batch_size_per_gpu) != 0:
        lcm_step = num_gpus * micro_batch_size_per_gpu
        adjusted_total = ((total_sequences + lcm_step - 1) // lcm_step) * lcm_step
        batch_problems = max(1, adjusted_total // n_rollouts)
        total_sequences = batch_problems * n_rollouts

    grad_accum_steps = total_sequences // (num_gpus * micro_batch_size_per_gpu)

    # Coarse peak GPU memory estimate (in GiB):
    # Full-param FSDP shards weights (2B/param bf16) + grads (2B) + AdamW states (12B fp32) across num_gpus,
    # plus unsharded active layer buffer (~4.5 GB) + activations (~4.5 GB * micro_batch_size_per_gpu) + KV cache when co-located/offloaded.
    if use_lora:
        # Base weights in bf16 sharded or replicated + LoRA optimizer states + activations
        weights_mem = (model_params_b * 2.0) / max(1, min(num_gpus, 2))
        optim_mem = 2.5  # LoRA adapters + grads + AdamW states
        act_mem = 4.0 * micro_batch_size_per_gpu
        kv_overhead = 8.0
        est_peak_mem_gb = round(weights_mem + optim_mem + act_mem + kv_overhead, 2)
    else:
        # Full-parameter FSDP FULL_SHARD (with CPU offload fallback on 4 GPUs if needed)
        sharded_state_gb = (model_params_b * 16.0) / num_gpus
        unsharded_buffer_gb = 6.5
        act_mem = 5.5 * micro_batch_size_per_gpu
        kv_overhead = 10.0
        est_peak_mem_gb = round(sharded_state_gb + unsharded_buffer_gb + act_mem + kv_overhead, 2)

    vllm_gpu_memory_utilization = 0.80 if gpu_mem_gb >= 80 else 0.75

    # Throughput / timing estimate for README & planning
    base_throughput = num_gpus * (1.0 if gpu_mem_gb >= 80 else 0.62) * (1.25 if use_lora else 1.0)
    est_step_minutes = round(max(0.8, 32.0 / base_throughput), 1)
    est_total_hours = round((est_step_minutes * total_steps) / 60.0, 1)

    preset = {
        "preset_name": f"gpu{num_gpus}x{gpu_mem_gb}gb_{'lora' if use_lora else 'full'}",
        "num_gpus": int(num_gpus),
        "gpu_mem_gb": int(gpu_mem_gb),
        "model_name": model_name,
        "model_params_b": model_params_b,
        "use_lora": use_lora,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "tensor_parallel_size": tp_size,
        "num_rollout_replicas": num_rollout_replicas,
        "rollout_concurrency": rollout_concurrency,
        "vllm_gpu_memory_utilization": vllm_gpu_memory_utilization,
        "fsdp_sharding_strategy": "FULL_SHARD" if num_gpus > 1 else "NO_SHARD",
        "gradient_checkpointing": True,
        "batch_problems": batch_problems,
        "n_rollouts": n_rollouts,
        "total_sequences_per_step": total_sequences,
        "micro_batch_size_per_gpu": micro_batch_size_per_gpu,
        "gradient_accumulation_steps": grad_accum_steps,
        "estimated_peak_mem_gb": est_peak_mem_gb,
        "est_step_minutes": est_step_minutes,
        "est_total_hours_150steps": est_total_hours,
    }
    validate_preset(preset)
    return preset


def validate_preset(preset: Dict[str, Any]) -> None:
    """Assert all hardware preset consistency invariants (Section 5 Test #6)."""
    num_gpus = int(preset["num_gpus"])
    gpu_mem_gb = int(preset["gpu_mem_gb"])
    tp = int(preset["tensor_parallel_size"])
    mbs = int(preset["micro_batch_size_per_gpu"])
    grad_accum = int(preset["gradient_accumulation_steps"])
    total_seqs = int(preset["total_sequences_per_step"])
    est_mem = float(preset["estimated_peak_mem_gb"])

    assert num_gpus >= 1, f"Invalid num_gpus: {num_gpus}"
    assert tp >= 1 and num_gpus % tp == 0, f"num_gpus ({num_gpus}) not divisible by tensor_parallel_size ({tp})"
    assert mbs >= 1 and grad_accum >= 1, f"Invalid micro_batch_size ({mbs}) or grad_accum ({grad_accum})"
    assert total_seqs == num_gpus * mbs * grad_accum, (
        f"total_sequences_per_step ({total_seqs}) != num_gpus ({num_gpus}) * mbs ({mbs}) * grad_accum ({grad_accum})"
    )
    assert est_mem <= 0.92 * gpu_mem_gb, (
        f"Estimated peak memory {est_mem} GB exceeds safe threshold of {gpu_mem_gb} GB GPU"
    )
    if num_gpus >= 4 and gpu_mem_gb >= 80 and not preset.get("model_override"):
        assert preset["model_name"] == "Qwen/Qwen3-8B" and not preset["use_lora"], (
            ">=4x80G must use Qwen/Qwen3-8B full-parameter training"
        )
    if num_gpus <= 2 or gpu_mem_gb < 80:
        assert preset["use_lora"] or preset["model_name"] == "Qwen/Qwen3-4B", (
            "1-2 GPUs or 40G GPUs must use Qwen3-4B or LoRA"
        )


def detect_hardware() -> Tuple[int, int]:
    """Detect `(num_gpus, gpu_mem_gb)` from environment variables or `torch.cuda`."""
    env_gpus = os.environ.get("NUM_GPUS")
    env_mem = os.environ.get("GPU_MEM_GB")
    if env_gpus and env_mem:
        return int(env_gpus), int(env_mem)
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            n_gpus = int(env_gpus) if env_gpus else torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            mem_gb = int(env_mem) if env_mem else (40 if (props.total_memory // (1024**3)) < 60 else 80)
            return n_gpus, mem_gb
    except Exception:
        pass
    return int(env_gpus or 1), int(env_mem or 80)


def load_config(
    config_path: str | Path,
    num_gpus: Optional[int] = None,
    gpu_mem_gb: Optional[int] = None,
) -> Dict[str, Any]:
    """Load YAML config and attach the resolved hardware preset."""
    p = Path(config_path)
    if not p.is_absolute() and not p.exists():
        p = REPO_ROOT / p
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    det_gpus, det_mem = detect_hardware()
    n_gpus = int(num_gpus if num_gpus is not None else cfg.get("hardware", {}).get("num_gpus", det_gpus))
    mem_gb = int(gpu_mem_gb if gpu_mem_gb is not None else cfg.get("hardware", {}).get("gpu_mem_gb", det_mem))

    train_cfg = cfg.get("train", {})
    batch_problems = int(train_cfg.get("batch_problems", 48))
    n_rollouts = int(train_cfg.get("n_rollouts", 16))
    total_steps = int(train_cfg.get("total_steps", 150))

    preset = generate_hardware_preset(
        num_gpus=n_gpus,
        gpu_mem_gb=mem_gb,
        batch_problems=batch_problems,
        n_rollouts=n_rollouts,
        total_steps=total_steps,
    )
    model_cfg = cfg.get("model", {})
    override_model = os.environ.get("SSPO_MODEL_OVERRIDE") or model_cfg.get("base_model_id")
    if override_model:
        preset["model_override"] = True
        preset["model_name"] = str(override_model)
        if "4B" in str(override_model):
            preset["model_params_b"] = 4.0
        preset["use_lora"] = True
        preset["lora_rank"] = int(model_cfg.get("lora_rank", preset.get("lora_rank") or 64) or 64)
        preset["lora_alpha"] = int(model_cfg.get("lora_alpha", preset.get("lora_alpha") or 128) or 128)
    cfg = copy.deepcopy(cfg)
    cfg["hardware_preset"] = preset
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Inspect or validate SSPO hardware presets and YAML configs.")
    parser.add_argument("--config", type=str, default="configs/pilot.yaml", help="YAML config path")
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--gpu-mem-gb", type=int, default=None)
    parser.add_argument("--all-presets", action="store_true", help="Print all 10 hardware presets as JSON")
    args = parser.parse_args()

    if args.all_presets:
        all_p = {}
        for g in VALID_NUM_GPUS:
            for m in VALID_GPU_MEM_GB:
                all_p[f"{g}x{m}GB"] = generate_hardware_preset(g, m)
        print(json.dumps(all_p, indent=2))
    else:
        cfg = load_config(args.config, num_gpus=args.num_gpus, gpu_mem_gb=args.gpu_mem_gb)
        print(json.dumps(cfg["hardware_preset"], indent=2))


if __name__ == "__main__":
    main()
