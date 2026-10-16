"""Test 06: Hardware Configuration Presets (`NUM_GPUS in {1,2,4,8,16}` x `GPU_MEM_GB in {40,80}`).

Section 5.6 requirement:
Generate all presets for `NUM_GPUS in {1, 2, 4, 8, 16}` x `GPU_MEM_GB in {40, 80}` and assert consistency
invariants (parallelism divisibility, batch divisibility, coarse peak memory within GPU limits, model/LoRA selection).
"""
from __future__ import annotations

from sspo.config import (
    VALID_GPU_MEM_GB,
    VALID_NUM_GPUS,
    generate_hardware_preset,
    load_config,
    validate_preset,
)


def test_all_10_hardware_presets_consistency():
    presets = []
    for num_gpus in VALID_NUM_GPUS:
        for gpu_mem_gb in VALID_GPU_MEM_GB:
            p = generate_hardware_preset(num_gpus=num_gpus, gpu_mem_gb=gpu_mem_gb)
            validate_preset(p)
            presets.append(p)

            # Explicit assertions
            assert p["num_gpus"] == num_gpus
            assert p["gpu_mem_gb"] == gpu_mem_gb
            assert num_gpus % p["tensor_parallel_size"] == 0
            assert p["total_sequences_per_step"] % (num_gpus * p["micro_batch_size_per_gpu"]) == 0
            assert p["estimated_peak_mem_gb"] <= 0.90 * gpu_mem_gb

            if num_gpus >= 4 and gpu_mem_gb >= 80:
                assert p["model_name"] == "Qwen/Qwen3-8B"
                assert p["use_lora"] is False
            else:
                assert p["use_lora"] is True or p["model_name"] == "Qwen/Qwen3-4B"

    assert len(presets) == 10

    # Also test loading smoke.yaml and pilot.yaml across all 10 hardware pairs
    for cfg_name in ("configs/smoke.yaml", "configs/pilot.yaml"):
        for num_gpus in VALID_NUM_GPUS:
            for gpu_mem_gb in VALID_GPU_MEM_GB:
                cfg = load_config(cfg_name, num_gpus=num_gpus, gpu_mem_gb=gpu_mem_gb)
                validate_preset(cfg["hardware_preset"])
