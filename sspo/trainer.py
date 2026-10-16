"""SSPO Resumable Training and Evaluation Loop (`run_train.sh` & `run_eval.sh`).

Features:
- Arms: `sspo`, `drgrpo_pass1`, `pkpo_pass8`, `checker_reward`, `frozen`.
- Task format: `--turns 1` (single-turn) and `--turns 2` (execution feedback on public examples + 1 revision).
- Dr. GRPO loss:
    `L(theta) = - (1 / (|B| * L_max)) * sum_{x in B} (k / n_x) * sum_i sum_t [ min(rho_{i,t} A_i, clip(rho_{i,t}, 1-eps, 1+eps) A_i) - beta_kl * KL_{i,t} ]`
- Fault-tolerant resume (`--resume` default):
    * Atomic checkpointing every `save_every` steps (default 10) with `checkpoints/latest` pointer.
    * Restores model weights, optimizer state, Python/NumPy/Torch RNG states, data cursor, and per-problem `Sigma^+_x`.
    * Rollout & execution results cached on disk by `(pid, code_hash, checker_version)`.
    * Writes `manifest.json`, `metrics.jsonl`, `eval/<set>_<ckpt>.jsonl`, `summary.json`, and `done` marker.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sspo.advantage import AdvantageResult, compute_arm_advantages
from sspo.atomic_io import (
    REPO_ROOT,
    append_metric_step_atomic,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    has_done_marker,
    read_jsonl,
    to_rel_path,
    write_done_marker,
    write_manifest,
)
from sspo.checker import (
    ExecutionCache,
    check_public_examples,
    extract_representative_program,
    format_public_feedback,
)
from sspo.config import load_config
from sspo.diag import default_mock_rollout_generator
from sspo.ec_common import last_code_block
from sspo.eval import evaluate_problem_rollouts, generate_tables_and_report, summarize_eval_rows


class TinyCausalLM(nn.Module):
    """Lightweight causal Transformer language model for CPU dry-run and unit tests."""

    def __init__(self, vocab_size: int = 256, d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(2048, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2, batch_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(b, seq_len)
        x = self.embed(input_ids % self.vocab_size) + self.pos_embed(positions % 2048)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1).bool()
        h = self.blocks(x, mask=causal_mask)
        return self.lm_head(h)


def compute_token_logprobs(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Compute per-token log-probabilities `log pi(x_t | x_{<t})` of shape `(batch, seq_len - 1)`."""
    ctx = torch.cuda.device(input_ids.device) if input_ids.is_cuda else contextlib.nullcontext()
    with ctx:
        out = model(input_ids, use_cache=False) if "use_cache" in model.forward.__code__.co_varnames else model(input_ids)
        logits = out.logits if hasattr(out, "logits") else out
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        v_size = shift_logits.size(-1)
        token_log_probs = -F.cross_entropy(
            shift_logits.reshape(-1, v_size),
            shift_labels.reshape(-1),
            reduction="none",
        ).view(shift_labels.shape)
        return token_log_probs


def compute_drgrpo_loss(
    policy_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    action_mask: torch.Tensor,
    advantages: torch.Tensor,
    loss_scales: torch.Tensor,
    num_problems_in_batch: int,
    l_max: int,
    clip_eps: float = 0.2,
    kl_coef: float = 0.001,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the exact SSPO / Dr. GRPO loss:

    `L(theta) = - (1 / (|B| * L_max)) * sum_{x in B} (k / n_x) * sum_i sum_t [ min(rho_{i,t} A_i, clip(rho_{i,t}, 1-eps, 1+eps) A_i) - kl_coef * KL_{i,t} ]`

    Args:
        policy_logprobs: `(N_seq, T)` current policy token log-probs
        old_logprobs: `(N_seq, T)` rollout policy token log-probs (detached)
        ref_logprobs: `(N_seq, T)` frozen reference policy token log-probs (detached)
        action_mask: `(N_seq, T)` binary mask (1 for generated action tokens across turns, 0 for prompt/feedback tokens)
        advantages: `(N_seq,)` scalar advantage `A_i` shared across all action tokens of rollout `i`
        loss_scales: `(N_seq,)` problem-level scale `k / n_x` for rollout `i`
        num_problems_in_batch: `|B|`
        l_max: `L_max` constant normalizer
    """
    log_ratio = torch.clamp(policy_logprobs - old_logprobs, min=-20.0, max=20.0)
    rho = torch.exp(log_ratio)
    adv_expanded = advantages.unsqueeze(-1)  # (N_seq, 1)

    surr1 = rho * adv_expanded
    surr2 = torch.clamp(rho, 1.0 - clip_eps, 1.0 + clip_eps) * adv_expanded
    pg_obj_per_token = torch.minimum(surr1, surr2) * action_mask

    # Unbiased non-negative KL estimator: exp(log_ref - log_pi) - (log_ref - log_pi) - 1
    log_ref_ratio = torch.clamp(ref_logprobs - policy_logprobs, min=-20.0, max=20.0)
    kl_per_token = (torch.exp(log_ref_ratio) - log_ref_ratio - 1.0) * action_mask

    seq_scale = loss_scales.unsqueeze(-1)  # (N_seq, 1) -> (k / n_x)
    denom = float(max(1, num_problems_in_batch) * max(1, l_max))

    pg_loss = -(seq_scale * pg_obj_per_token).sum() / denom
    kl_loss = kl_coef * (seq_scale * kl_per_token).sum() / denom
    total_loss = pg_loss + kl_loss

    active_tokens = torch.clamp(action_mask.sum(), min=1.0)
    mean_kl = float((kl_per_token.sum() / active_tokens).detach().cpu().item())
    clip_frac = float(
        (((torch.abs(rho - 1.0) > clip_eps).float() * action_mask).sum() / active_tokens).detach().cpu().item()
    )

    stats = {
        "loss": float(total_loss.detach().cpu().item()),
        "pg_loss": float(pg_loss.detach().cpu().item()),
        "kl": mean_kl,
        "clip_frac": clip_frac,
    }
    return total_loss, stats


def resolve_local_model_path(model_name: str) -> str:
    """Resolve local directory or shared HF cache snapshot if available, else return `model_name`."""
    if os.path.isdir(model_name):
        return model_name
    sanitized = model_name.replace("/", "--")
    local_repo_model = REPO_ROOT / "models" / sanitized
    if local_repo_model.is_dir():
        return str(local_repo_model)
    for hub_root in [
        os.environ.get("HF_HUB_CACHE", ""),
        os.path.join(os.environ.get("HF_HOME", ""), "hub"),
        "/gscratch/simondu/xuanqi26/a40-model-lab/cache/huggingface/hub",
    ]:
        if not hub_root:
            continue
        snap_root = Path(hub_root) / f"models--{sanitized}" / "snapshots"
        if snap_root.is_dir():
            snaps = sorted([p for p in snap_root.iterdir() if p.is_dir()])
            if snaps:
                return str(snaps[-1])
    return model_name


class RolloutEngine:
    """Rollout generator supporting both real `vLLM` (batched + LoRA) and CPU mock mode."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        mock: bool = False,
        mock_fn: Optional[Callable[[Dict[str, Any], int, int, int], List[List[Dict[str, Any]]]]] = None,
        enable_lora: bool = True,
    ):
        self.cfg = cfg
        self.mock = mock
        self.mock_fn = mock_fn or default_mock_rollout_generator
        self.vllm_llm = None
        self.tokenizer = None
        self.enable_lora = enable_lora

        if not self.mock:
            from vllm import LLM

            preset = cfg.get("hardware_preset", {})
            raw_model_name = os.environ.get("SSPO_MODEL_OVERRIDE") or preset.get("model_name", "Qwen/Qwen3-8B")
            model_path = resolve_local_model_path(raw_model_name)
            # If >=2 GPUs and PyTorch training will run on cuda:1, run vLLM on cuda:0 with TP=1
            tp = int(os.environ.get("SSPO_VLLM_TP", preset.get("tensor_parallel_size", 1)))
            if torch.cuda.is_available() and torch.cuda.device_count() >= 2 and os.environ.get("SSPO_SPLIT_DEVICES", "1") == "1":
                tp = 1
            gpu_util = float(os.environ.get("SSPO_VLLM_MEM", preset.get("vllm_gpu_memory_utilization", 0.75)))
            max_len = int(cfg.get("rollout", {}).get("max_context_length", 6144))
            vllm_dtype = "bfloat16"
            max_num_seqs = 64
            enforce_eager = False
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                cap_major = torch.cuda.get_device_capability(0)[0]
                if cap_major < 8:
                    vllm_dtype = "float16"
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if torch.cuda.device_count() == 1 and os.environ.get("SSPO_VLLM_ONLY", "0") != "1":
                    gpu_util = min(gpu_util, 0.47 if total_gb < 30.0 else 0.52)
                    if total_gb < 30.0:
                        max_len = min(max_len, 2560)
                        max_num_seqs = 32
                        enforce_eager = False
                elif total_gb < 30.0:
                    gpu_util = min(gpu_util, 0.72)
                    max_len = min(max_len, 4096)
            max_lora_rank = max(64, int(preset.get("lora_rank", 64) or 64))
            self.max_len = max_len
            self.vllm_llm = LLM(
                model=model_path,
                dtype=vllm_dtype,
                tensor_parallel_size=tp,
                max_model_len=max_len,
                gpu_memory_utilization=gpu_util,
                enable_prefix_caching=True,
                enable_lora=bool(enable_lora),
                max_lora_rank=max_lora_rank,
                max_num_seqs=max_num_seqs,
                enforce_eager=enforce_eager,
            )
            self.tokenizer = self.vllm_llm.get_tokenizer()

    @staticmethod
    def _encode_text_bytes(text: str, max_len: int = 64) -> List[int]:
        raw = [b for b in text.encode("utf-8", "replace")[:max_len]]
        return raw if raw else [1]

    def _format_prompt(self, msgs: List[Dict[str, str]]) -> str:
        assert self.tokenizer is not None
        try:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def sample_problem_trajectories(
        self,
        prob: Dict[str, Any],
        n: int,
        turns: int = 1,
        seed: int = 0,
        lora_path: Optional[str] = None,
        lora_id: int = 1,
    ) -> List[List[Dict[str, Any]]]:
        """Sample `n` rollout trajectories for a single problem."""
        res_map = self.sample_batch_trajectories(
            [prob], n=n, turns=turns, seeds=[seed], lora_path=lora_path, lora_id=lora_id
        )
        return res_map[prob["pid"]]

    def sample_batch_trajectories(
        self,
        probs: List[Dict[str, Any]],
        n: int,
        turns: int = 1,
        seeds: Optional[List[int]] = None,
        lora_path: Optional[str] = None,
        lora_id: int = 1,
    ) -> Dict[str, List[List[Dict[str, Any]]]]:
        """Sample `n` rollout trajectories for a batch of problems in a single batched vLLM call."""
        if not probs:
            return {}
        if seeds is None:
            seeds = [idx for idx in range(len(probs))]

        if self.mock:
            out_map: Dict[str, List[List[Dict[str, Any]]]] = {}
            for prob, s_val in zip(probs, seeds):
                raw_trajs = self.mock_fn(prob, n, turns, s_val)
                out_trajs: List[List[Dict[str, Any]]] = []
                for idx, t_list in enumerate(raw_trajs):
                    enriched_turns: List[Dict[str, Any]] = []
                    t0 = dict(t_list[0])
                    code0 = t0.get("code") or last_code_block(t0.get("text"))
                    pub_all0, pub_frac0, details0 = check_public_examples(code0, prob)
                    p_ids0 = self._encode_text_bytes(prob.get("question", ""), max_len=24)
                    r_ids0 = self._encode_text_bytes(t0.get("text", code0 or "pass"), max_len=32)
                    enriched_turns.append(
                        {
                            "turn_index": 0,
                            "text": t0.get("text", ""),
                            "code": code0,
                            "public_all": pub_all0,
                            "public_frac": pub_frac0,
                            "public_details": details0,
                            "prompt_ids": p_ids0,
                            "response_ids": r_ids0,
                        }
                    )
                    if turns >= 2:
                        fb_text = format_public_feedback(details0)
                        t1_src = t_list[1] if len(t_list) > 1 else t_list[0]
                        code1 = t1_src.get("code") or last_code_block(t1_src.get("text"))
                        pub_all1, pub_frac1, details1 = check_public_examples(code1, prob)
                        p_ids1 = self._encode_text_bytes(fb_text, max_len=24)
                        r_ids1 = self._encode_text_bytes(t1_src.get("text", code1 or "pass"), max_len=32)
                        enriched_turns.append(
                            {
                                "turn_index": 1,
                                "feedback": fb_text,
                                "text": t1_src.get("text", ""),
                                "code": code1,
                                "public_all": pub_all1,
                                "public_frac": pub_frac1,
                                "public_details": details1,
                                "prompt_ids": p_ids1,
                                "response_ids": r_ids1,
                            }
                        )
                    out_trajs.append(enriched_turns)
                out_map[prob["pid"]] = out_trajs
            return out_map

        from vllm import SamplingParams
        from sspo.ec_common import pmap

        assert self.vllm_llm is not None and self.tokenizer is not None
        rollout_cfg = self.cfg.get("rollout", {})
        temp = float(rollout_cfg.get("temperature", 1.0))
        top_p = float(rollout_cfg.get("top_p", 1.0))
        max_tokens = int(rollout_cfg.get("max_tokens", 2048))

        lora_req = None
        if lora_path and os.path.isdir(lora_path) and (Path(lora_path) / "adapter_config.json").exists():
            from vllm.lora.request import LoRARequest

            lora_req = LoRARequest(f"sspo_lora_{lora_id}", int(max(1, lora_id)), str(lora_path))

        prompts_1: List[str] = []
        user_prompts: List[str] = []
        sps_1: List[Any] = []
        for prob, s_val in zip(probs, seeds):
            up = (
                "Solve the following competitive programming problem in Python 3 (standard input / standard output).\n"
                "Provide your complete program inside a single ```python code block.\n\n"
                f"{prob['question']}"
            )
            user_prompts.append(up)
            prompts_1.append(self._format_prompt([{"role": "user", "content": up}]))
            sps_1.append(SamplingParams(n=n, temperature=temp, top_p=top_p, max_tokens=max_tokens, seed=int(s_val)))

        outs_1 = self.vllm_llm.generate(prompts_1, sps_1, lora_request=lora_req, use_tqdm=False)

        # Evaluate public examples across all generated turn-1 programs in parallel
        pub_jobs: List[Tuple[int, int, Optional[str], Dict[str, Any]]] = []
        parsed_1: List[List[Tuple[str, Optional[str], List[int]]]] = []
        p_ids_list: List[List[int]] = []

        for b_idx, (prob, o1) in enumerate(zip(probs, outs_1)):
            p_ids0 = list(o1.prompt_token_ids)
            p_ids_list.append(p_ids0)
            row_parsed = []
            for c_idx, comp in enumerate(o1.outputs):
                text0 = comp.text
                code0 = last_code_block(text0.split("</think>")[-1])
                row_parsed.append((text0, code0, list(comp.token_ids)))
                if turns >= 2:
                    pub_jobs.append((b_idx, c_idx, code0, prob))
            parsed_1.append(row_parsed)

        pub_res_map: Dict[Tuple[int, int], Tuple[bool, float, List[Dict[str, Any]]]] = {}
        if turns >= 2 and pub_jobs:
            pub_results = pmap(lambda item: check_public_examples(item[2], item[3]), pub_jobs)
            for (b_idx, c_idx, _, _), pres in zip(pub_jobs, pub_results):
                pub_res_map[(b_idx, c_idx)] = pres

        out_map = {}
        turn2_reqs: List[Tuple[str, int, int, str, int]] = []  # (pid, b_idx, c_idx, p2_str, seed2)

        for b_idx, prob in enumerate(probs):
            pid = prob["pid"]
            p_ids0 = p_ids_list[b_idx]
            trajs: List[List[Dict[str, Any]]] = []
            for c_idx, (text0, code0, r_ids0) in enumerate(parsed_1[b_idx]):
                pub_all0, pub_frac0, details0 = pub_res_map.get((b_idx, c_idx), (bool(code0), 1.0 if code0 else 0.0, []))
                turn0 = {
                    "turn_index": 0,
                    "text": text0,
                    "code": code0,
                    "public_all": pub_all0,
                    "public_frac": pub_frac0,
                    "public_details": details0,
                    "prompt_ids": p_ids0,
                    "response_ids": r_ids0,
                }
                trajs.append([turn0])
                if turns >= 2 and not pub_all0:
                    fb_str = format_public_feedback(details0)
                    t0_short = text0 if len(text0) <= 3200 else (text0[:1600] + "\n...\n" + text0[-1600:])
                    msgs2 = [
                        {"role": "user", "content": user_prompts[b_idx]},
                        {"role": "assistant", "content": t0_short},
                        {"role": "user", "content": fb_str},
                    ]
                    p2_str = self._format_prompt(msgs2)
                    turn2_reqs.append((pid, b_idx, c_idx, p2_str, int(seeds[b_idx]) + 1000 + c_idx))
            out_map[pid] = trajs

        if turns >= 2 and turn2_reqs:
            p2_prompts = [r[3] for r in turn2_reqs]
            p2_sps = [
                SamplingParams(
                    n=1,
                    temperature=temp,
                    top_p=top_p,
                    max_tokens=min(max_tokens, 1024),
                    seed=r[4],
                )
                for r in turn2_reqs
            ]
            outs_2 = self.vllm_llm.generate(p2_prompts, p2_sps, lora_request=lora_req, use_tqdm=False)
            for (pid, b_idx, c_idx, _, _), o2 in zip(turn2_reqs, outs_2):
                comp2 = o2.outputs[0]
                text2 = comp2.text
                code2 = last_code_block(text2.split("</think>")[-1])
                pub_all2, pub_frac2, details2 = check_public_examples(code2, probs[b_idx])
                out_map[pid][c_idx].append(
                    {
                        "turn_index": 1,
                        "text": text2,
                        "code": code2,
                        "public_all": pub_all2,
                        "public_frac": pub_frac2,
                        "public_details": details2,
                        "prompt_ids": list(o2.prompt_token_ids),
                        "response_ids": list(comp2.token_ids),
                    }
                )

        return out_map


def pack_trajectory_tokens(
    turns: List[Dict[str, Any]],
    max_seq_len: int = 256,
) -> Tuple[List[int], List[float]]:
    """Concatenate prompt and response token IDs across turns and build the binary `action_mask` (1 on response tokens).

    Returns `(input_ids, action_mask_shifted)` where `len(action_mask_shifted) == len(input_ids) - 1`.
    """
    input_ids: List[int] = []
    is_action: List[float] = []
    for t in turns:
        p_ids = [int(x) for x in t.get("prompt_ids", [1, 2])]
        r_ids = [int(x) for x in t.get("response_ids", [3, 4])]
        input_ids.extend(p_ids)
        is_action.extend([0.0] * len(p_ids))
        input_ids.extend(r_ids)
        is_action.extend([1.0] * len(r_ids))

    if len(input_ids) < 2:
        input_ids = [1, 2]
        is_action = [0.0, 1.0]

    input_ids = input_ids[:max_seq_len]
    is_action = is_action[:max_seq_len]
    # Shift mask to align with target tokens input_ids[1:]
    return input_ids, is_action[1:]


def save_checkpoint_atomic(
    run_dir: Path,
    step: int,
    policy_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_cursor: int,
    sigma_plus_map: Dict[str, Set[str]],
    prev_adv_stds: Dict[str, float],
) -> Path:
    """Atomically write checkpoint directory `checkpoints/step_XXXX` and update `checkpoints/latest` pointer."""
    ckpt_root = run_dir / "checkpoints"
    ckpt_name = f"step_{step:04d}"
    ckpt_dir = ckpt_root / ckpt_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # For PEFT LoRA models, save only trainable adapter parameters + HF peft directory to save disk space
    trainable_names = {name for name, p in policy_model.named_parameters() if p.requires_grad}
    if hasattr(policy_model, "peft_config"):
        state_to_save = {k: v.detach().cpu() for k, v in policy_model.state_dict().items() if k in trainable_names or "lora_" in k}
        lora_dir = ckpt_dir / "lora_adapter"
        policy_model.save_pretrained(str(lora_dir))
    else:
        state_to_save = {k: v.detach().cpu() for k, v in policy_model.state_dict().items()}

    tmp_model = ckpt_dir / ".model.pt.tmp"
    torch.save(state_to_save, tmp_model)
    os.replace(tmp_model, ckpt_dir / "model.pt")

    tmp_opt = ckpt_dir / ".optimizer.pt.tmp"
    torch.save(optimizer.state_dict(), tmp_opt)
    os.replace(tmp_opt, ckpt_dir / "optimizer.pt")

    tmp_rng = ckpt_dir / ".rng_state.pt.tmp"
    rng_payload = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    torch.save(rng_payload, tmp_rng)
    os.replace(tmp_rng, ckpt_dir / "rng_state.pt")

    trainer_state = {
        "step": int(step),
        "data_cursor": int(data_cursor),
        "sigma_plus": {k: sorted(v) for k, v in sigma_plus_map.items()},
        "prev_adv_stds": prev_adv_stds,
    }
    atomic_write_json(ckpt_dir / "trainer_state.json", trainer_state)
    atomic_write_text(ckpt_root / "latest", ckpt_name + "\n")
    return ckpt_dir


def load_latest_checkpoint(
    run_dir: Path,
    policy_model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Optional[Dict[str, Any]]:
    """Load the checkpoint pointed to by `checkpoints/latest` if it exists."""
    latest_file = run_dir / "checkpoints" / "latest"
    if not latest_file.exists():
        return None
    ckpt_name = latest_file.read_text(encoding="utf-8").strip()
    if not ckpt_name:
        return None
    ckpt_dir = run_dir / "checkpoints" / ckpt_name
    state_file = ckpt_dir / "trainer_state.json"
    if not state_file.exists():
        return None

    saved_sd = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=True)
    strict_load = not hasattr(policy_model, "peft_config")
    policy_model.load_state_dict(saved_sd, strict=strict_load)
    try:
        opt_sd = torch.load(ckpt_dir / "optimizer.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(opt_sd)
        for p, state in optimizer.state.items():
            for k, v in list(state.items()):
                if torch.is_tensor(v):
                    if k == "step":
                        state[k] = v.cpu()
                    else:
                        state[k] = v.to(device=p.device, dtype=p.dtype)
    except Exception:
        optimizer.state.clear()
    if (ckpt_dir / "rng_state.pt").exists():
        rng_state = torch.load(ckpt_dir / "rng_state.pt", map_location="cpu", weights_only=False)
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])

    return json.loads(state_file.read_text(encoding="utf-8"))


def run_training(
    config_path: str | Path,
    arm: str = "sspo",
    seed: int = 0,
    run_dir: Optional[str | Path] = None,
    turns: Optional[int] = None,
    max_steps: Optional[int] = None,
    stop_after_step: Optional[int] = None,
    resume: bool = True,
    mock: bool = False,
    mock_rollout_fn: Optional[Callable[[Dict[str, Any], int, int, int], List[List[Dict[str, Any]]]]] = None,
) -> Dict[str, Any]:
    """Run or resume training for one arm (`sspo`, `drgrpo_pass1`, `pkpo_pass8`, `checker_reward`, `frozen`)."""
    cfg = load_config(config_path)
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    rollout_cfg = cfg.get("rollout", {})
    checker_cfg = cfg.get("checker", {})
    preset = cfg["hardware_preset"]

    if turns is not None:
        rollout_cfg["turns"] = int(turns)
    n_turns = int(rollout_cfg.get("turns", 1))

    exp_run_dir = Path(run_dir) if run_dir else (REPO_ROOT / "results" / f"{arm}_seed{seed}")
    exp_run_dir.mkdir(parents=True, exist_ok=True)

    # Check if already completed
    if resume and has_done_marker(exp_run_dir):
        return {
            "status": "already_done",
            "run_dir": to_rel_path(exp_run_dir),
            "arm": arm,
            "seed": seed,
        }

    # Write manifest.json with relative paths
    write_manifest(
        run_dir=exp_run_dir,
        config=cfg,
        arm=arm,
        seed=seed,
        num_gpus=preset["num_gpus"],
        gpu_mem_gb=preset["gpu_mem_gb"],
        preset=preset,
        model_name=preset["model_name"],
    )

    # Handle `frozen` arm (no training steps; writes step_0000 pointer and done marker)
    if arm == "frozen":
        ckpt_root = exp_run_dir / "checkpoints" / "step_0000"
        ckpt_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(ckpt_root / "trainer_state.json", {"step": 0, "arm": "frozen"})
        atomic_write_text(exp_run_dir / "checkpoints" / "latest", "step_0000\n")
        atomic_write_jsonl(exp_run_dir / "metrics.jsonl", [])
        write_done_marker(exp_run_dir, {"arm": "frozen", "steps": 0})
        return {"status": "completed", "run_dir": to_rel_path(exp_run_dir), "arm": arm, "steps_run": 0}

    # Seed RNGs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    use_mock = bool(mock or cfg.get("mode") == "smoke" or os.environ.get("SSPO_MOCK_LLM", "0") == "1")

    # Initialize RolloutEngine FIRST so vLLM profiles cuda:0 cleanly before PyTorch allocates training tensors
    rollout_engine = RolloutEngine(cfg, mock=use_mock, mock_fn=mock_rollout_fn, enable_lora=True)

    # Initialize policy model & frozen reference model
    ref_model: Optional[nn.Module] = None
    if use_mock or not torch.cuda.is_available():
        policy_model: nn.Module = TinyCausalLM(vocab_size=256, d_model=64, n_heads=4, n_layers=2)
        ref_model = copy.deepcopy(policy_model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        device = torch.device("cpu")
    else:
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model

        # Place PyTorch training model on cuda:1 if >=2 GPUs so cuda:0 is 100% dedicated to vLLM
        if torch.cuda.device_count() >= 2 and os.environ.get("SSPO_SPLIT_DEVICES", "1") == "1":
            device = torch.device("cuda:1")
        else:
            device = torch.device("cuda:0")

        raw_model_name = os.environ.get("SSPO_MODEL_OVERRIDE") or preset["model_name"]
        model_path = resolve_local_model_path(raw_model_name)
        cap_major = torch.cuda.get_device_capability(device)[0]
        pt_dtype = torch.float16 if cap_major < 8 else torch.bfloat16
        policy_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=pt_dtype,
            attn_implementation="sdpa",
        ).to(device)
        policy_model.gradient_checkpointing_enable()
        policy_model.enable_input_require_grads()

        lora_r = int(preset.get("lora_rank", 64) or 32)
        lora_alpha = int(preset.get("lora_alpha", 128) or (lora_r * 2))
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        policy_model = get_peft_model(policy_model, lora_cfg)

    lr = float(train_cfg.get("lr", 1e-6))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
        foreach=False,
    )

    # Load training problems (prefer train_active.jsonl produced by diag if present)
    active_train_path = (
        REPO_ROOT / "data" / "train_active_300.jsonl"
        if (REPO_ROOT / "data" / "train_active_300.jsonl").exists()
        else (REPO_ROOT / "data" / "train_active.jsonl")
    )
    default_train_path = REPO_ROOT / data_cfg.get("train_file", "data/train.jsonl")
    sample_path = REPO_ROOT / "data" / "sample_problems.jsonl"
    if active_train_path.exists() and len(read_jsonl(active_train_path)) > 0:
        problems = read_jsonl(active_train_path)
    elif default_train_path.exists() and len(read_jsonl(default_train_path)) > 0:
        problems = read_jsonl(default_train_path)
    else:
        problems = [p for p in read_jsonl(sample_path) if not p["pid"].startswith("sample-bad")]

    if not problems:
        raise RuntimeError("No training problems found")

    # Initialize Sigma^+_x from reference solutions if available
    cache_root = REPO_ROOT / checker_cfg.get("cache_dir", "cache")
    cache = ExecutionCache(cache_root=cache_root, checker_version=str(checker_cfg.get("version", "v1")))
    sigma_plus_map: Dict[str, Set[str]] = {}
    ref_items = [(p["pid"], p["reference_solution"], p) for p in problems if p.get("reference_solution")]
    if ref_items:
        ref_evs = cache.evaluate_batch_programs(ref_items)
        for (pid, _, _), ref_ev in zip(ref_items, ref_evs):
            sigma_plus_map.setdefault(pid, set())
            if ref_ev["chk"] == 1 and ref_ev["g"] == 1:
                sigma_plus_map[pid].add(ref_ev["sig"])
    for prob in problems:
        sigma_plus_map.setdefault(prob["pid"], set())
    cache.reset_counters()

    start_step = 1
    data_cursor = 0
    prev_adv_stds: Dict[str, float] = {}
    active_lora_dir = exp_run_dir / "checkpoints" / "active_lora"

    batch_problems = min(len(problems), int(train_cfg.get("batch_problems", 48)))

    if resume:
        loaded_state = load_latest_checkpoint(exp_run_dir, policy_model, optimizer)
        if loaded_state is not None:
            start_step = int(loaded_state["step"]) + 1
            data_cursor = int(loaded_state.get("data_cursor", 0))
            for k_pid, sigs in loaded_state.get("sigma_plus", {}).items():
                sigma_plus_map[k_pid] = set(sigs)
            prev_adv_stds = {k: float(v) for k, v in loaded_state.get("prev_adv_stds", {}).items()}
            if hasattr(policy_model, "peft_config"):
                policy_model.save_pretrained(str(active_lora_dir))
        if (
            (exp_run_dir / "metrics.jsonl").exists()
            and hasattr(policy_model, "peft_config")
            and (active_lora_dir / "adapter_model.safetensors").exists()
        ):
            existing_metrics = read_jsonl(exp_run_dir / "metrics.jsonl")
            if existing_metrics:
                last_m_step = max(int(r.get("step", 0)) for r in existing_metrics if not r.get("stale"))
                if last_m_step >= start_step:
                    start_step = last_m_step + 1
                    data_cursor = last_m_step * batch_problems
                    try:
                        from peft import set_peft_model_state_dict
                        from safetensors.torch import load_file

                        sd_lora = {
                            k: v.to(device=device, dtype=pt_dtype)
                            for k, v in load_file(str(active_lora_dir / "adapter_model.safetensors")).items()
                        }
                        set_peft_model_state_dict(policy_model, sd_lora)
                    except Exception:
                        pass

    if max_steps is not None:
        total_steps = int(max_steps)
    elif arm == "checker_reward":
        total_steps = int(train_cfg.get("checker_reward_steps", 50))
    else:
        total_steps = int(train_cfg.get("total_steps", 150))
    n_rollouts = int(train_cfg.get("n_rollouts", 16))
    k_deploy = min(n_rollouts, int(train_cfg.get("k_deploy", 8)))
    alpha = float(train_cfg.get("alpha", 0.7))
    center_select = str(train_cfg.get("center_select", "none"))
    clip_eps = float(train_cfg.get("clip_eps", 0.2))
    kl_coef = float(train_cfg.get("kl_coef", 0.001))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    save_every = int(train_cfg.get("save_every", 10))
    l_max = int(rollout_cfg.get("max_tokens", 2048)) if not use_mock else 64
    max_train_seq_len = 128 if use_mock else int(os.environ.get("SSPO_MAX_TRAIN_SEQ_LEN", "1536"))
    micro_bs = max(1, int(preset.get("micro_batch_size", 2))) if not use_mock else 64
    if not use_mock and torch.cuda.is_available() and torch.cuda.device_count() == 1:
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if total_gb < 30.0:
            micro_bs = 1
            max_train_seq_len = min(max_train_seq_len, 1024)

    rollout_cache_dir = exp_run_dir / "rollout_cache"
    rollout_cache_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_run_dir / "metrics.jsonl"

    steps_executed = 0
    for step in range(start_step, total_steps + 1):
        t_step_start = time.time()
        batch: List[Dict[str, Any]] = []
        for _ in range(batch_problems):
            batch.append(problems[data_cursor % len(problems)])
            data_cursor += 1

        # Rollout & execution evaluation (with step rollout cache for exact resume if killed mid-step)
        step_rollout_file = rollout_cache_dir / f"step_{step:04d}.json"
        cached_step_rollouts: Dict[str, List[List[Dict[str, Any]]]] = {}
        if step_rollout_file.exists():
            try:
                cached_step_rollouts = json.loads(step_rollout_file.read_text(encoding="utf-8"))
            except Exception:
                cached_step_rollouts = {}

        missing_probs: List[Dict[str, Any]] = []
        missing_seeds: List[int] = []
        for p_idx, prob in enumerate(batch):
            if prob["pid"] not in cached_step_rollouts:
                missing_probs.append(prob)
                missing_seeds.append(seed * 100000 + step * 100 + p_idx)

        if missing_probs:
            lora_path_for_step = str(active_lora_dir) if (step > 1 and active_lora_dir.exists()) else None
            sampled_map = rollout_engine.sample_batch_trajectories(
                missing_probs,
                n=n_rollouts,
                turns=n_turns,
                seeds=missing_seeds,
                lora_path=lora_path_for_step,
                lora_id=step,
            )
            cached_step_rollouts.update(sampled_map)
            atomic_write_json(step_rollout_file, cached_step_rollouts)

        # Evaluate all candidate representative programs across the batch in parallel
        eval_items: List[Tuple[str, Optional[str], dict]] = []
        batch_trajs_list: List[List[List[Dict[str, Any]]]] = []
        for prob in batch:
            pid = prob["pid"]
            trajs = cached_step_rollouts[pid]
            batch_trajs_list.append(trajs)
            for t_turns in trajs:
                rep = extract_representative_program(t_turns, prob=prob)
                eval_items.append((pid, rep["code"], prob))

        all_eval_results = cache.evaluate_batch_programs(eval_items)
        eval_cursor = 0

        all_seq_ids: List[List[int]] = []
        all_seq_masks: List[List[float]] = []
        all_seq_advs: List[float] = []
        all_seq_scales: List[float] = []

        n_active_problems = 0
        n_dropped_zero_adv = 0
        sum_n_C = 0
        sum_n_T = 0
        sum_n_W = 0
        sum_n_W_star = 0
        sum_n_clusters = 0
        n_impure_problems = 0
        eps_hat_list: List[float] = []
        chk_pass_list: List[float] = []
        pass_at_1_list: List[float] = []
        adv_all_vals: List[float] = []
        adv_cov_vals: List[float] = []
        adv_sel_vals: List[float] = []
        neg_wrong_violations = 0

        for p_idx, (prob, trajs) in enumerate(zip(batch, batch_trajs_list)):
            pid = prob["pid"]
            g_list: List[int] = []
            chk_list: List[int] = []
            sig_list: List[str] = []
            for _ in trajs:
                ev = all_eval_results[eval_cursor]
                eval_cursor += 1
                g_list.append(ev["g"])
                chk_list.append(ev["chk"])
                sig_list.append(ev["sig"])

            resp_lens: List[int] = []
            for t_turns in trajs:
                _, sm = pack_trajectory_tokens(t_turns, max_seq_len=max_train_seq_len)
                resp_lens.append(int(sum(1 for x in sm if x > 0.5)))

            adv_res: AdvantageResult = compute_arm_advantages(
                arm=arm,
                g_list=g_list,
                chk_list=chk_list,
                sig_list=sig_list,
                k=k_deploy,
                alpha=alpha,
                sigma_plus=sigma_plus_map.get(pid, set()),
                center_select=center_select,
                pid=pid,
                step=step,
                total_steps=total_steps,
                response_lengths=resp_lens,
                turn_count=n_turns,
            )

            # Update Sigma^+_x for future steps
            sigma_plus_map[pid] = adv_res.updated_sigma_plus
            prev_adv_stds[pid] = float(np.std(adv_res.advantages))

            grp = adv_res.group
            sum_n_C += grp.n_C
            sum_n_T += grp.n_T
            sum_n_W += grp.n_W_total
            sum_n_W_star += grp.n_W_star
            sum_n_clusters += len(grp.w_clusters)
            if grp.cluster_impure:
                n_impure_problems += 1
            eps_hat_list.append(grp.eps_hat)
            chk_pass_list.append(grp.chk_pass_rate)
            pass_at_1_list.append(float(np.mean(g_list)) if g_list else 0.0)

            adv_all_vals.extend(adv_res.advantages.tolist())
            adv_cov_vals.extend(adv_res.adv_cov_centered.tolist())
            adv_sel_vals.extend(adv_res.adv_sel_used.tolist())

            if arm == "sspo" and center_select == "none" and not grp.cluster_impure:
                for i_cand, g_val in enumerate(g_list):
                    if g_val == 0 and adv_res.advantages[i_cand] > 1e-9:
                        neg_wrong_violations += 1

            # Drop problems where all A_i == 0
            if adv_res.all_zero:
                n_dropped_zero_adv += 1
                continue

            n_active_problems += 1
            for i_cand, t_turns in enumerate(trajs):
                seq_ids, seq_mask = pack_trajectory_tokens(t_turns, max_seq_len=max_train_seq_len)
                all_seq_ids.append(seq_ids)
                all_seq_masks.append(seq_mask)
                all_seq_advs.append(float(adv_res.advantages[i_cand]))
                all_seq_scales.append(float(adv_res.loss_scale))

        # Policy gradient + KL loss and AdamW step (micro-batched to bound peak VRAM)
        if all_seq_ids:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            n_total_seqs = len(all_seq_ids)
            old_lp_chunks: List[torch.Tensor] = []
            ref_lp_chunks: List[torch.Tensor] = []
            mb_ranges = [(i, min(n_total_seqs, i + micro_bs)) for i in range(0, n_total_seqs, micro_bs)]

            policy_model.eval()
            with torch.no_grad():
                for s_i, e_i in mb_ranges:
                    sub_ids = all_seq_ids[s_i:e_i]
                    sub_max = max(len(s) for s in sub_ids)
                    pad_ids = torch.tensor([s + [0] * (sub_max - len(s)) for s in sub_ids], dtype=torch.long, device=device)
                    old_lp = compute_token_logprobs(policy_model, pad_ids).detach().cpu()
                    old_lp_chunks.append(old_lp)
                    if ref_model is not None:
                        ref_lp = compute_token_logprobs(ref_model, pad_ids).detach().cpu()
                    elif hasattr(policy_model, "disable_adapter"):
                        with policy_model.disable_adapter():
                            ref_lp = compute_token_logprobs(policy_model, pad_ids).detach().cpu()
                    else:
                        ref_lp = old_lp.clone()
                    ref_lp_chunks.append(ref_lp)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            policy_model.train()
            optimizer.zero_grad()
            total_loss_val = 0.0
            total_pg_val = 0.0
            total_kl_sum = 0.0
            total_clip_sum = 0.0
            total_active_tokens = 0.0

            for chunk_idx, (s_i, e_i) in enumerate(mb_ranges):
                sub_ids = all_seq_ids[s_i:e_i]
                sub_masks = all_seq_masks[s_i:e_i]
                sub_max = max(len(s) for s in sub_ids)
                pad_ids = torch.tensor([s + [0] * (sub_max - len(s)) for s in sub_ids], dtype=torch.long, device=device)
                pad_masks = torch.tensor(
                    [m + [0.0] * (sub_max - 1 - len(m)) for m in sub_masks], dtype=torch.float32, device=device
                )
                sub_adv = torch.tensor(all_seq_advs[s_i:e_i], dtype=torch.float32, device=device)
                sub_scale = torch.tensor(all_seq_scales[s_i:e_i], dtype=torch.float32, device=device)

                curr_lp = compute_token_logprobs(policy_model, pad_ids)
                if arm == "agent":
                    from benchmark.agent_algorithm import PolicyLossBatch, compute_policy_loss

                    pl_batch = PolicyLossBatch(
                        policy_logprobs=curr_lp,
                        old_logprobs=old_lp_chunks[chunk_idx].to(device),
                        ref_logprobs=ref_lp_chunks[chunk_idx].to(device),
                        action_mask=pad_masks,
                        advantages=sub_adv,
                        loss_scales=sub_scale,
                        num_problems_in_batch=len(batch),
                        l_max=l_max,
                        clip_eps=clip_eps,
                        kl_coef=kl_coef,
                        step=step,
                        total_steps=total_steps,
                    )
                    res_loss = compute_policy_loss(pl_batch)
                    if isinstance(res_loss, tuple):
                        mb_loss, mb_stats = res_loss
                    else:
                        mb_loss = res_loss
                        mb_stats = {
                            "loss": float(mb_loss.detach().item()),
                            "pg_loss": float(mb_loss.detach().item()),
                            "kl": 0.0,
                            "clip_frac": 0.0,
                        }
                else:
                    mb_loss, mb_stats = compute_drgrpo_loss(
                        policy_logprobs=curr_lp,
                        old_logprobs=old_lp_chunks[chunk_idx].to(device),
                        ref_logprobs=ref_lp_chunks[chunk_idx].to(device),
                        action_mask=pad_masks,
                        advantages=sub_adv,
                        loss_scales=sub_scale,
                        num_problems_in_batch=len(batch),
                        l_max=l_max,
                        clip_eps=clip_eps,
                        kl_coef=kl_coef,
                    )
                mb_loss.backward()
                del curr_lp, mb_loss
                n_tok_mb = float(pad_masks.sum().item())
                total_loss_val += mb_stats["loss"]
                total_pg_val += mb_stats["pg_loss"]
                total_kl_sum += mb_stats["kl"] * n_tok_mb
                total_clip_sum += mb_stats["clip_frac"] * n_tok_mb
                total_active_tokens += n_tok_mb

            grad_norm = float(torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm).item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(policy_model, "peft_config"):
                active_lora_dir.mkdir(parents=True, exist_ok=True)
                policy_model.save_pretrained(str(active_lora_dir))

            denom_tok = max(1.0, total_active_tokens)
            loss_stats = {
                "loss": total_loss_val,
                "pg_loss": total_pg_val,
                "kl": total_kl_sum / denom_tok,
                "clip_frac": total_clip_sum / denom_tok,
            }
        else:
            loss_stats = {"loss": 0.0, "pg_loss": 0.0, "kl": 0.0, "clip_frac": 0.0}
            grad_norm = 0.0

        wall_s = round(time.time() - t_step_start, 4)
        nb = max(1, len(batch))
        adv_arr = np.array(adv_all_vals, dtype=np.float64) if adv_all_vals else np.zeros(1)
        metric_row = {
            "step": step,
            "arm": arm,
            "seed": seed,
            "n_problems": len(batch),
            "n_active_problems": n_active_problems,
            "n_dropped_zero_adv": n_dropped_zero_adv,
            "counts": {
                "n_C": sum_n_C,
                "n_T": sum_n_T,
                "n_W": sum_n_W,
                "n_W_star": sum_n_W_star,
                "n_clusters_mean": round(sum_n_clusters / nb, 3),
            },
            "adv_stats": {
                "mean": round(float(np.mean(adv_arr)), 6),
                "std": round(float(np.std(adv_arr)), 6),
                "max": round(float(np.max(adv_arr)), 6),
                "min": round(float(np.min(adv_arr)), 6),
                "cov_mean": round(float(np.mean(adv_cov_vals)) if adv_cov_vals else 0.0, 6),
                "sel_mean": round(float(np.mean(adv_sel_vals)) if adv_sel_vals else 0.0, 6),
                "neg_wrong_viol": int(neg_wrong_violations),
            },
            "loss": round(loss_stats["loss"], 6),
            "pg_loss": round(loss_stats["pg_loss"], 6),
            "kl": round(loss_stats["kl"], 6),
            "clip_frac": round(loss_stats["clip_frac"], 6),
            "grad_norm": round(grad_norm, 6),
            "cluster_impure_rate": round(n_impure_problems / nb, 6),
            "eps_hat": round(float(np.mean(eps_hat_list)) if eps_hat_list else 0.0, 6),
            "chk_pass_rate": round(float(np.mean(chk_pass_list)) if chk_pass_list else 0.0, 6),
            "pass_at_1_batch": round(float(np.mean(pass_at_1_list)) if pass_at_1_list else 0.0, 6),
            "wall_time_s": wall_s,
            "cache_hit_rate": round(cache.hit_rate, 6),
        }
        append_metric_step_atomic(metrics_path, step=step, row=metric_row)
        steps_executed += 1

        should_save = (
            (step % save_every == 0)
            or (step == total_steps)
            or (stop_after_step is not None and step >= stop_after_step)
        )
        if should_save:
            save_checkpoint_atomic(
                run_dir=exp_run_dir,
                step=step,
                policy_model=policy_model,
                optimizer=optimizer,
                data_cursor=data_cursor,
                sigma_plus_map=sigma_plus_map,
                prev_adv_stds=prev_adv_stds,
            )

        if stop_after_step is not None and step >= stop_after_step and step < total_steps:
            return {
                "status": "interrupted_for_resume_test",
                "run_dir": to_rel_path(exp_run_dir),
                "arm": arm,
                "last_step": step,
                "steps_run": steps_executed,
                "cache_hit_rate": cache.hit_rate,
            }

    write_done_marker(exp_run_dir, {"arm": arm, "seed": seed, "total_steps": total_steps})
    return {
        "status": "completed",
        "run_dir": to_rel_path(exp_run_dir),
        "arm": arm,
        "seed": seed,
        "last_step": total_steps,
        "steps_run": steps_executed,
        "cache_hit_rate": cache.hit_rate,
    }


def run_evaluation(
    run_dir: str | Path,
    config_path: str | Path = "configs/pilot.yaml",
    ckpt: str = "latest",
    eval_split: str = "test",
    resume: bool = True,
    mock: bool = False,
) -> Dict[str, Any]:
    """Evaluate a trained (or `frozen`) run directory on `test` or `dev` split and update `results/report.md`."""
    rdir = Path(run_dir)
    if not rdir.is_absolute() and not rdir.exists():
        rdir = REPO_ROOT / rdir
    rdir.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint name
    if ckpt == "latest" and (rdir / "checkpoints" / "latest").exists():
        ckpt_name = (rdir / "checkpoints" / "latest").read_text(encoding="utf-8").strip()
    else:
        ckpt_name = ckpt

    eval_out_dir = rdir / "eval"
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    eval_jsonl_path = eval_out_dir / f"{eval_split}_{ckpt_name}.jsonl"
    eval_done_marker = eval_out_dir / f".{eval_split}_{ckpt_name}.done"

    if resume and eval_done_marker.exists() and eval_jsonl_path.exists():
        rows = read_jsonl(eval_jsonl_path)
        summary = summarize_eval_rows(rows, run_name=rdir.name)
        generate_tables_and_report(rdir.parent)
        return summary

    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("eval", {})
    checker_cfg = cfg.get("checker", {})
    rollout_cfg = cfg.get("rollout", {})

    split_file = REPO_ROOT / data_cfg.get(f"{eval_split}_file", f"data/{eval_split}.jsonl")
    if not split_file.exists():
        split_file = REPO_ROOT / "data" / "sample_problems.jsonl"
    problems = [p for p in read_jsonl(split_file) if not p["pid"].startswith("sample-bad")]
    max_eval = int(data_cfg.get("max_eval_problems", 300))
    problems = problems[:max_eval]
    if eval_split == "test":
        from benchmark.hidden_test_vault import inject_hidden_tests_in_memory

        problems = inject_hidden_tests_in_memory(problems)

    arm = rdir.name.split("_seed")[0]
    if (rdir / "manifest.json").exists():
        try:
            arm = str(json.loads((rdir / "manifest.json").read_text(encoding="utf-8")).get("arm", arm))
        except Exception:
            pass
    if "turns2" in rdir.name and not arm.endswith("_turns2"):
        arm = f"{arm}_turns2"

    n_eval = int(eval_cfg.get("n_eval", 64))
    k_list = [int(k) for k in eval_cfg.get("k_list", [1, 2, 4, 8, 16, 32, 64]) if int(k) <= n_eval]
    n_subsets = int(eval_cfg.get("n_subsets", 200))
    subset_seed = int(eval_cfg.get("subset_seed", 42))
    turns = 2 if "turns2" in rdir.name else int(rollout_cfg.get("turns", 1))

    cache_root = REPO_ROOT / checker_cfg.get("cache_dir", "cache")
    cache = ExecutionCache(cache_root=cache_root, checker_version=str(checker_cfg.get("version", "v1")))
    use_mock = bool(mock or cfg.get("mode") == "smoke" or os.environ.get("SSPO_MOCK_LLM", "0") == "1")

    def arm_mock_rollout(prob: Dict[str, Any], n: int, t_cnt: int, s_val: int) -> List[List[Dict[str, Any]]]:
        rng = random.Random(s_val + abs(hash(str(prob.get("pid", "")))) % 10007)
        ref_sol = prob.get("reference_solution") or "print(0)\n"
        wrong_sols = prob.get("wrong_solutions") or ["print('WRONG')\n"]
        tle_sol = prob.get("tle_solution") or "import sys\nsys.exit(1)\n"
        p_c = {"sspo": 0.42, "sspo_turns2": 0.44, "pkpo_pass8": 0.34, "drgrpo_pass1": 0.32, "frozen": 0.25, "checker_reward": 0.18, "agent": 0.36}.get(
            arm, 0.30
        )
        p_w1 = {"sspo": 0.20, "sspo_turns2": 0.19, "pkpo_pass8": 0.42, "drgrpo_pass1": 0.40, "frozen": 0.45, "checker_reward": 0.65, "agent": 0.30}.get(
            arm, 0.40
        )
        out = []
        for i in range(n):
            u = (i + rng.random()) / float(max(1, n))
            if u < p_c:
                c = ref_sol
            elif u < p_c + p_w1:
                c = wrong_sols[0]
            else:
                c = tle_sol
            out.append([{"text": f"```python\n{c}\n```", "code": c}])
        return out

    ckpt_lora_dir = rdir / "checkpoints" / ckpt_name / "lora_adapter"
    if not ckpt_lora_dir.exists() and (rdir / "checkpoints" / "active_lora").exists():
        ckpt_lora_dir = rdir / "checkpoints" / "active_lora"
    eval_lora_path = str(ckpt_lora_dir) if (ckpt_lora_dir.exists() and arm != "frozen") else None

    os.environ.setdefault("SSPO_VLLM_ONLY", "1")
    engine = RolloutEngine(cfg, mock=use_mock, mock_fn=arm_mock_rollout if use_mock else None, enable_lora=bool(eval_lora_path))
    partial_file = eval_out_dir / f".{eval_split}_{ckpt_name}.partial.json"
    done_by_pid: Dict[str, Dict[str, Any]] = {}
    if resume and partial_file.exists():
        try:
            done_by_pid = json.loads(partial_file.read_text(encoding="utf-8"))
        except Exception:
            done_by_pid = {}

    chunk_size = 16
    for c_start in range(0, len(problems), chunk_size):
        chunk_probs = problems[c_start : c_start + chunk_size]
        todo_probs = [p for p in chunk_probs if p["pid"] not in done_by_pid]
        if todo_probs:
            todo_seeds = [subset_seed + c_start + i for i, p in enumerate(chunk_probs) if p["pid"] not in done_by_pid]
            traj_map = engine.sample_batch_trajectories(
                todo_probs, n=n_eval, turns=turns, seeds=todo_seeds, lora_path=eval_lora_path, lora_id=999
            )
            eval_items: List[Tuple[str, Optional[str], dict]] = []
            for p in todo_probs:
                for t_turns in traj_map[p["pid"]]:
                    rep = extract_representative_program(t_turns, prob=p)
                    eval_items.append((p["pid"], rep["code"], p))
            all_evs = cache.evaluate_batch_programs(eval_items)
            ev_cursor = 0
            for p in todo_probs:
                pid = p["pid"]
                contest_id = str(p.get("contest_id", f"c_{pid}"))
                trajs = traj_map[pid]
                g_list, chk_list, sig_list = [], [], []
                for _ in trajs:
                    ev = all_evs[ev_cursor]
                    ev_cursor += 1
                    g_list.append(ev["g"])
                    chk_list.append(ev["chk"])
                    sig_list.append(ev["sig"])
                done_by_pid[pid] = evaluate_problem_rollouts(
                    pid=pid,
                    contest_id=contest_id,
                    g_list=g_list,
                    chk_list=chk_list,
                    sig_list=sig_list,
                    k_list=k_list,
                    n_subsets=n_subsets,
                    subset_seed=subset_seed,
                )
            atomic_write_json(partial_file, done_by_pid)

    eval_rows = [done_by_pid[p["pid"]] for p in problems if p["pid"] in done_by_pid]
    atomic_write_jsonl(eval_jsonl_path, eval_rows)
    summary = summarize_eval_rows(eval_rows, run_name=rdir.name)
    summary["arm"] = arm
    summary["ckpt"] = ckpt_name
    summary["eval_split"] = eval_split
    summary["eval_file"] = to_rel_path(eval_jsonl_path)
    atomic_write_json(rdir / "summary.json", summary)
    write_done_marker(eval_done_marker, {"ckpt": ckpt_name, "split": eval_split, "n_problems": len(eval_rows)})

    generate_tables_and_report(rdir.parent)
    return summary


def main():
    parser = argparse.ArgumentParser(description="SSPO Training and Evaluation CLI.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Run or resume a training arm")
    p_train.add_argument("--config", type=str, default="configs/pilot.yaml")
    p_train.add_argument(
        "--arm",
        type=str,
        default="sspo",
        choices=["agent", "sspo", "drgrpo_pass1", "pkpo_pass8", "checker_reward", "frozen"],
    )
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--run-dir", type=str, default=None)
    p_train.add_argument("--turns", type=int, choices=[1, 2], default=None)
    p_train.add_argument("--max-steps", type=int, default=None)
    p_train.add_argument("--stop-after-step", type=int, default=None)
    p_train.add_argument("--resume", action="store_true", default=True)
    p_train.add_argument("--no-resume", dest="resume", action="store_false")
    p_train.add_argument("--mock", action="store_true")

    p_eval = sub.add_parser("eval", help="Evaluate a run directory and update summary/tables/report")
    p_eval.add_argument("--run-dir", type=str, required=True)
    p_eval.add_argument("--config", type=str, default="configs/pilot.yaml")
    p_eval.add_argument("--ckpt", type=str, default="latest")
    p_eval.add_argument("--split", type=str, default="test")
    p_eval.add_argument("--resume", action="store_true", default=True)
    p_eval.add_argument("--no-resume", dest="resume", action="store_false")
    p_eval.add_argument("--mock", action="store_true")

    args = parser.parse_args()
    if args.cmd == "train":
        res = run_training(
            config_path=args.config,
            arm=args.arm,
            seed=args.seed,
            run_dir=args.run_dir,
            turns=args.turns,
            max_steps=args.max_steps,
            stop_after_step=args.stop_after_step,
            resume=args.resume,
            mock=args.mock,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
    elif args.cmd == "eval":
        res = run_evaluation(
            run_dir=args.run_dir,
            config_path=args.config,
            ckpt=args.ckpt,
            eval_split=args.split,
            resume=args.resume,
            mock=args.mock,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
