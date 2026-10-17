# SSPO-Bench: An Agentic Benchmark for RL Algorithm Discovery under Test-Time Code Selection

**SSPO-Bench** is a self-contained benchmark environment designed to evaluate **AI Research & Coding Agents** on their ability to discover and implement novel **Reinforcement Learning (RL) algorithms** for code generation under **test-time selection (`Select@K`)**.

Unlike hyperparameter sweeps or unconstrained repo edits, SSPO-Bench fixes the base model (`Qwen/Qwen3-4B` with full `40,960`-token context window), calibrated dataset splits across **two strictly disjoint benchmarks (`Benchmark 1` and `Benchmark 2`)**, rollout budget, optimizer hyperparameters, and evaluation protocol—isolating the **RL credit assignment and policy loss objective** inside a single, verified sandbox module ([`benchmark/agent_algorithm.py`](benchmark/agent_algorithm.py)).

---

## 1. Benchmark Motivation & Task Formulation

When code generation models are deployed at test time, practitioners rarely take a single greedy sample (`Pass@1`). Instead, $K$ candidate programs $Y_K = (y_1, \dots, y_K)$ are sampled and filtered by a **test-time selector $\mathcal{S}(Y_K)$**:
1. **Public Example Filter ($\text{Chk}(y) \in \{0, 1\}$)**: Candidates are executed on the problem's public example inputs. If any candidate passes ($\text{Chk}(y) = 1$), failing candidates ($\text{Chk}(y) = 0$) are discarded.
2. **Execution Signature Majority Vote ($\sigma(y)$)**: Surviving candidates are grouped by their canonical output signature $\sigma(y)$ on public/probe inputs. The selector chooses a candidate from the largest output cluster (breaking ties uniformly at random).

### Why Standard RL Can Fail Under Test-Time Selection
During RL training, $n = 16$ rollouts are sampled per problem $x$, yielding three signals per rollout $i$:
- $g_i \in \{0, 1\}$: Hidden-test correctness (training-time reward oracle, **never** available at test time),
- $\text{Chk}_i \in \{0, 1\}$: Public-example checker outcome,
- $\sigma_i \in \Sigma^*$: Canonical execution output signature (or `"TLE"` / `"RE"`).

Candidates partition into:
- **$C = \{i : g_i = 1, \text{Chk}_i = 1\}$**: Correct candidates that survive the public filter,
- **$T = \{i : \text{Chk}_i = 0\}$**: Candidates rejected by the public filter,
- **$W_m = \{i : g_i = 0, \text{Chk}_i = 1, \sigma_i = s_m\}$**: **False-positive wrong clusters** that pass the public examples ($\text{Chk}_i = 1$) and collude on the same wrong output $s_m$. Whenever $|W_m| \ge |C|$, majority voting selects a wrong program even when a correct solution was generated!

---

## 2. Official Dual-Benchmark Leaderboards (`Select@8` @ `40,960` Context)

All **7 baseline and reference algorithms** (`40` runs total, `443` checkpoint evaluations) were trained and evaluated under the locked SSPO-Bench protocol (`Qwen/Qwen3-4B`, `max_context_length=40960`, `max_tokens=8192`, `max_train_seq_len=8192`, LoRA $r=64, \alpha=128$, `30` training steps, `batch_problems=16`, $N_{\text{train}}=16, K=8$, evaluated with $N_{\text{eval}}=16$ rollouts per problem).

### 2.1 Benchmark 1 (`data/`, `results/` — 100 Test / 100 Dev / 91 Train Problems, 21 Runs)

| Method (`arm`) | Turns | Seeds | Best `Select@8` | Final `Select@8` (`step_0030`) | Val-Selected Test `Select@8` | `Pass@1` (Best) | `Pass@8` (Best) | `Chk` Pass Rate | Summary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`sspo_turns2`** *(Multi-Turn Reference)* | 2 | **5** (`seed0..4`) | **43.34% ± 1.13%** | **41.49% ± 1.25%** | **41.59% ± 0.92%** | **21.89% ± 0.39%** | **48.61% ± 0.90%** | **24.09%** | 2-turn public checker execution feedback + SSPO (`+4.39 pt` over `frozen`, `WIN`) |
| **`tang_maj_loo`** *(Meta `2503.19595`)* | 1 | **3** (`seed0..2`) | **41.02% ± 0.40%** | **39.71% ± 1.34%** | **39.33% ± 1.18%** | 17.00% ± 0.38% | 43.62% ± 1.06% | 19.06% | Mean-centered LOO majority-vote estimator on full pool $Y$ (`+2.07 pt` over `frozen`) |
| **`sspo`** *(Selection-Aware RL)* | 1 | **5** (`seed0..4`, 7 total) | **39.80% ± 0.77%** *(39.93% 7-seed)* | **39.23% ± 0.50%** | **38.92% ± 0.63%** | **17.44% ± 0.10%** | **44.12% ± 0.66%** | **19.53%** | Exact $\binom{n-1}{K-1}$ DP LOO selector marginal credit ($\alpha=0.7$) |
| **`checker_reward`** *(Public-Only RL)* | 1 | 1 (`seed0`) | 40.05% | 37.66% (`step_0020`) | 38.11% | 17.31% | 44.35% | 19.50% | Collapses at step 20 (`-1.29 pt` below `frozen`) due to false positives |
| **`pkpo_pass8`** *(DeepMind `2505.15201`)* | 1 | 2 (`seed0..1`) | 39.91% ± 1.10% | 38.24% ± 0.97% | 38.62% ± 0.54% | 16.75% ± 0.09% | 43.91% ± 2.23% | 18.44% | Exact `Pass@8` LOO estimator $s_i^{\text{loo}-1}$ on $g(y)$ |
| **`drgrpo_pass1`** *(Default Agent Template)* | 1 | 2 (`seed0..1`) | 39.32% ± 0.42% | 38.79% ± 1.18% | 38.87% ± 1.06% | 17.25% ± 0.44% | 44.11% ± 0.08% | 19.03% | Standard `Pass@1` Dr. GRPO on $g(y)$ |
| **`frozen`** *(Untrained Base Model)* | 1 | 1 (`seed0`) | 38.95% | 38.95% | 38.95% | 16.50% | 43.62% | 18.44% | Paired seed-matched `Qwen/Qwen3-4B` (`40,960` context) |

### 2.2 Benchmark 2 (`data_b2/`, `results_b2/` — 64 Test / 64 Dev / 64 Train Problems, 19 Runs, `0` Overlap with Benchmark 1)

| Method (`arm`) | Turns | Seeds | Best `Select@8` | Epoch-5 `Select@8` (`step_0020`) | Final `Select@8` (`step_0030`) | Val-Selected Test `Select@8` | `Pass@1` (Best) | `Pass@8` (Best) | Summary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`sspo_turns2`** *(Multi-Turn Reference)* | 2 | **5** (`seed0..4`) | **46.95% ± 1.19%** | **44.19% ± 0.90%** | **43.90% ± 1.57%** | **44.55% ± 1.71%** | **23.44% ± 0.64%** | **53.18% ± 2.13%** | **`+13.00 pt` (Peak) / `+10.60 pt` (Val-Sel) over `frozen`; 6/6 Holm-significant `WIN`s (`p_holm <= 0.036`)** |
| **`checker_reward`** *(Public-Only RL)* | 1 | 1 (`seed0`) | 40.87% (`step_0020`) | 40.87% | 40.87% (`step_0020`) | 37.09% | 21.09% | 48.39% | Val-selected (`step_0015`) drops to `37.09%` |
| **`drgrpo_pass1`** *(Default Agent Template)* | 1 | 2 (`seed0..1`) | 40.52% ± 1.00% | 38.71% ± 1.88% | 38.16% ± 0.71% | 38.02% ± 0.91% | 19.97% ± 0.76% | 47.19% ± 0.48% | Standard `Pass@1` Dr. GRPO on $g(y)$ |
| **`tang_maj_loo`** *(Meta `2503.19595`)* | 1 | **3** (`seed0..2`) | 40.34% ± 1.29% | 37.64% ± 1.11% | 38.33% ± 0.72% | 37.92% ± 1.67% | 20.18% ± 0.75% | 46.52% ± 1.42% | Best 1-turn final checkpoint mean (`38.33%`) |
| **`pkpo_pass8`** *(DeepMind `2505.15201`)* | 1 | 2 (`seed0..1`) | 40.15% ± 1.86% | 35.79% ± 0.28% | 37.86% ± 0.93% | 37.05% ± 2.07% | 20.61% ± 0.56% | 46.11% ± 2.52% | Exact `Pass@8` LOO estimator on $g(y)$ |
| **`sspo`** *(Selection-Aware RL)* | 1 | **5** (`seed0..4`) | **39.18% ± 0.40%** | **38.41% ± 1.94%** | 37.17% ± 0.89% | 37.31% ± 1.38% | 20.06% ± 0.69% | 45.24% ± 1.39% | **`+5.22 pt` over `frozen` (`p_holm = 0.040`, `WIN`)**; lowest cross-seed SD (`±0.40%`) |
| **`frozen`** *(Untrained Base Model)* | 1 | 1 (`seed0`) | 33.95% | 33.95% | 33.95% | 33.95% | 18.36% | 40.44% | Paired seed-matched `Qwen/Qwen3-4B` (`40,960` context) |

- Detailed per-seed metrics across both benchmarks: [`leaderboard/LEADERBOARD.md`](leaderboard/LEADERBOARD.md) & [`leaderboard/baselines.json`](leaderboard/baselines.json)
- Self-contained technical & statistical analysis report: [`results/ANALYSIS_REPORT.md`](results/ANALYSIS_REPORT.md)
- Benchmark 1 tables & report: [`results/report.md`](results/report.md), [`results/tables/checkpoint_progression.csv`](results/tables/checkpoint_progression.csv), [`results/tables/paired_bootstrap_ci.csv`](results/tables/paired_bootstrap_ci.csv)
- Benchmark 2 tables & report: [`results_b2/report.md`](results_b2/report.md), [`results_b2/tables/checkpoint_progression.csv`](results_b2/tables/checkpoint_progression.csv), [`results_b2/tables/paired_bootstrap_ci.csv`](results_b2/tables/paired_bootstrap_ci.csv)
- Archived `4,096`-context results (for analysis comparison only): [`results_4k_archive/`](results_4k_archive/)

---

## 3. Agent Sandbox, Degrees of Freedom & Anti-Hacking Guardrails

### What the Agent Can Edit
The agent is permitted to edit **only one file**:
- **[`benchmark/agent_algorithm.py`](benchmark/agent_algorithm.py)** (starts initialized with vanilla `drgrpo_pass1`).

Inside `benchmark/agent_algorithm.py`, the agent implements two composable hooks:
1. **`compute_group_advantages(ctx: GroupContext) -> GroupAdvantageOutput`**:
   - Receives `g_list`, `chk_list`, `sig_list`, `n=16`, `k=8`, `step`, `total_steps`, `response_lengths`, `sequence_logprobs`, `ref_sequence_logprobs`, and a persistent per-problem state dictionary `history_state`.
   - Returns per-candidate `advantages: np.ndarray` of shape `(n,)`, `loss_scale: float` (`in [0.0, 2.0]`), and `drop_problem: bool`.
2. **`compute_policy_loss(batch: PolicyLossBatch) -> Tuple[torch.Tensor, Dict[str, float]]`**:
   - Receives token-level `policy_logprobs`, `old_logprobs`, `ref_logprobs`, `action_mask`, `advantages`, and `loss_scales`.
   - Returns a scalar differentiable `torch.Tensor` loss and diagnostic stats.

### Three-Layer Anti-Hacking & Verification System ([`benchmark/verifier.py`](benchmark/verifier.py))
1. **Layer 1 — Cryptographic File Lock (`benchmark/manifest_locks.json`)**:
   All locked framework, configuration, and dataset files (`sspo/*.py`, `configs/benchmark_locked.yaml`, `data/*`, `data_b2/*`) are verified via SHA-256 hashes before training and before evaluation. Modifying any file outside `benchmark/agent_algorithm.py` raises `TAMPER_DETECTED`.
2. **Layer 2 — Static AST & Import Sandbox**:
   `benchmark/agent_algorithm.py` is parsed via Python's `ast` module. Filesystem, network, subprocess, and reflection primitives (`os`, `sys`, `open`, `eval`, `exec`, `subprocess`, `getattr`, `__globals__`, etc.) are strictly blocked.
3. **Layer 3 — Mathematical & Functional Contract**:
   Before any GPU training is launched, `verifier.py` tests `agent_algorithm.py` on 5 synthetic rollout pools to enforce:
   - **Permutation Equivariance**: Permuting candidate order by $\pi$ must permute `advantages` by $\pi$ (preventing index-based hardcoding),
   - **Finiteness & Bounds**: No `NaN`/`Inf`, $|A_i| \le 50.0$, $0 \le \text{loss\_scale} \le 2.0$,
   - **Differentiability & Gradient Flow**: `compute_policy_loss` must backpropagate finite, non-zero gradients to `policy_logprobs`,
   - **CPU Latency Budget**: `< 50 ms` per problem call.

### Train / Validation / Encrypted Hidden Test Split Isolation
All splits are strictly **`contest_id`-disjoint** (`0` contest overlap across `train`, `dev`, `test`) and calibrated under `Qwen/Qwen3-4B` (`40,960` context):
- **Benchmark 1 (`data/`)**: `train.jsonl` (`91` active problems), `dev.jsonl` (`100` validation problems), `test.jsonl` + `test_hidden.enc` (`100` encrypted test problems).
- **Benchmark 2 (`data_b2/`)**: `train.jsonl` (`64` active problems), `dev.jsonl` (`64` validation problems), `test.jsonl` + `test_hidden.enc.part00..05` (`64` encrypted test problems, `0` problem overlap with Benchmark 1).
- On disk, `test.jsonl` has `"hidden": []` and `"reference_solution": null` stripped. Hidden test cases are decrypted strictly **in memory** during `eval-test` via `benchmark/hidden_test_vault.py` (which transparently supports both single-file `.enc` and multi-part `.enc.part*` vaults).

---

## 4. Quickstart & Commands

### Installation
```bash
pip install -r requirements-dev.txt
# For GPU training & vLLM rollout generation:
pip install -r requirements-gpu.txt
```

### 1. Pre-Flight Verification (< 3 seconds on CPU)
```bash
python -m benchmark.evaluate_agent verify
```

### 2. End-to-End Smoke Test (< 5 seconds on CPU)
```bash
python -m benchmark.evaluate_agent smoke
```

### 3. Run Locked 30-Step RL Training & Validation Evaluation
```bash
# Train the agent's RL algorithm under locked hyperparameters
python -m benchmark.evaluate_agent train --seed 0 --run-dir results/agent_seed0

# Evaluate all saved checkpoints on the 100-problem Validation Set (dev)
python -m benchmark.evaluate_agent eval-dev --run-dir results/agent_seed0
```

### 4. Final Held-Out Test Evaluation & Scorecard Generation
```bash
python -m benchmark.evaluate_agent eval-test --run-dir results/agent_seed0
```
Outputs `results/agent_seed0/benchmark_scorecard.json` comparing the agent's algorithm (`val_selected_test`, `chk_selected_test`, `final_checkpoint_test`, and `oracle_best_test`) against all published baselines.

### Giving SSPO-Bench to an AI Coding Agent
Feed [`benchmark/AGENT_PROMPT.md`](benchmark/AGENT_PROMPT.md) directly to your agent (e.g., Claude Code, Codex, OpenHands, SWE-agent, Aider). Reference implementations of `drgrpo_pass1`, `pkpo_pass8`, `tang_maj_loo`, `checker_reward`, and `sspo` are provided in [`benchmark/reference_algorithms/`](benchmark/reference_algorithms/) and [`sspo/advantages.py`](sspo/advantages.py).

---

## 5. Repository Structure

```text
├── benchmark/
│   ├── agent_algorithm.py        # [AGENT EDITABLE] The ONLY file agents modify
│   ├── verifier.py               # 3-layer SHA-256 lock, AST sandbox & math contract verifier
│   ├── evaluate_agent.py         # Unified CLI for verify / smoke / train / eval-dev / eval-test
│   ├── hidden_test_vault.py      # Encrypted test-set vault (single-file & multi-part shard support)
│   ├── manifest_locks.json       # SHA-256 checksums of all locked framework & dataset files
│   ├── AGENT_PROMPT.md           # Ready-to-use prompt for evaluating AI agents
│   └── reference_algorithms/     # Reference implementations (drgrpo_pass1, pkpo_pass8, sspo, checker_reward)
├── leaderboard/
│   ├── LEADERBOARD.md            # Official dual-benchmark leaderboard & per-seed breakdown
│   └── baselines.json            # Machine-readable baseline results (Benchmark 1 & Benchmark 2)
├── configs/
│   ├── benchmark_locked.yaml     # Locked hyperparameters for official agent evaluation (40,960 context)
│   ├── pilot.yaml                # Benchmark 1 experiment configuration (40,960 context)
│   ├── pilot_b2.yaml             # Benchmark 2 experiment configuration (40,960 context)
│   └── smoke.yaml                # Fast CPU smoke test configuration
├── data/                         # Benchmark 1 splits (91 train / 100 dev / 100 encrypted test)
├── data_b2/                      # Benchmark 2 splits (64 train / 64 dev / 64 encrypted test, 0 overlap)
├── results/                      # Benchmark 1 (21 runs, 230 evals) artifacts, eval/*.jsonl & tables/*.csv
├── results_b2/                   # Benchmark 2 (19 runs, 213 evals) artifacts, eval/*.jsonl & tables/*.csv
├── results_4k_archive/           # Archived 4,096-context experimental results for analysis reference
├── sspo/                         # Core rollout, checker, DP, selector, trainer & evaluation library
└── tests/                        # Unit tests (DP, advantages, pipeline, resume & benchmark guardrails)
```
