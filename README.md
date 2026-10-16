# SSPO-Bench: An Agentic Benchmark for RL Algorithm Discovery under Test-Time Code Selection

**SSPO-Bench** is a self-contained benchmark environment designed to evaluate **AI Research & Coding Agents** on their ability to discover and implement novel **Reinforcement Learning (RL) algorithms** for code generation under **test-time selection (`Select@K`)**.

Unlike hyperparameter sweeps or unconstrained repo edits, SSPO-Bench fixes the base model, dataset splits, rollout budget, optimizer hyperparameters, and evaluation protocol—isolating the **RL credit assignment and policy loss objective** inside a single, verified sandbox module ([`benchmark/agent_algorithm.py`](benchmark/agent_algorithm.py)).

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

## 2. Official Baseline Leaderboard (`Select@8`, 100 Held-Out Test Problems)

All baseline algorithms were trained and evaluated under the locked SSPO-Bench protocol (`Qwen/Qwen2.5-Coder-3B-Instruct`, LoRA $r=16, \alpha=32$, `30` training steps, `batch_problems=16`, $N_{\text{train}}=16, K=8$, evaluated with $N_{\text{eval}}=16$ rollouts per problem across 100 test problems).

| Method (`arm`) | Turns | Seeds | Best `Select@8` | Final `Select@8` (`step_0030`) | `Pass@1` (Best) | `Pass@8` (Best) | `Chk` Pass Rate | Cluster Impure Rate | Summary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`sspo_turns2`** *(Multi-Turn Reference)* | 2 | 1 | **19.88%** | 15.56% | **13.63%** | **27.03%** | **10.56%** | 2.00% | 2-turn public checker execution feedback + SSPO |
| **`pkpo_pass8`** *(Coverage RL)* | 1 | 2 | **18.45%** | **18.03%** | 12.28% | 25.85% | 9.44% | 0.50% | Exact `Pass@8` LOO estimator on $g(y)$ |
| **`drgrpo_pass1`** *(Default Agent Template)* | 1 | 2 | **18.44%** | 16.54% | 11.78% | 24.74% | 9.41% | 1.50% | Standard `Pass@1` Dr. GRPO on $g(y)$ |
| **`sspo`** *(Selection-Aware RL)* | 1 | 3 | **18.21%** | 17.34% | 12.33% | 25.16% | 9.73% | 1.00% | DP leave-one-out selector marginal credit ($\alpha=0.7$) |
| **`checker_reward`** *(Public-Only RL)* | 1 | 1 | 18.12% | 15.68% | 11.94% | 24.35% | 9.56% | 0.00% | Collapses at step 20 (`-1.16 pt` below `frozen`) due to false positives |
| **`frozen`** *(Untrained Base Model)* | 1 | 1 | 16.84% | 16.84% | 11.12% | 23.00% | 8.75% | 0.00% | Zero-shot `Qwen2.5-Coder-3B-Instruct` |

- Detailed per-seed metrics: [`leaderboard/LEADERBOARD.md`](leaderboard/LEADERBOARD.md) & [`leaderboard/baselines.json`](leaderboard/baselines.json)
- Full 47-checkpoint progression table: [`results/tables/checkpoint_progression.csv`](results/tables/checkpoint_progression.csv)
- Paired cluster-bootstrap 95% CIs: [`results/tables/paired_bootstrap_ci.csv`](results/tables/paired_bootstrap_ci.csv)
- Training step curves (260 steps): [`results/tables/training_curves.csv`](results/tables/training_curves.csv)

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
   All 33 framework, configuration, and dataset files (`sspo/*.py`, `configs/benchmark_locked.yaml`, `data/*`) are verified via SHA-256 hashes before training and before evaluation. Modifying any file outside `benchmark/agent_algorithm.py` raises `TAMPER_DETECTED`.
2. **Layer 2 — Static AST & Import Sandbox**:
   `benchmark/agent_algorithm.py` is parsed via Python's `ast` module. Filesystem, network, subprocess, and reflection primitives (`os`, `sys`, `open`, `eval`, `exec`, `subprocess`, `getattr`, `__globals__`, etc.) are strictly blocked.
3. **Layer 3 — Mathematical & Functional Contract**:
   Before any GPU training is launched, `verifier.py` tests `agent_algorithm.py` on 5 synthetic rollout pools to enforce:
   - **Permutation Equivariance**: Permuting candidate order by $\pi$ must permute `advantages` by $\pi$ (preventing index-based hardcoding),
   - **Finiteness & Bounds**: No `NaN`/`Inf`, $|A_i| \le 50.0$, $0 \le \text{loss\_scale} \le 2.0$,
   - **Differentiability & Gradient Flow**: `compute_policy_loss` must backpropagate finite, non-zero gradients to `policy_logprobs`,
   - **CPU Latency Budget**: `< 50 ms` per problem call.

### Train / Validation / Encrypted Hidden Test Split Isolation
- **`data/train.jsonl` (500 problems)**: Used during the 30-step RL training loop.
- **`data/dev.jsonl` (100 problems — Validation Split)**: Open for the agent to run validation evaluations (`eval-dev`), analyze failure modes, and perform validation-based checkpoint selection.
- **`data/test.jsonl` + `data/test_hidden.enc` (100 problems — Encrypted Hidden Test Split)**:
  On disk, `data/test.jsonl` has `"hidden": []` and `"reference_solution": null` stripped. The 40 hidden test cases per test problem are compressed and encrypted inside `data/test_hidden.enc` (HMAC-SHA256 authenticated stream cipher) and decrypted strictly **in memory** during `eval-test`. Furthermore, `benchmark/agent_algorithm.py` is **never invoked** during evaluation.

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
Feed [`benchmark/AGENT_PROMPT.md`](benchmark/AGENT_PROMPT.md) directly to your agent (e.g., Claude Code, Codex, OpenHands, SWE-agent, Aider). Reference implementations of `drgrpo_pass1`, `pkpo_pass8`, `checker_reward`, and `sspo` in the exact `agent_algorithm.py` format are provided in [`benchmark/reference_algorithms/`](benchmark/reference_algorithms/).

---

## 5. Repository Structure

```text
├── benchmark/
│   ├── agent_algorithm.py        # [AGENT EDITABLE] The ONLY file agents modify
│   ├── verifier.py               # 3-layer SHA-256 lock, AST sandbox & math contract verifier
│   ├── evaluate_agent.py         # Unified CLI for verify / smoke / train / eval-dev / eval-test
│   ├── hidden_test_vault.py      # Encrypted test-set vault (in-memory decryption only)
│   ├── manifest_locks.json       # SHA-256 checksums of all 33 locked files
│   ├── AGENT_PROMPT.md           # Ready-to-use prompt for evaluating AI agents
│   └── reference_algorithms/     # Reference implementations (drgrpo_pass1, pkpo_pass8, sspo, checker_reward)
├── leaderboard/
│   ├── LEADERBOARD.md            # Official baseline leaderboard & per-seed breakdown
│   └── baselines.json            # Machine-readable baseline results
├── configs/
│   ├── benchmark_locked.yaml     # Locked hyperparameters for official agent evaluation
│   ├── pilot.yaml                # Pilot experiment configuration
│   └── smoke.yaml                # Fast CPU smoke test configuration
├── data/
│   ├── train.jsonl               # 500 unique-output training problems
│   ├── dev.jsonl                 # 100 held-out validation problems
│   ├── test.jsonl                # 100 test prompts & public examples (hidden tests stripped on disk)
│   └── test_hidden.enc           # Encrypted hidden test cases for the 100 test problems
├── results/                      # Full experimental artifacts, metrics.jsonl, eval/*.jsonl & tables/*.csv
├── sspo/                         # Core rollout, checker, DP, selector, trainer & evaluation library
└── tests/                        # Unit tests (DP, advantages, pipeline, resume & benchmark guardrails)
```
