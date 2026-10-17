# SSPO-Bench: Task Prompt for AI Research & Coding Agents

> Copy and paste this prompt directly to the AI agent being evaluated on **SSPO-Bench**.

---

## Your Task: Discover a Better RL Algorithm for Test-Time Code Selection (`Select@8`)

You are operating inside **SSPO-Bench**, an RL algorithm discovery benchmark for competitive programming under test-time selection.

### 1. Problem Setup & Fixed Environment
- **Base Policy**: `Qwen/Qwen3-4B` with LoRA (`r = 64, alpha = 128`).
- **Context Window (Locked)**: `max_context_length = 40960` tokens, `max_tokens = 8192`, `max_train_seq_len = 8192`.
- **Training Budget (Locked)**: `30` steps, `batch_problems = 16`, `n_rollouts = 16` trajectories per problem, `lr = 1e-6`, `kl_coef = 0.001`, `clip_eps = 0.2`.
- **Deployment Target (`Select@8`)**:
  At test time, for each problem $x$, $K = 8$ candidate programs $Y_K = (y_1, \dots, y_K)$ are sampled from the policy and passed through a deterministic selector $\mathcal{S}(Y_K)$:
  1. **Public Example Filter ($\text{Chk}(y) \in \{0, 1\}$)**: Each candidate is executed on the problem's public example inputs. If at least one candidate passes ($\text{Chk}(y) = 1$), failing candidates ($\text{Chk}(y) = 0$) are discarded.
  2. **Execution Signature Majority Vote ($\sigma(y)$)**: Surviving candidates are grouped by their canonical output signature $\sigma(y)$ on public/probe inputs. The selector chooses a candidate from the largest output cluster (breaking ties uniformly at random).
- **Primary Evaluation Metric**: **`Select@8`** — the expected hidden-test accuracy of the single program selected by $\mathcal{S}$ from a random subset of $K = 8$ rollouts (estimated over $N_{\text{eval}} = 16$ rollouts on 100 problems).

### 2. Strict Sandbox Rules (Anti-Hacking Guardrails)
- **You may ONLY edit one file: [`benchmark/agent_algorithm.py`](agent_algorithm.py).**
- All other files (`sspo/*`, `configs/*`, `data/*`, `benchmark/verifier.py`, `benchmark/evaluate_agent.py`) are cryptographically locked via SHA-256 checksums in `benchmark/manifest_locks.json`. Modifying any locked file will immediately fail verification (`TAMPER_DETECTED`).
- Inside `benchmark/agent_algorithm.py`, you must implement two functions:
  1. `compute_group_advantages(ctx: GroupContext) -> GroupAdvantageOutput`:
     Computes per-candidate advantages `advantages: np.ndarray` of shape `(n,)` and `loss_scale: float` (`in [0.0, 2.0]`) from training rollout outcomes:
     - `ctx.g_list` (`Tuple[int, ...]`): Hidden-test pass/fail in `{0, 1}` (available during training only)
     - `ctx.chk_list` (`Tuple[int, ...]`): Public example checker pass/fail in `{0, 1}`
     - `ctx.sig_list` (`Tuple[str, ...]`): Canonical execution output signature
     - `ctx.response_lengths`, `ctx.sequence_logprobs`, `ctx.ref_sequence_logprobs`, `ctx.step`, `ctx.total_steps`, `ctx.history_state`
  2. `compute_policy_loss(batch: PolicyLossBatch) -> Tuple[torch.Tensor, Dict[str, float]]`:
     Computes the differentiable token-level policy gradient + regularization loss from `policy_logprobs`, `old_logprobs`, `ref_logprobs`, `action_mask`, `advantages`, and `loss_scales`.

### 3. Workflow & Commands
1. **Verify Your Algorithm (< 3s on CPU)**:
   ```bash
   python -m benchmark.evaluate_agent verify
   ```
   Checks SHA-256 repo locks, AST import sandbox, permutation equivariance, finiteness, differentiability, and CPU latency (< 50 ms/problem).
2. **End-to-End Smoke Test (< 5s on CPU)**:
   ```bash
   python -m benchmark.evaluate_agent smoke
   ```
   Runs a 4-step mock training + validation/test evaluation loop to verify your algorithm integrates cleanly with the trainer.
3. **Full 30-Step Training & Validation Evaluation (`data/dev.jsonl`, 100 problems)**:
   ```bash
   python -m benchmark.evaluate_agent train --seed 0 --run-dir results/agent_seed0
   python -m benchmark.evaluate_agent eval-dev --run-dir results/agent_seed0
   ```
   Use the validation set (`dev`) to iterate on your RL algorithm and inspect `results/agent_seed0/dev_progression.json`.
4. **Final Held-Out Test Evaluation (`data/test.jsonl` + `data/test_hidden.enc`, 100 problems)**:
   ```bash
   python -m benchmark.evaluate_agent eval-test --run-dir results/agent_seed0
   ```
   Produces `results/agent_seed0/benchmark_scorecard.json` comparing your algorithm against all published baselines (`frozen`: `38.64%`, `checker_reward`: `37.66%` final / `40.05%` best, `drgrpo_pass1`: `38.79%` final / `39.32%` best, `pkpo_pass8`: `38.24%` final / `39.91%` best, `sspo`: `39.44%` final / `40.11%` best, `sspo_turns2`: `43.22%` final & best).
