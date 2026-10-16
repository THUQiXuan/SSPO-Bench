# SSPO-Bench Official Baseline Leaderboard

All baselines are trained and evaluated under the locked SSPO-Bench protocol:
- **Base Policy**: `Qwen/Qwen2.5-Coder-3B-Instruct` (LoRA `r=16, alpha=32`)
- **Training Budget**: `30` steps (`20` steps for `checker_reward`), `batch_problems=16`, `n_rollouts=16`, `k_deploy=8`, `lr=1e-6`, `kl_coef=0.001`, `clip_eps=0.2`
- **Evaluation Protocol**: `100` held-out test problems, `N_eval=16` rollouts per problem at `temperature=1.0`, `K=8` deployment budget with public-example filter + execution-signature majority-vote selector $\mathcal{S}$.

---

## 1. Main Baseline Leaderboard (`Select@8` & Auxiliary Metrics)

| Method (`arm`) | Turns | Seeds | Best `Select@8` | Final `Select@8` (`step_0030`) | `Pass@1` (Best) | `Pass@8` (Best) | `Chk` Pass Rate | Cluster Impure Rate | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`sspo_turns2`** *(Multi-Turn Reference)* | 2 | 1 | **19.88%** | 15.56% | **13.63%** | **27.03%** | **10.56%** | 2.00% | Two-turn public-checker execution feedback + SSPO |
| **`pkpo_pass8`** *(Coverage RL)* | 1 | 2 | **18.45%** | **18.03%** | 12.28% | 25.85% | 9.44% | 0.50% | Exact `Pass@8` LOO estimator on $g(y)$ |
| **`drgrpo_pass1`** *(Starting Agent Template)* | 1 | 2 | **18.44%** | 16.54% | 11.78% | 24.74% | 9.41% | 1.50% | Standard `Pass@1` Dr. GRPO on $g(y)$ |
| **`sspo`** *(Selection-Aware RL)* | 1 | 3 | **18.21%** | 17.34% | 12.33% | 25.16% | 9.73% | 1.00% | DP leave-one-out selector marginal credit ($\alpha=0.7$) |
| **`checker_reward`** *(Public-Only RL)* | 1 | 1 | 18.12% | 15.68% | 11.94% | 24.35% | 9.56% | 0.00% | Collapses at step 20 (`-1.16 pt` below `frozen`) due to false positives |
| **`frozen`** *(Untrained Base Model)* | 1 | 1 | 16.84% | 16.84% | 11.12% | 23.00% | 8.75% | 0.00% | Zero-shot `Qwen2.5-Coder-3B-Instruct` |

---

## 2. Per-Seed Breakdown

| Run Name | Arm | Seed | Best Checkpoint | Best `Select@8` | `Pass@1` | `Pass@8` | `Gap@8` | `Chk` Rate | Final Checkpoint | Final `Select@8` |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|
| `frozen_seed0` | `frozen` | 0 | `step_0000` | 16.84% | 11.12% | 23.00% | -0.13% | 8.75% | `step_0000` | 16.84% |
| `checker_reward_seed0` | `checker_reward` | 0 | `step_0015` | 18.12% | 11.94% | 24.35% | -0.10% | 9.56% | `step_0020` | 15.68% |
| `drgrpo_pass1_seed0` | `drgrpo_pass1` | 0 | `step_0015` | 17.91% | 11.50% | 24.74% | +0.22% | 9.31% | `step_0030` | 15.69% |
| `drgrpo_pass1_seed1` | `drgrpo_pass1` | 1 | `step_0015` | 18.96% | 12.06% | 24.73% | -0.08% | 9.50% | `step_0030` | 17.38% |
| `pkpo_pass8_seed0` | `pkpo_pass8` | 0 | `step_0030` | 18.15% | 12.44% | 26.17% | +0.01% | 9.25% | `step_0030` | 18.15% |
| `pkpo_pass8_seed1` | `pkpo_pass8` | 1 | `step_0020` | 18.75% | 12.12% | 25.53% | +0.09% | 9.62% | `step_0030` | 17.90% |
| `sspo_seed0` | `sspo` | 0 | `step_0005` | 18.33% | 12.00% | 25.53% | +0.42% | 9.62% | `step_0030` | 16.84% |
| `sspo_seed1` | `sspo` | 1 | `step_0030` | 18.41% | 12.44% | 25.29% | +0.03% | 9.75% | `step_0030` | 18.41% |
| `sspo_seed2` | `sspo` | 2 | `step_0010` | 17.88% | 12.56% | 24.65% | +0.45% | 9.81% | `step_0030` | 16.77% |
| `sspo_turns2_seed0` | `sspo_turns2` | 0 | `step_0005` | **19.88%** | **13.63%** | **27.03%** | +0.18% | **10.56%** | `step_0030` | 15.56% |

Full per-checkpoint progression across all 47 evaluated checkpoints is available in [`results/tables/checkpoint_progression.csv`](../results/tables/checkpoint_progression.csv).
