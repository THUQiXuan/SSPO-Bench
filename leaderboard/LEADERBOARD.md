# SSPO-Bench Official Baseline Leaderboard (`Qwen/Qwen3-4B` @ `40,960` Context)

All baselines are trained and evaluated under the locked SSPO-Bench protocol:
- **Base Policy**: `Qwen/Qwen3-4B` (LoRA `r=64, alpha=128`)
- **Context Window**: `max_context_length=40960`, `max_tokens=8192`, `max_train_seq_len=8192`
- **Training Budget**: `30` steps (`20` steps for `checker_reward`), `batch_problems=16`, `n_rollouts=16`, `k_deploy=8`, `lr=1e-6`, `kl_coef=0.001`, `clip_eps=0.2`
- **Dataset Splits (`contest_id`-disjoint)**: `91` active calibrated training problems (`data/train.jsonl` / `data/train_active.jsonl`), `100` validation problems (`data/dev.jsonl`), and `100` encrypted held-out test problems (`data/test.jsonl` + `data/test_hidden.enc`).
- **Evaluation Protocol**: `100` held-out test problems, `N_eval=16` rollouts per problem at `temperature=1.0`, `K=8` deployment budget with public-example filter + execution-signature majority-vote selector $\mathcal{S}$.

---

## 1. Main Baseline Leaderboard (`Select@8` & Auxiliary Metrics)

| Method (`arm`) | Turns | Seeds | Best `Select@8` | Final `Select@8` (`step_0030`) | Val-Selected Test `Select@8` | `Pass@1` (Best) | `Pass@8` (Best) | `Chk` Pass Rate | Cluster Impure Rate | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`sspo_turns2`** *(Multi-Turn Reference)* | 2 | 1 | **43.22%** | **43.22%** | **41.78%** | **22.06%** | **48.04%** | **24.31%** | 10.00% | Two-turn public-checker execution feedback + SSPO (`+4.58 pt` over `frozen`) |
| **`sspo`** *(Selection-Aware RL)* | 1 | 3 | **40.11%** | **39.44%** | **38.92%** | **17.39%** | **44.57%** | **19.40%** | 8.00% | DP leave-one-out selector marginal credit ($\alpha=0.7$); best 1-turn method across all metrics |
| **`checker_reward`** *(Public-Only RL)* | 1 | 1 | 40.05% | 37.66% | 38.11% | 17.31% | 44.35% | 19.50% | 11.00% | Collapses at step 20 (`-0.98 pt` below `frozen`) due to false positives |
| **`pkpo_pass8`** *(Coverage RL)* | 1 | 2 | 39.91% | 38.24% | 38.63% | 16.75% | 43.91% | 18.44% | 7.00% | Exact `Pass@8` LOO estimator on $g(y)$ |
| **`drgrpo_pass1`** *(Starting Agent Template)* | 1 | 2 | 39.32% | 38.79% | 38.87% | 17.25% | 44.11% | 19.03% | 5.50% | Standard `Pass@1` Dr. GRPO on $g(y)$ |
| **`frozen`** *(Untrained Base Model)* | 1 | 1 | 38.64% | 38.64% | 38.64% | 16.19% | 42.92% | 17.44% | 8.00% | Zero-shot `Qwen/Qwen3-4B` (`40,960` context) |

---

## 2. Per-Seed Breakdown

| Run Name | Arm | Seed | Best Checkpoint | Best `Select@8` | `Pass@1` | `Pass@8` | `Gap@8` | `Chk` Rate | Final Checkpoint | Final `Select@8` | Val-Selected Checkpoint (`dev` $\to$ `test`) |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|---|
| `frozen_seed0` | `frozen` | 0 | `step_0000` | 38.64% | 16.19% | 42.92% | 1.11% | 17.44% | `step_0000` | 38.64% | `step_0000` (38.64%) |
| `checker_reward_seed0` | `checker_reward` | 0 | `step_0015` | 40.05% | 17.31% | 44.35% | 1.87% | 19.50% | `step_0020` | 37.66% | `step_0010` (38.11%) |
| `drgrpo_pass1_seed0` | `drgrpo_pass1` | 0 | `step_0005` | 39.02% | 17.56% | 44.16% | 1.28% | 19.44% | `step_0030` | 37.95% | `step_0025` (38.12%) |
| `drgrpo_pass1_seed1` | `drgrpo_pass1` | 1 | `step_0030` | 39.62% | 16.94% | 44.05% | 1.43% | 18.62% | `step_0030` | 39.62% | `step_0030` (39.62%) |
| `pkpo_pass8_seed0` | `pkpo_pass8` | 0 | `step_0020` | 40.68% | 16.69% | 45.48% | 1.71% | 18.06% | `step_0030` | 38.92% | `step_0005` (39.01%) |
| `pkpo_pass8_seed1` | `pkpo_pass8` | 1 | `step_0020` | 39.13% | 16.81% | 42.33% | 1.16% | 18.81% | `step_0030` | 37.55% | `step_0025` (38.24%) |
| `sspo_seed0` | `sspo` | 0 | `step_0020` | 40.09% | 17.38% | 44.67% | 1.50% | 19.75% | `step_0030` | 39.53% | `step_0025` (38.18%) |
| `sspo_seed1` | `sspo` | 1 | `step_0030` | 39.79% | 17.31% | 44.28% | 1.71% | 19.19% | `step_0030` | 39.79% | `step_0020` (39.74%) |
| `sspo_seed2` | `sspo` | 2 | `step_0010` | 40.46% | 17.50% | 44.77% | 1.49% | 19.25% | `step_0030` | 39.00% | `step_0020` (38.84%) |
| `sspo_turns2_seed0` | `sspo_turns2` | 0 | `step_0030` | **43.22%** | **22.06%** | **48.04%** | 1.85% | **24.31%** | `step_0030` | **43.22%** | `step_0015` (**41.78%**) |

Full per-checkpoint progression across all 106 `dev` + `test` checkpoint snapshots is available in [`results/tables/checkpoint_progression.csv`](../results/tables/checkpoint_progression.csv).

---

## 3. Context Window Analysis Reference (`40,960` Context vs Archived `4,096` Context)

For analysis reference, the earlier `4,096`-token context runs (`max_tokens=1536`, `max_train_seq_len=2048`) are preserved in `results_4k_archive/` (not used for active benchmark evaluation):

| Method (`arm`) | `40,960` Context Best `Select@8` | `40,960` Context Final `Select@8` | Archived `4,096` Context Best `Select@8` | Archived `4,096` Context Final `Select@8` |
|---|---:|---:|---:|---:|
| **`sspo_turns2`** | **43.22%** | **43.22%** | 19.88% | 15.56% |
| **`sspo`** | **40.11%** | **39.44%** | 18.21% | 17.34% |
| **`pkpo_pass8`** | 39.91% | 38.24% | 18.45% | 18.03% |
| **`drgrpo_pass1`** | 39.32% | 38.79% | 18.44% | 16.54% |
| **`checker_reward`** | 40.05% | 37.66% | 18.12% | 15.68% |
| **`frozen`** | 38.64% | 38.64% | 16.84% | 16.84% |
