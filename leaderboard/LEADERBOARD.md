# SSPO-Bench Official Baseline Leaderboard (`Qwen/Qwen3-4B` @ `40,960` Context)

All baselines are trained and evaluated under the locked SSPO-Bench protocol across **two strictly disjoint competitive-programming benchmarks** (**Benchmark 1**: `data/` & `results/`, 21 runs; **Benchmark 2**: `data_b2/` & `results_b2/`, 19 runs; `0` shared `pid`s and `0` shared `question`s):
- **Base Policy**: `Qwen/Qwen3-4B` (LoRA `r=64, alpha=128`, `bfloat16`)
- **Context Window**: `max_context_length=40960`, `max_tokens=8192`, `max_train_seq_len=8192`
- **Training Budget**: `30` steps (`20` steps for `checker_reward`), `batch_problems=16`, `n_rollouts=16`, `k_deploy=8`, `lr=1e-6`, `kl_coef=0.001`, `clip_eps=0.2`
- **Dataset Splits (`contest_id`-disjoint)**:
  - **Benchmark 1 (`configs/pilot.yaml`)**: `91` active calibrated training problems (`data/train.jsonl`), `100` validation problems (`data/dev.jsonl`), and `100` encrypted held-out test problems (`data/test.jsonl` + `data/test_hidden.enc`).
  - **Benchmark 2 (`configs/pilot_b2.yaml`)**: `64` active calibrated training problems (`data_b2/train.jsonl`), `64` validation problems (`data_b2/dev.jsonl`), and `64` encrypted held-out test problems (`data_b2/test.jsonl` + `data_b2/test_hidden.enc.part*`).
- **Evaluation Protocol**: Held-out test problems evaluated with deterministic paired rollout seeds (`subset_seed + c_start + i`), `N_eval=16` rollouts per problem at `temperature=1.0`, `K=8` deployment budget with public-example filter + execution-signature majority-vote selector $\mathcal{S}$.

---

## 1. Benchmark 1 Main Leaderboard (`data/` & `results/`, 100 Test Problems, 21 Runs)

| Method (`arm`) | Turns | Seeds | Peak `Select@8` (Mean ± Std) | Final `Select@8` (`step_0030`) | Val-Selected Test `Select@8` (`dev` $\to$ `test`) | Final `Dev` `Select@8` | `Pass@1` (`step_0030`) | `Pass@8` (`step_0030`) | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`sspo_turns2`** *(Multi-Turn SSPO)* | 2 | 5 (`seed0..4`) | **43.34% ± 1.13%** | **41.49% ± 1.25%** | **41.59% ± 0.92%** | **36.00% ± 1.66%** | **21.18% ± 0.55%** | **46.01% ± 1.17%** | Two-turn public-checker execution feedback + SSPO; beats all 1-turn arms on `Select@8` and `Pass@1` ($p < 0.0001$) |
| **`tang_maj_loo`** *(Tang et al., 2503.19595)* | 1 | 3 (`seed0..2`) | 41.02% ± 0.40% | 39.71% ± 1.34% | 39.33% ± 1.18% | 32.74% ± 0.56% | 16.88% ± 0.41% | 42.71% ± 0.46% | Mean-centered LOO majority-vote estimator ($A_i = f(Y)-f(Y_{-i})$, $A'_i = A_i - \bar{A}$) |
| **`sspo`** *(1-Turn Selection-Aware RL)* | 1 | 5 (`seed0..4`) | 39.80% ± 0.77% | 39.23% ± 0.50% | 38.92% ± 0.63% | 33.33% ± 0.51% | 17.10% ± 0.48% | 43.38% ± 1.33% | Exact $O(n^2 k)$ DP subset-LOO selector marginal credit ($\alpha=0.7$); 7-seed (`seed0..6`) peak mean `39.93% ± 0.68%` |
| **`checker_reward`** *(Public-Only RL)* | 1 | 1 (`seed0`) | 40.05% ± 0.00% | 37.66% ± 0.00% | 38.11% ± 0.00% | 31.30% ± 0.00% | 16.12% ± 0.00% | 42.15% ± 0.00% | Collapses at step 20 (`-1.29 pt` below `frozen`) due to public-test false positives |
| **`pkpo_pass8`** *(DeepMind, 2505.15201)* | 1 | 2 (`seed0..1`) | 39.91% ± 1.10% | 38.24% ± 0.97% | 38.62% ± 0.54% | 31.10% ± 0.60% | 17.03% ± 0.66% | 42.59% ± 0.11% | Exact `Pass@8` subset-LOO estimator $s_i^{\text{loo}-1}$ on $g(y)$ |
| **`drgrpo_pass1`** *(Starting Agent Template)* | 1 | 2 (`seed0..1`) | 39.32% ± 0.42% | 38.79% ± 1.18% | 38.87% ± 1.06% | 33.40% ± 2.83% | 16.88% ± 0.09% | 43.23% ± 1.16% | Standard `Pass@1` Dr. GRPO on $g(y)$ |
| **`frozen`** *(Untrained Base Model)* | 1 | 1 (`seed0`) | 38.95% ± 0.00% | 38.95% ± 0.00% | 38.95% ± 0.00% | 32.23% ± 0.00% | 16.50% ± 0.00% | 43.62% ± 0.00% | Zero-shot `Qwen/Qwen3-4B` (`40,960` context, paired seed-matched eval) |

---

## 2. Benchmark 2 Main Leaderboard (`data_b2/` & `results_b2/`, 64 Test Problems, 19 Runs)

| Method (`arm`) | Turns | Seeds | Peak `Select@8` (Mean ± Std) | Val-Selected Test `Select@8` (`dev` $\to$ `test`) | Epoch-5 (`step_0020`) `Select@8` | Final `Select@8` (`step_0030`) | Final `Dev` `Select@8` | `Pass@1` (`step_0030`) | `Pass@8` (`step_0030`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`sspo_turns2`** *(Multi-Turn SSPO)* | 2 | 5 (`seed0..4`) | **46.95% ± 1.19%** | **44.55% ± 1.71%** | **43.15% ± 3.21%** | **43.90% ± 1.57%** | **41.67% ± 1.40%** | **22.30% ± 0.53%** | **50.38% ± 2.03%** |
| **`checker_reward`** *(Public-Only RL)* | 1 | 1 (`seed0`) | 40.87% ± 0.00% | 37.09% ± 0.00% | 40.87% ± 0.00% | 40.87% ± 0.00% | 35.37% ± 0.00% | 21.09% ± 0.00% | 48.39% ± 0.00% |
| **`drgrpo_pass1`** *(Starting Agent Template)* | 1 | 2 (`seed0..1`) | 40.52% ± 1.00% | 38.02% ± 0.91% | 38.71% ± 1.88% | 38.16% ± 0.71% | 35.50% ± 0.40% | 19.87% ± 0.76% | 45.42% ± 0.10% |
| **`tang_maj_loo`** *(Tang et al., 2503.19595)* | 1 | 3 (`seed0..2`) | 40.34% ± 1.29% | 37.92% ± 1.67% | 37.64% ± 1.11% | 38.33% ± 0.72% | 33.62% ± 1.37% | 19.43% ± 0.43% | 45.33% ± 2.45% |
| **`pkpo_pass8`** *(DeepMind, 2505.15201)* | 1 | 2 (`seed0..1`) | 40.15% ± 1.86% | 37.05% ± 2.07% | 35.79% ± 0.28% | 37.86% ± 0.93% | 33.82% ± 1.70% | 19.87% ± 0.21% | 43.50% ± 1.65% |
| **`sspo`** *(1-Turn Selection-Aware RL)* | 1 | 5 (`seed0..4`) | 39.18% ± 0.40% | 37.31% ± 1.38% | 38.41% ± 1.94% | 37.17% ± 0.89% | 34.74% ± 3.04% | 19.36% ± 0.55% | 43.60% ± 1.00% |
| **`frozen`** *(Untrained Base Model)* | 1 | 1 (`seed0`) | 33.95% ± 0.00% | 33.95% ± 0.00% | 33.95% ± 0.00% | 33.95% ± 0.00% | 33.27% ± 0.00% | 18.36% ± 0.00% | 40.44% ± 0.00% |

> **Key Statistical Highlights on Benchmark 2**:
> - Under both **Val-Selected Checkpoint (`dev` $\to$ `test`)** and **Peak Checkpoint**, **`sspo_turns2`** (`5` seeds) beats **all 6 baselines** (`frozen`, `sspo`, `tang_maj_loo`, `drgrpo_pass1`, `pkpo_pass8`, `checker_reward`) with **Holm-Bonferroni corrected significance ($p_{\text{holm}} \le 0.036$, `win = True` on 6/6 comparisons)**, achieving **`+10.60 pt`** (`dev`-selected) and **`+13.00 pt`** (peak) over `frozen_seed0` (`33.95%`).
> - Single-turn **`sspo`** (`5` seeds) significantly outperforms `frozen_seed0` (`33.95%`) at both the epoch-matched `step_0020` (`5` epochs, **`38.41%`**, `+4.46 pt`, `95% CI: [+0.39, +9.01]`, $p=0.032$) and peak checkpoint (**`39.18%`**, `+5.22 pt`, `95% CI: [+1.17, +9.74]`, $p_{\text{holm}}=0.040$).

---

## 3. Per-Seed Breakdown — Benchmark 1 (`results/`, 21 Runs)

| Run Name | Arm | Seed | Peak Checkpoint | Peak `Select@8` | Final (`step_0030`) `Select@8` | Final `Pass@1` | Final `Pass@8` | Final `Dev` `Select@8` | Val-Selected Checkpoint (`dev` $\to$ `test`) |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `frozen_seed0` | `frozen` | 0 | `step_0000` | 38.95% | 38.95% | 16.50% | 43.62% | 32.23% | `step_0000` (`dev=32.23%` $\to$ `test=38.95%`) |
| `checker_reward_seed0` | `checker_reward` | 0 | `step_0015` | 40.05% | 37.66% (`step_0020`) | 16.12% | 42.15% | 31.30% | `step_0010` (`dev=32.34%` $\to$ `test=38.11%`) |
| `drgrpo_pass1_seed0` | `drgrpo_pass1` | 0 | `step_0005` | 39.02% | 37.95% | 16.81% | 42.40% | 31.40% | `step_0025` (`dev=32.10%` $\to$ `test=38.12%`) |
| `drgrpo_pass1_seed1` | `drgrpo_pass1` | 1 | `step_0030` | 39.62% | 39.62% | 16.94% | 44.05% | 35.40% | `step_0030` (`dev=35.40%` $\to$ `test=39.62%`) |
| `pkpo_pass8_seed0` | `pkpo_pass8` | 0 | `step_0020` | 40.68% | 38.92% | 17.50% | 42.51% | 31.52% | `step_0005` (`dev=34.45%` $\to$ `test=39.01%`) |
| `pkpo_pass8_seed1` | `pkpo_pass8` | 1 | `step_0020` | 39.13% | 37.55% | 16.56% | 42.66% | 30.67% | `step_0025` (`dev=32.98%` $\to$ `test=38.24%`) |
| `tang_maj_loo_seed0` | `tang_maj_loo` | 0 | `step_0015` | 41.25% | 39.06% | 17.31% | 42.38% | 33.27% | `step_0010` (`dev=34.23%` $\to$ `test=40.14%`) |
| `tang_maj_loo_seed1` | `tang_maj_loo` | 1 | `step_0030` | 41.25% | 41.25% | 16.81% | 42.53% | 32.78% | `step_0015` (`dev=33.71%` $\to$ `test=39.87%`) |
| `tang_maj_loo_seed2` | `tang_maj_loo` | 2 | `step_0010` | 40.55% | 38.82% | 16.50% | 43.23% | 32.16% | `step_0025` (`dev=33.59%` $\to$ `test=37.98%`) |
| `sspo_seed0` | `sspo` | 0 | `step_0020` | 40.09% | 39.53% | 16.75% | 44.98% | 32.79% | `step_0025` (`dev=32.96%` $\to$ `test=38.18%`) |
| `sspo_seed1` | `sspo` | 1 | `step_0030` | 39.79% | 39.79% | 17.31% | 44.28% | 32.81% | `step_0020` (`dev=33.23%` $\to$ `test=39.74%`) |
| `sspo_seed2` | `sspo` | 2 | `step_0010` | 40.46% | 39.00% | 17.44% | 42.25% | 33.46% | `step_0020` (`dev=33.84%` $\to$ `test=38.84%`) |
| `sspo_seed3` | `sspo` | 3 | `step_0030` | 38.49% | 38.49% | 17.56% | 43.54% | 33.86% | `step_0030` (`dev=33.86%` $\to$ `test=38.49%`) |
| `sspo_seed4` | `sspo` | 4 | `step_0025` | 40.15% | 39.34% | 16.44% | 41.84% | 33.75% | `step_0030` (`dev=33.75%` $\to$ `test=39.34%`) |
| `sspo_seed5` *(extra)* | `sspo` | 5 | `step_0010` | 40.49% | 38.38% | 16.62% | 42.72% | 32.84% | `step_0020` (`dev=33.18%` $\to$ `test=36.41%`) |
| `sspo_seed6` *(extra)* | `sspo` | 6 | `step_0010` | 40.07% | 38.32% | 16.62% | 41.34% | 33.41% | `step_0020` (`dev=35.03%` $\to$ `test=39.27%`) |
| `sspo_turns2_seed0` | `sspo_turns2` | 0 | `step_0030` | **43.22%** | **43.22%** | **22.06%** | **48.04%** | 35.18% | `step_0015` (`dev=37.46%` $\to$ `test=41.78%`) |
| `sspo_turns2_seed1` | `sspo_turns2` | 1 | `step_0025` | **42.38%** | **41.06%** | **21.06%** | **45.68%** | 34.39% | `step_0020` (`dev=38.50%` $\to$ `test=40.99%`) |
| `sspo_turns2_seed2` | `sspo_turns2` | 2 | `step_0015` | **43.63%** | **42.11%** | **20.88%** | **45.67%** | **38.26%** | `step_0010` (`dev=39.81%` $\to$ `test=43.02%`) |
| `sspo_turns2_seed3` | `sspo_turns2` | 3 | `step_0015` | **45.10%** | **39.90%** | **20.62%** | **45.67%** | 37.24% | `step_0025` (`dev=39.30%` $\to$ `test=40.61%`) |
| `sspo_turns2_seed4` | `sspo_turns2` | 4 | `step_0020` | **42.35%** | **41.15%** | **21.25%** | **45.01%** | 34.93% | `step_0015` (`dev=38.72%` $\to$ `test=41.54%`) |

---

## 4. Per-Seed Breakdown — Benchmark 2 (`results_b2/`, 19 Runs)

| Run Name | Arm | Seed | Peak Checkpoint | Peak `Select@8` | Epoch-5 (`step_0020`) `Select@8` | Final (`step_0030`) `Select@8` | Final `Pass@1` | Final `Pass@8` | Final `Dev` `Select@8` | Val-Selected Checkpoint (`dev` $\to$ `test`) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `frozen_seed0` | `frozen` | 0 | `step_0000` | 33.95% | 33.95% | 33.95% | 18.36% | 40.44% | 33.27% | `step_0000` (`dev=33.27%` $\to$ `test=33.95%`) |
| `checker_reward_seed0` | `checker_reward` | 0 | `step_0020` | 40.87% | 40.87% | 40.87% (`step_0020`) | 21.09% | 48.39% | 35.37% | `step_0015` (`dev=35.48%` $\to$ `test=37.09%`) |
| `drgrpo_pass1_seed0` | `drgrpo_pass1` | 0 | `step_0015` | 41.23% | 40.04% | 38.67% | 20.41% | 45.35% | 35.78% | `step_0030` (`dev=35.78%` $\to$ `test=38.67%`) |
| `drgrpo_pass1_seed1` | `drgrpo_pass1` | 1 | `step_0010` | 39.82% | 37.38% | 37.66% | 19.34% | 45.49% | 35.21% | `step_0020` (`dev=36.02%` $\to$ `test=37.38%`) |
| `pkpo_pass8_seed0` | `pkpo_pass8` | 0 | `step_0025` | 38.83% | 35.99% | 38.51% | 19.73% | 44.66% | 35.02% | `step_0030` (`dev=35.02%` $\to$ `test=38.51%`) |
| `pkpo_pass8_seed1` | `pkpo_pass8` | 1 | `step_0010` | 41.47% | 35.59% | 37.20% | 20.02% | 42.33% | 32.62% | `step_0020` (`dev=35.61%` $\to$ `test=35.59%`) |
| `tang_maj_loo_seed0` | `tang_maj_loo` | 0 | `step_0010` | 41.15% | 37.77% | 37.53% | 18.95% | 42.75% | 35.18% | `step_0030` (`dev=35.18%` $\to$ `test=37.53%`) |
| `tang_maj_loo_seed1` | `tang_maj_loo` | 1 | `step_0010` | 41.02% | 38.69% | 38.95% | 19.73% | 45.60% | 33.04% | `step_0005` (`dev=35.05%` $\to$ `test=39.75%`) |
| `tang_maj_loo_seed2` | `tang_maj_loo` | 2 | `step_0015` | 38.85% | 36.47% | 38.51% | 19.63% | 47.63% | 32.65% | `step_0020` (`dev=35.14%` $\to$ `test=36.47%`) |
| `sspo_seed0` | `sspo` | 0 | `step_0025` | 39.13% | 39.00% | 36.59% | 19.53% | 43.34% | 36.81% | `step_0030` (`dev=36.81%` $\to$ `test=36.59%`) |
| `sspo_seed1` | `sspo` | 1 | `step_0005` | 39.44% | 39.39% | 36.54% | 18.95% | 43.27% | 37.45% | `step_0030` (`dev=37.45%` $\to$ `test=36.54%`) |
| `sspo_seed2` | `sspo` | 2 | `step_0020` | 39.66% | 39.66% | 37.46% | 18.85% | 45.10% | 31.36% | `step_0025` (`dev=35.42%` $\to$ `test=39.34%`) |
| `sspo_seed3` | `sspo` | 3 | `step_0030` | 38.61% | 34.98% | 38.61% | 20.21% | 43.88% | 36.58% | `step_0015` (`dev=36.73%` $\to$ `test=37.98%`) |
| `sspo_seed4` | `sspo` | 4 | `step_0020` | 39.04% | 39.04% | 36.65% | 19.24% | 42.38% | 31.49% | `step_0025` (`dev=35.91%` $\to$ `test=35.85%`) |
| `sspo_turns2_seed0` | `sspo_turns2` | 0 | `step_0020` | **47.41%** | **47.41%** | **43.69%** | **22.36%** | **50.89%** | **43.84%** | `step_0015` (`dev=45.08%` $\to$ `test=45.52%`) |
| `sspo_turns2_seed1` | `sspo_turns2` | 1 | `step_0005` | **48.67%** | **43.80%** | **46.47%** | **22.95%** | **52.83%** | **41.54%** | `step_0015` (`dev=42.47%` $\to$ `test=44.83%`) |
| `sspo_turns2_seed2` | `sspo_turns2` | 2 | `step_0015` | **45.62%** | **44.61%** | **44.08%** | **22.46%** | **51.46%** | **39.91%** | `step_0020` (`dev=40.34%` $\to$ `test=44.61%`) |
| `sspo_turns2_seed3` | `sspo_turns2` | 3 | `step_0005` | **46.13%** | **40.43%** | **42.59%** | **22.27%** | **47.69%** | **41.49%** | `step_0005` (`dev=42.36%` $\to$ `test=46.13%`) |
| `sspo_turns2_seed4` | `sspo_turns2` | 4 | `step_0015` | **46.92%** | **39.53%** | **42.68%** | **21.48%** | **49.02%** | **41.59%** | `step_0005` (`dev=43.11%` $\to$ `test=41.68%`) |

Full per-checkpoint progression across all 443 `dev` + `test` checkpoint snapshots (230 on Benchmark 1 + 213 on Benchmark 2) is available in [`results/tables/checkpoint_progression.csv`](../results/tables/checkpoint_progression.csv) and [`results_b2/tables/checkpoint_progression.csv`](../results_b2/tables/checkpoint_progression.csv).
