# SSPO-Bench 工作审查指南与 Reviewer Agent 提示词 (`REVIEWER_GUIDE.md`)

本文档旨在帮助人类审查者或 **Reviewer Agent** 快速定位、核验并审计 **SSPO-Bench** 项目的所有代码实现、数据集切分、双基准（Benchmark 1 & Benchmark 2）实验原始日志、统计检验表与防作弊评测沙箱。

---

## 1. 直接发给 Reviewer Agent 的提示词（Prompt）

你可以直接复制以下提示词发送给 Reviewer Agent：

```markdown
请你作为独立的 Reviewer Agent，对位于 `/Users/xuan/codex_workspace/sspo_v1` 的 SSPO-Bench 项目进行全面审查。
请先阅读审查导航文件 `/Users/xuan/codex_workspace/sspo_v1/REVIEWER_GUIDE.md`，然后按以下维度独立核查并给出审查报告：

1. **实验设置与上下文窗口（40,960 Context）**：
   - 检查 `/Users/xuan/codex_workspace/sspo_v1/configs/benchmark_locked.yaml`、`configs/pilot.yaml` 和 `configs/pilot_b2.yaml`，确认基座模型是否为 `Qwen/Qwen3-4B`、上下文长度 `max_context_length` 是否设为 `40960`（`max_tokens: 8192`, `max_train_seq_len: 8192`），且所有对比方法共享相同的超参数。

2. **双基准数据集隔离性与防泄漏（`data/` 与 `data_b2/`）**：
   - 检查 Benchmark 1（`/Users/xuan/codex_workspace/sspo_v1/data/`）与 Benchmark 2（`/Users/xuan/codex_workspace/sspo_v1/data_b2/`）内部的 `train.jsonl`、`dev.jsonl`、`test.jsonl` 是否严格按 `contest_id` 零交集切分；
   - 验证 Benchmark 1 与 Benchmark 2 之间是否 `0` 题目 ID（`pid`）重叠、`0` 题目文本（`question`）重叠；
   - 验证磁盘上的 `test.jsonl` 是否已剥离隐藏测试集（`"hidden": []`），且加密保险箱 `data/test_hidden.enc` 与 `data_b2/test_hidden.enc.part00..05` 能否通过 `benchmark/hidden_test_vault.py` 正常在内存中解密。

3. **对比方法与数学实现的正确性（7 种方法）**：
   - 检查 `/Users/xuan/codex_workspace/sspo_v1/sspo/dp.py`、`/Users/xuan/codex_workspace/sspo_v1/sspo/advantage.py` 以及 `/Users/xuan/codex_workspace/sspo_v1/benchmark/reference_algorithms/`，核对以下方法的实现是否与数学定义及原论文严格一致：
     1. `frozen`（零样本基座，配对种子评测）
     2. `checker_reward`（仅公开样例奖励 RL）
     3. `drgrpo_pass1`（标准 Pass@1 Dr. GRPO）
     4. `pkpo_pass8`（DeepMind `2505.15201` 全子集无偏留一法 Pass@8 估计器 $s_i^{\text{loo}-1}$）
     5. `tang_maj_loo`（Meta Tang et al. `2503.19595` 均值中心化全池留一多数投票估计器 $A_i = f(Y) - f(Y_{-i}), A'_i = A_i - \bar{A}$）
     6. `sspo`（我们的单轮挑选感知策略优化，$\binom{n-1}{K-1}$ 精确 DP 留一法挑选器边际信用 $\Delta_i^{\mathcal{S}}$，$\alpha=0.7$，跑满 5+ 个随机种子）
     7. `sspo_turns2`（我们的两轮公开样例执行反馈修正 + SSPO，跑满 5 个随机种子）

4. **双基准实验原始数据完整性与统计检验（40 个训练运行，443 份检查点评测）**：
   - 核查 Benchmark 1（`/Users/xuan/codex_workspace/sspo_v1/results/`，21 个运行，230 份检查点评测）与 Benchmark 2（`/Users/xuan/codex_workspace/sspo_v1/results_b2/`，19 个运行，213 份检查点评测）下的原始训练指标 `metrics.jsonl`、逐题评测快照 `eval/dev_step_*.jsonl` 与 `eval/test_step_*.jsonl` 是否真实、完整；
   - 核对 `/Users/xuan/codex_workspace/sspo_v1/results/ANALYSIS_REPORT.md`、`/Users/xuan/codex_workspace/sspo_v1/leaderboard/LEADERBOARD.md`、`/Users/xuan/codex_workspace/sspo_v1/leaderboard/baselines.json` 中的数字是否与原始评测文件及 `results/tables/*.csv`、`results_b2/tables/*.csv` 完全一致。

5. **运行自动化防篡改校验与单元测试**：
   - 在 `/Users/xuan/codex_workspace/sspo_v1` 下使用 `/Users/xuan/codex_workspace/.venv_sspo_test/bin/python` 运行 `python -m benchmark.verifier`、`python -m benchmark.evaluate_agent smoke` 和 `python -m pytest -q`，确认全部通过。
```

---

## 2. 项目关键路径全景导航（Directory & File Map）

### 2.1 基础环境与仓库地址
| 项目 | 路径 / 地址 |
|---|---|
| **本地工作区根目录** | [`/Users/xuan/codex_workspace/sspo_v1`](file:///Users/xuan/codex_workspace/sspo_v1) |
| **本地 Python 测试环境** | `/Users/xuan/codex_workspace/.venv_sspo_test/bin/python` |
| **Hyak 集群工作区目录** | `hyak-login:/mmfs1/gscratch/simondu/xuanqi26/sspo_v1` |
| **GitHub 远程仓库** | `https://github.com/THUQiXuan/SSPO-Bench`（`main` 分支） |

---

### 2.2 核心分析报告与排行榜（建议最先阅读）
| 文件绝对路径 | 内容说明 |
|---|---|
| [`/Users/xuan/codex_workspace/sspo_v1/results/ANALYSIS_REPORT.md`](file:///Users/xuan/codex_workspace/sspo_v1/results/ANALYSIS_REPORT.md) | **Self-Contained 完整技术与实验分析报告**：从零讲清问题定义、挑选器两阶段机制、7 种对比方法的数学公式（含与 Meta `2503.19595`、DeepMind `2505.15201`、字节 Seed `2508.10751` 的关系）、`40,960` Context 锁定参数，以及 Benchmark 1 和 Benchmark 2 的完整实验与统计检验分析 |
| [`/Users/xuan/codex_workspace/sspo_v1/leaderboard/LEADERBOARD.md`](file:///Users/xuan/codex_workspace/sspo_v1/leaderboard/LEADERBOARD.md) | **官方双基准排行榜（Markdown）**：汇总 Benchmark 1（21 runs）和 Benchmark 2（19 runs）的跨种子均值 $\pm$ 标准差、逐种子明细以及与早期 4K Context 的对比 |
| [`/Users/xuan/codex_workspace/sspo_v1/leaderboard/baselines.json`](file:///Users/xuan/codex_workspace/sspo_v1/leaderboard/baselines.json) | **机器可读排行榜（JSON）**：包含双基准所有方法、每个种子的精确指标字典 |
| [`/Users/xuan/codex_workspace/sspo_v1/README.md`](file:///Users/xuan/codex_workspace/sspo_v1/README.md) | **项目主页文档**：任务概述、双基准主表、沙箱防作弊说明与 CLI 快速上手指南 |

---

### 2.3 双基准实验原始结果与统计表路径

#### （1）Benchmark 1：主基准（100 Test / 100 Dev / 91 Train，共 21 个训练运行，230 份检查点评测）
- **目录路径**：[`/Users/xuan/codex_workspace/sspo_v1/results/`](file:///Users/xuan/codex_workspace/sspo_v1/results)
- **自动生成的汇总报告**：[`/Users/xuan/codex_workspace/sspo_v1/results/report.md`](file:///Users/xuan/codex_workspace/sspo_v1/results/report.md)
- **汇总数据表（CSV）**：
  - 主对比表：[`/Users/xuan/codex_workspace/sspo_v1/results/tables/main_comparison.csv`](file:///Users/xuan/codex_workspace/sspo_v1/results/tables/main_comparison.csv)
  - 全检查点（`step_0000`..`step_0030`）轨迹表：[`/Users/xuan/codex_workspace/sspo_v1/results/tables/checkpoint_progression.csv`](file:///Users/xuan/codex_workspace/sspo_v1/results/tables/checkpoint_progression.csv)
  - 按比赛簇两级配对 Bootstrap 95% 置信区间与 Holm 校正表：[`/Users/xuan/codex_workspace/sspo_v1/results/tables/paired_bootstrap.csv`](file:///Users/xuan/codex_workspace/sspo_v1/results/tables/paired_bootstrap.csv)
  - 590 步训练曲线表：[`/Users/xuan/codex_workspace/sspo_v1/results/tables/training_curves.csv`](file:///Users/xuan/codex_workspace/sspo_v1/results/tables/training_curves.csv)
- **21 个运行子目录（每个子目录下含 `metrics.jsonl`、`summary.json`、`dev_progression.json`、`test_progression.json` 及 `eval/{dev,test}_step_*.jsonl` 逐题 16-rollout 原始评测记录）**：
  - 我们的方法：`sspo_seed0..6`（7 个种子）、`sspo_turns2_seed0..4`（5 个种子）
  - 文献与基线方法：`tang_maj_loo_seed0..2`（3 个种子）、`pkpo_pass8_seed0..1`（2 个种子）、`drgrpo_pass1_seed0..1`（2 个种子）、`checker_reward_seed0`（1 个种子）、`frozen_seed0`（1 个种子）

#### （2）Benchmark 2：独立验证基准（64 Test / 64 Dev / 64 Train，与 Benchmark 1 零题目重叠，共 19 个训练运行，213 份检查点评测）
- **目录路径**：[`/Users/xuan/codex_workspace/sspo_v1/results_b2/`](file:///Users/xuan/codex_workspace/sspo_v1/results_b2)
- **自动生成的汇总报告**：[`/Users/xuan/codex_workspace/sspo_v1/results_b2/report.md`](file:///Users/xuan/codex_workspace/sspo_v1/results_b2/report.md)
- **汇总数据表（CSV）**：
  - 主对比表：[`/Users/xuan/codex_workspace/sspo_v1/results_b2/tables/main_comparison.csv`](file:///Users/xuan/codex_workspace/sspo_v1/results_b2/tables/main_comparison.csv)
  - 全检查点轨迹表：[`/Users/xuan/codex_workspace/sspo_v1/results_b2/tables/checkpoint_progression.csv`](file:///Users/xuan/codex_workspace/sspo_v1/results_b2/tables/checkpoint_progression.csv)
  - 按比赛簇两级配对 Bootstrap 95% 置信区间表：[`/Users/xuan/codex_workspace/sspo_v1/results_b2/tables/paired_bootstrap.csv`](file:///Users/xuan/codex_workspace/sspo_v1/results_b2/tables/paired_bootstrap.csv)
  - 530 步训练曲线表：[`/Users/xuan/codex_workspace/sspo_v1/results_b2/tables/training_curves.csv`](file:///Users/xuan/codex_workspace/sspo_v1/results_b2/tables/training_curves.csv)
- **19 个运行子目录**：
  - 我们的方法：`sspo_seed0..4`（5 个种子）、`sspo_turns2_seed0..4`（5 个种子）
  - 文献与基线方法：`tang_maj_loo_seed0..2`（3 个种子）、`pkpo_pass8_seed0..1`（2 个种子）、`drgrpo_pass1_seed0..1`（2 个种子）、`checker_reward_seed0`（1 个种子）、`frozen_seed0`（1 个种子）

#### （3）早期 `4,096` Context 归档结果（仅作上下文长度消融对比参考）
- **目录路径**：[`/Users/xuan/codex_workspace/sspo_v1/results_4k_archive/`](file:///Users/xuan/codex_workspace/sspo_v1/results_4k_archive)

---

### 2.4 核心代码、数据集与评测沙箱路径
| 模块类别 | 绝对路径 | 审查要点 |
|---|---|---|
| **锁定超参配置** | [`/Users/xuan/codex_workspace/sspo_v1/configs/benchmark_locked.yaml`](file:///Users/xuan/codex_workspace/sspo_v1/configs/benchmark_locked.yaml)<br>[`/Users/xuan/codex_workspace/sspo_v1/configs/pilot.yaml`](file:///Users/xuan/codex_workspace/sspo_v1/configs/pilot.yaml)<br>[`/Users/xuan/codex_workspace/sspo_v1/configs/pilot_b2.yaml`](file:///Users/xuan/codex_workspace/sspo_v1/configs/pilot_b2.yaml) | `Qwen/Qwen3-4B`、`max_context_length: 40960`、`max_tokens: 8192`、LoRA $r=64, \alpha=128$、30 步训练 |
| **精确 DP 算法** | [`/Users/xuan/codex_workspace/sspo_v1/sspo/dp.py`](file:///Users/xuan/codex_workspace/sspo_v1/sspo/dp.py) | 分治多项式卷积精确计算 $\binom{n}{K}$ 挑选器胜率 $J_{\mathcal{S}}$ 与留一法边际贡献 $\Delta_i^{\mathcal{S}}$（误差 $< 10^{-14}$） |
| **7 种优势函数** | [`/Users/xuan/codex_workspace/sspo_v1/sspo/advantage.py`](file:///Users/xuan/codex_workspace/sspo_v1/sspo/advantage.py) | `drgrpo_pass1`、`pkpo_pass8`、`tang_maj_loo`、`checker_reward`、`sspo` 的核心计算与自适应权重 $w_x$ |
| **测试时挑选器** | [`/Users/xuan/codex_workspace/sspo_v1/sspo/selector.py`](file:///Users/xuan/codex_workspace/sspo_v1/sspo/selector.py) | 公开样例过滤 $V(Y_K)$ + 规范化执行输出签名 $\sigma(y)$ 多数投票 |
| **训练与评测循环** | [`/Users/xuan/codex_workspace/sspo_v1/sspo/trainer.py`](file:///Users/xuan/codex_workspace/sspo_v1/sspo/trainer.py)<br>[`/Users/xuan/codex_workspace/sspo_v1/sspo/eval.py`](file:///Users/xuan/codex_workspace/sspo_v1/sspo/eval.py)<br>[`/Users/xuan/codex_workspace/sspo_v1/sspo/report.py`](file:///Users/xuan/codex_workspace/sspo_v1/sspo/report.py) | 40K Context Rollout、2-Turn 修正提示构造、断点续跑、配对种子评测、两级 Bootstrap 检验 |
| **Agent 评测沙箱** | [`/Users/xuan/codex_workspace/sspo_v1/benchmark/agent_algorithm.py`](file:///Users/xuan/codex_workspace/sspo_v1/benchmark/agent_algorithm.py)<br>[`/Users/xuan/codex_workspace/sspo_v1/benchmark/verifier.py`](file:///Users/xuan/codex_workspace/sspo_v1/benchmark/verifier.py)<br>[`/Users/xuan/codex_workspace/sspo_v1/benchmark/evaluate_agent.py`](file:///Users/xuan/codex_workspace/sspo_v1/benchmark/evaluate_agent.py)<br>[`/Users/xuan/codex_workspace/sspo_v1/benchmark/hidden_test_vault.py`](file:///Users/xuan/codex_workspace/sspo_v1/benchmark/hidden_test_vault.py) | 46 个文件的 SHA-256 哈希锁（`manifest_locks.json`）、AST 静态导入/调用白名单、置换等变性与梯度流校验、隐藏测试集内存解密 |
| **参考算法实现** | [`/Users/xuan/codex_workspace/sspo_v1/benchmark/reference_algorithms/`](file:///Users/xuan/codex_workspace/sspo_v1/benchmark/reference_algorithms) | `drgrpo_pass1.py`, `pkpo_pass8.py`, `tang_maj_loo.py`, `checker_reward.py`, `sspo.py` |
| **双基准数据集** | [`/Users/xuan/codex_workspace/sspo_v1/data/`](file:///Users/xuan/codex_workspace/sspo_v1/data)<br>[`/Users/xuan/codex_workspace/sspo_v1/data_b2/`](file:///Users/xuan/codex_workspace/sspo_v1/data_b2) | Benchmark 1（91/100/100 题）与 Benchmark 2（64/64/64 题）数据切分与加密测试集 |
| **单元测试套件** | [`/Users/xuan/codex_workspace/sspo_v1/tests/`](file:///Users/xuan/codex_workspace/sspo_v1/tests) | 11 个测试文件（共 21 项测试），覆盖 DP 精度、定理性质、沙箱护栏、断点续跑与统计估计器 |

---

## 3. 一键核验命令（供 Reviewer 快速执行）

在 `/Users/xuan/codex_workspace/sspo_v1` 目录下运行以下命令，可在 30 秒内完成全部代码、数据隔离、加密保险箱、三层防作弊校验器与单元测试的自动验证：

```bash
# 1. 运行三层防作弊校验器（验证 46 个锁定文件的 SHA-256、AST 沙箱与数学契约）
/Users/xuan/codex_workspace/.venv_sspo_test/bin/python -m benchmark.verifier

# 2. 运行端到端 CPU Smoke 测试
/Users/xuan/codex_workspace/.venv_sspo_test/bin/python -m benchmark.evaluate_agent smoke

# 3. 运行全部 21 项单元测试（DP 精度 <= 1e-14、Theorem 2 性质、沙箱防篡改等）
/Users/xuan/codex_workspace/.venv_sspo_test/bin/python -m pytest -q

# 4. 重新从原始评测日志生成 Benchmark 1 与 Benchmark 2 的统计表与报告，验证数据 100% 自洽
/Users/xuan/codex_workspace/.venv_sspo_test/bin/python -m sspo.report --results-dir results
/Users/xuan/codex_workspace/.venv_sspo_test/bin/python -m sspo.report --results-dir results_b2 --data-dir data_b2
```
