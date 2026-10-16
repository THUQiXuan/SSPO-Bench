# SSPO v1 实验执行打勾清单 (`FRIEND_CHECKLIST.md`)

这份清单供你在 GPU 机器上按顺序打勾核对。每完成一步即可打勾 `[x]`；若中途中断，直接重跑当前未打勾的那条命令即可自动续跑。

---

## 0. 启动前准备
- [ ] **进入持久化终端**：已启动 `tmux new -s sspo` 或 `screen -S sspo`（防止 SSH 断连中断任务）。
- [ ] **检查 GPU 与驱动**：运行 `nvidia-smi` 确认可见 1–16 张 H100/A100 GPU，且 Driver Version `>= 535`。
- [ ] **检查磁盘空间**：运行 `df -h .` 确认当前目录所在分区可用空间 `>= 120 GB`。
- [ ] **检查网络连接**：项目已内置固定 `HF_TOKEN`（无需手动设置；如在国内网络，可选执行 `export HF_ENDPOINT="https://hf-mirror.com"`）。

---

## 1. 环境、模型与数据准备
- [ ] **安装 GPU 环境**（预计 5–10 分钟）：
  - 命令：`bash scripts/setup_env.sh`
  - 验收标志：生成 `.venv/done`，且脚本输出 `Environment setup complete!`。
- [ ] **下载基座模型权重**（预计 3–8 分钟）：
  - 命令：`bash scripts/download_models.sh`
  - 验收标志：生成 `cache/models/done`。
- [ ] **下载并过滤 CodeContests 数据集**（预计 5–10 分钟）：
  - 命令：`bash scripts/prepare_data.sh`
  - 验收标志：生成 `data/done`、`data/filter_stats.json`、`data/code_contests_train.jsonl`、`data/code_contests_dev200.jsonl`、`data/code_contests_test.jsonl`。

---

## 2. 10 分钟 Smoke 环境验证
- [ ] **跑通 Smoke 端到端验证**（8 题、2 步训练 + 评测，预计 5–10 分钟）：
  - 命令：`bash scripts/run_all.sh --smoke-only`
  - 验收标志：终端显示 `[run_all] --smoke-only specified. Smoke verification succeeded!`，且生成 `results/smoke_sspo/summary.json`。

---

## 3. 免训练诊断（Gate 1）与检查器构建
*(注：如果你直接运行 `bash scripts/run_all.sh`，以下第 3、4、5 节会自动按顺序全部执行，你可以直接看文件生成进度打勾)*

- [ ] **构建冻结执行检查器（生成辅助程序与确认参照）**：
  - 命令：`bash scripts/build_checkers.sh`
  - 验收标志：生成 `cache/checkers/done` 与 `cache/checkers/ec_SystemTests.jsonl`。
- [ ] **运行免训练诊断（Gate 1）并筛出活跃训练题池**：
  - 命令：`bash scripts/run_diag.sh`
  - 验收标志：生成 `results/diag/done`、`results/diag/summary.json`（确认 `dp_max_abs_diff < 1e-9`）以及 `data/active_train_pool.jsonl`。

---

## 4. 五个实验对比臂训练（Gate 2）
- [ ] **臂 1：`sspo`（主方法，$\alpha=0.7, J^-$，150 步，seed 0）**：
  - 命令：`bash scripts/run_train.sh sspo 0`
  - 验收标志：生成 `results/sspo_seed0/done` 与 `results/sspo_seed0/metrics.jsonl`（共 150 行）。
- [ ] **臂 2：`drgrpo_pass1`（Dr. GRPO pass@1 基线，150 步，seed 0）**：
  - 命令：`bash scripts/run_train.sh drgrpo_pass1 0`
  - 验收标志：生成 `results/drgrpo_pass1_seed0/done` 与 `results/drgrpo_pass1_seed0/metrics.jsonl`。
- [ ] **臂 3：`pkpo_pass8`（PKPO pass@8 闭式基线，150 步，seed 0）**：
  - 命令：`bash scripts/run_train.sh pkpo_pass8 0`
  - 验收标志：生成 `results/pkpo_pass8_seed0/done` 与 `results/pkpo_pass8_seed0/metrics.jsonl`。
- [ ] **臂 4：`checker_reward`（检查器奖励作弊探针臂，50 步，seed 0）**：
  - 命令：`bash scripts/run_train.sh checker_reward 0`
  - 验收标志：生成 `results/checker_reward_seed0/done` 与 `results/checker_reward_seed0/metrics.jsonl`。
- [ ] **臂 5：`frozen`（不训练，基座策略多采样 + 执行挑选器，seed 0）**：
  - 命令：`bash scripts/run_train.sh frozen 0`
  - 验收标志：生成 `results/frozen_seed0/done`。
- [ ] **（可选推荐）第二随机种子复现（`seed 1`）**：
  - 命令：`bash scripts/run_train.sh sspo 1` 与 `bash scripts/run_train.sh drgrpo_pass1 1`
  - 验收标志：生成 `results/sspo_seed1/done` 与 `results/drgrpo_pass1_seed1/done`。

---

## 5. 独立采样评测与汇总报告生成
- [ ] **评测 `sspo_seed0`**：
  - 命令：`bash scripts/run_eval.sh results/sspo_seed0 latest`
  - 验收标志：生成 `results/sspo_seed0/summary.json` 与 `results/sspo_seed0/eval/test_latest.jsonl`。
- [ ] **评测 `drgrpo_pass1_seed0`**：
  - 命令：`bash scripts/run_eval.sh results/drgrpo_pass1_seed0 latest`
  - 验收标志：生成 `results/drgrpo_pass1_seed0/summary.json`。
- [ ] **评测 `pkpo_pass8_seed0`**：
  - 命令：`bash scripts/run_eval.sh results/pkpo_pass8_seed0 latest`
  - 验收标志：生成 `results/pkpo_pass8_seed0/summary.json`。
- [ ] **评测 `checker_reward_seed0`**：
  - 命令：`bash scripts/run_eval.sh results/checker_reward_seed0 latest`
  - 验收标志：生成 `results/checker_reward_seed0/summary.json`。
- [ ] **评测 `frozen_seed0`**：
  - 命令：`bash scripts/run_eval.sh results/frozen_seed0 latest`
  - 验收标志：生成 `results/frozen_seed0/summary.json`。
- [ ] **检查汇总表与实验报告已自动生成**：
  - 验收标志：存在 `results/tables/main_comparison.csv` 与 `results/report.md`。

---

## 6. 打包回传结果
- [ ] **生成回传压缩包**：
  - 命令：`bash scripts/collect_results.sh`
  - 验收标志：仓库根目录生成 `sspo_results_bundle.zip`（终端打印文件大小与内容摘要）。
- [ ] **把 `sspo_results_bundle.zip` 发回** 🎉
