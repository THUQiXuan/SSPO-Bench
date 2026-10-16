# SSPO v1 本地无 GPU 测试报告 (`TEST_REPORT.md`)

本报告记录了在无 GPU 的 macOS (Apple Silicon) 开发机上执行的全部 10 项必做单元测试、集成干跑与脚本静态检查结果，并明确列出仅能在目标 Linux GPU 机器上验证的部分。

---

## 1. 一键复现全部本地测试命令

在配置了轻量级本地 CPU 测试环境（见 `requirements-dev.txt`）后，运行以下单条命令即可复现本报告中的全部检查：

```bash
bash scripts/check_local.sh
```

实际执行输出摘要：
- `bash -n scripts/*.sh`：**11 / 11 脚本通过**
- `shellcheck -S error scripts/*.sh`：**11 / 11 脚本 0 错误通过**
- `pytest -v tests/`：**17 / 17 测试用例全部 PASSED（耗时 54.45 秒）**
- 端到端烟雾测试干跑 `bash scripts/run_all.sh --smoke-only --mock`：**PASSED**
- 定理 1(b) / 命题 C 探针 `bash scripts/run_probe.sh`：**PASSED**

---

## 2. 第 5 节十项必测用例逐项结果与命令

### 测试 1：动态规划正确性与簇不纯回退 (`tests/test_01_dp_correctness.py`)
- **运行命令**：`pytest -v tests/test_01_dp_correctness.py`
- **测试内容**：
  1. 随机生成 **200 组**类计数配置（覆盖 $n=16, k=8$ 以及小尺寸 $(n,k)\in\{(6,3),(8,4),(10,5),(12,6)\}$，包含并列最大错误簇、$\mathcal{W}^*\neq\emptyset$、全 $\mathcal{T}$、全 $\mathcal{C}$、单元素簇等全部边界情形）。
  2. 逐项比对动态规划（`COUNTS` / `TABLE` / `EXPECT` / `J` / `DIFF` / `UHAT`）算出的 $J_{\text{DP}}$、留一期望 $\hat{U}_{-i}$、挑选项优势 $A_i^S$ 与对全部 $\binom{n-1}{k-1}$ 个子集（$n=16, k=8$ 时为 $6435$ 个子集）暴力枚举的结果。
  3. 构造簇不纯样本（$\mathcal{C}$ 跨多个签名或某 $\mathcal{W}_m$ 与 $\mathcal{C}$ 同签名），验证自动触发 6435 子集精确枚举回退路径。
- **结果**：**PASSED**（DP 与暴力枚举最大绝对误差 $\le 8.88 \times 10^{-16} \ll 10^{-9}$）。

### 测试 2：定理 2 符号性质、PKPO 恒等式与不中心化反例 (`tests/test_02_theorem2_properties.py`)
- **运行命令**：`pytest -v tests/test_02_theorem2_properties.py`
- **测试内容**：
  1. 在 150 组簇纯度配置上验证定理 2：对 $i\in\mathcal{T}$ 恒有 $A_i^S = 0$；对 $g(y_i)=0$（含 $\mathcal{T}, \mathcal{W}^*, \mathcal{W}_m$）恒有 $A_i^S \le 0$ 且总优势 $A_i \le 0$；对 $i\in\mathcal{C}$ 恒有 $A_i^S \ge 0$。
  2. 交叉验证覆盖项闭式乘以 $k/n$ 后与 PKPO 二值闭式 $\frac{k}{n} A_i^{\text{cov}} = \mathbb{I}[i\in\mathcal{C}]\frac{\binom{n-n_{\mathcal{C}}}{k-1}}{\binom{n}{k}}$ 逐项相等（最大误差 $< 10^{-15}$）。
  3. 验证「为何挑选项不减留一均值」的两个数值反例：
     - $n=16, k=8, n_{\mathcal{C}}=1, n_{\mathcal{W}_1}=2, n_{\mathcal{T}}=13$：中心化后被淘汰程序获得正优势 $\approx +0.0210$（单候选优势差 $-0.0210394985$），不中心化时精确为 $0$。
     - $n=16, k=8, n_{\mathcal{C}}=2, n_{\mathcal{W}_1}=2, n_{\mathcal{T}}=12$：中心化后差值为 $-0.0271949246$（`-0.0272`），误差 $2.1 \times 10^{-16}$。
- **结果**：**PASSED**。

### 测试 3：无偏策略梯度蒙特卡洛与精确检验 (`tests/test_03_unbiased_mc.py`)
- **运行命令**：`pytest -v tests/test_03_unbiased_mc.py`
- **测试内容**：
  1. 构造 3 类离散表格式策略 $\pi_\theta = \text{softmax}(\theta)$（类别 $(\mathcal{C}, \mathcal{W}_1, \mathcal{T})$，$n=8, k=4$）。
  2. 用中心有限差分计算 $J_S^{(k)}(\theta)$ 与 $J_{\text{cov}}^{(k)}(\theta)$ 的真实梯度，先与精确多项分布期望 $\mathbb{E}[\frac{k}{n}\sum_{i=1}^n A_i \nabla_\theta \log \pi_\theta(y_i)]$ 比对（误差 $< 10^{-9}$），再与 $M=40{,}000$ 次独立蒙特卡洛采样均值比对。
- **结果**：**PASSED**（蒙特卡洛估计与有限差分梯度之差在 $2.5\,\text{SE}$ 抽样噪声内，相对误差 $< 1.2\%$）。

### 测试 4：冻结执行检查器流水线 (`tests/test_04_checker_pipeline.py`)
- **运行命令**：`pytest -v tests/test_04_checker_pipeline.py`
- **测试内容**：
  1. 使用 3 道手写测试题（两数之和、浮点均值、数组平方和），通过 `ec_gen --mock` 生成手写输入生成器、暴力解、校验器，完整跑通 `ec_build -> ec_run -> es_label -> evaluate_chk_and_sig`。
  2. 逐条验证 $\text{Chk}(y)$ 的充要条件：（a）公开样例全过；（b）所有压力输入在时限内无错跑完；（c）所有确认参照输入（$a(x)\ge 2$）与确认输出一致。
  3. 验证 $\text{sig}(y)$ 与 `out_key` 的规范化语义：多余空格/换行归一、浮点数按 6 位有效数字（`%.6g`）归一（`1.5`、`1.500000`、`1.5000001` 产生完全相同的 `sig`）、检查输入顺序固定。
- **结果**：**PASSED**。

### 测试 5：数据过滤、代表程序提取与打标流水线 (`tests/test_05_data_pipeline.py`)
- **运行命令**：`pytest -v tests/test_05_data_pipeline.py`
- **测试内容**：
  1. 对 `data/sample_problems.jsonl` 中的 25 道原始题运行 `sspo/data_pipeline.py`，断言 5 道不合规题（题面含 `"print any"` 多解描述、交互题/自定义 checker、测试数不足、隐藏测试期望输出不一致）被准确过滤，保留 **20 道**唯一答案题并写出 `data/filter_stats.json`。
  2. 验证单轮与双轮轨迹的代表程序提取规则 `extract_representative_program`：取最后一个通过全部公开样例的程序，若均未通过则取最后一个程序。
  3. 对 20 道题运行隐藏测试打标，校验输出 JSONL Schema 与 `README.md` 及 `data/schema.json` 一致。
- **结果**：**PASSED**。

### 测试 6：10 档 GPU 硬件预设生成与一致性校验 (`tests/test_06_config_presets.py`)
- **运行命令**：`pytest -v tests/test_06_config_presets.py`
- **测试内容**：
  遍历 `NUM_GPUS ∈ {1, 2, 4, 8, 16} × GPU_MEM_GB ∈ {40, 80}` 全部 10 个组合，验证张量并行度整除 `NUM_GPUS`、总采样预算 `batch_problems × n_samples` 被 `data_parallel_size × micro_batch_size × grad_accum_steps` 整除、`>= 4×80G` 选用 `Qwen/Qwen3-8B` 全参而 `1–2 卡` 或 `40G` 选用 `Qwen/Qwen3-4B` + LoRA、粗估单卡显存不超过物理显存的 92%。
- **结果**：**PASSED**。

### 测试 7：CPU 最小 Transformer 训练循环干跑与 `verl` 接口测试 (`tests/test_07_train_dryrun.py`)
- **运行命令**：`pytest -v tests/test_07_train_dryrun.py`
- **测试内容**：
  1. 使用 CPU 上的轻量级自回归 Transformer（`TinyCausalLM`）与 mock 沙箱采样，分别对 `sspo`、`drgrpo_pass1`、`pkpo_pass8`、`checker_reward`、`frozen` 五个臂以及 `--turns 2` 双轮反馈模式运行 2 步完整训练循环（采样 -> 检查器评估 -> DP/闭式优势 -> Dr. GRPO 裁剪损失 + KL -> 反向传播参数更新 -> 检查点与指标落盘）。
  2. 验证 `sspo/verl_patch.py` 的 PyTorch Tensor 接口 `compute_sspo_outcome_advantage` 输出形状、梯度分离与 token 掩码广播正确。
- **结果**：**PASSED**。

### 测试 8：强制中断与幂等续跑测试 (`tests/test_08_resume_fault_tolerance.py`)
- **运行命令**：`pytest -v tests/test_08_resume_fault_tolerance.py`
- **测试内容**：
  使用 `--stop-after-step 1` 在完成第 1 步后模拟进程被强杀中断，随后不改参数重新运行同一训练命令，验证：
  1. 自动从 `checkpoints/latest`（`step_0001`）恢复模型、优化器、RNG 与数据游标，只执行第 2 步。
  2. `metrics.jsonl` 精确包含 `steps = [1, 2]`（无重复行、无损坏行）。
  3. 沙箱执行缓存 `cache/exec_cache/` 命中生效。
  4. `manifest.json`、`metrics.jsonl`、`summary.json` 中不包含任何宿主机绝对路径。
- **结果**：**PASSED**。

### 测试 9：Bash 脚本静态检查与 `README.md` 命令解析测试 (`tests/test_09_scripts_and_cli.py`)
- **运行命令**：`pytest -v tests/test_09_scripts_and_cli.py`
- **测试内容**：
  1. 对 `scripts/` 下全部 11 个脚本在干净环境变量下执行 `bash -n`、`shellcheck -S error`、`--help` 与 `--dry-run`。
  2. 自动提取 `README.md` 中所有 ````bash ... ```` 代码块，在干净的子 shell 中验证语法解析并对每条 `bash scripts/...` 命令执行 `--dry-run`。
- **结果**：**PASSED**。

### 测试 10：无偏 `pass@k`、`select@k` 评测估计量与两级 Bootstrap (`tests/test_10_eval_estimators.py`)
- **运行命令**：`pytest -v tests/test_10_eval_estimators.py`
- **测试内容**：
  1. 在人造数据上验证 `pass_at_k_unbiased(n, c, k)` 与超几何解析公式 $1 - \binom{n-c}{k}/\binom{n}{k}$ 完全一致（误差 $< 10^{-15}$）。
  2. 验证 `select_at_k_estimator` 的随机子集估计（$\ge 200$ 子集）收敛到精确全子集均值，且大错误簇压制正确簇时 `is_critical_at_8` 准确识别关键题。
  3. 验证按比赛聚簇（Contest Cluster）+ 题目内重复抽样的两级配对 Bootstrap 95% 置信区间与 Holm-Bonferroni 多重检验校正。
- **结果**：**PASSED**。

---

## 3. 只能在目标 GPU 机器上验证的部分（本机未运行）

由于当前开发机为无 NVIDIA GPU 的 macOS，以下组件已编写完整代码与配置，但**未在本机实际启动，需由朋友在 Linux GPU 机器上通过 `setup_env.sh` 与 `run_all.sh --smoke-only` 验证**：

1. **CUDA 环境与 vLLM 推理引擎**：
   - `vLLM` 在 H100/A100 上的张量并行（TP=1/2）、PagedAttention KV Cache 分配以及与训练进程的显存共存。
2. **真实基座模型（`Qwen/Qwen3-8B` 与 `Qwen/Qwen3-4B`）加载与微调**：
   - 真实权重的 BF16 全参微调与 LoRA (`peft`) 前反向传播耗时与显存占用（本机使用 `TinyCausalLM` 在 CPU 上验证了完全相同的优势-损失-反向传播循环）。
3. **`verl` 分布式 Ray/FSDP 运行时集成**：
   - `sspo/verl_patch.py` 的核心优势函数已在 CPU PyTorch Tensor 上完成单元测试，但挂载到 `verl` 多卡 Ray Actor 集群的运行仅能在 Linux CUDA 环境下启动。
4. **全量 `deepmind/code_contests` 下载与 `Qwen3-8B` 辅助程序生成**：
   - 本机使用 25 道内置样本题与 `--mock` 辅助程序验证了完整流水线；目标机将通过 `prepare_data.sh` 与 `build_checkers.sh` 在真实数据集和模型上运行。
5. **Linux 内核级 `unshare -rn` 网络命名空间**：
   - macOS 不支持 Linux `unshare` 系统调用（本机测试了进程内 `socket` 封锁、`RLIMIT_AS`、`RLIMIT_CPU` 与自动探测回退逻辑；Linux 目标机上若开启非特权 userns 将自动支持 `SSPO_UNSHARE_NET=1`）。
