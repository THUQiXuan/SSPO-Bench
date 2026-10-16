# 已知风险与排查手册 (`KNOWN_RISKS.md`)

本文档列出在目标 Linux GPU 机器上运行 SSPO v1 实验时最可能遇到的工程与方法边界风险，以及代码包中已内置的自动保护机制和手动应对办法。

---

## 1. 显存不足（CUDA Out of Memory, OOM）

### 风险原因
在同一组 GPU 上同时驻留 vLLM 推理引擎（KV Cache）和 PyTorch / FSDP / LoRA 策略优化器状态时，若 vLLM 预分配显存比例过高或长代码序列（4096 tokens）峰值激活过大，可能触发 CUDA OOM。

### 已内置保护与应对办法
1. **硬件档位自动降级**：`sspo/config.py` 会检测 `NUM_GPUS` 与单卡显存 `GPU_MEM_GB`。当单卡为 `40GB`（如 A100-40G）或总卡数 `<= 2` 时，自动切换为 `Qwen/Qwen3-4B` + `LoRA (r=32)` 并将 `micro_batch_size` 降至 `1–2`。
2. **手动强制降档命令**：如果在 `4×80GB` 跑 `Qwen3-8B` 全参微调仍因机器上有其他进程占用显存而 OOM，可通过环境变量强制使用 40GB LoRA 档位重跑（无需清理已有进度）：
   ```bash
   GPU_MEM_GB=40 bash scripts/run_all.sh
   ```
3. **调低 vLLM 显存占用率**：在 `configs/pilot.yaml` 中将 `vllm_gpu_memory_utilization` 从 `0.45` 调低至 `0.30`。

---

## 2. vLLM / `verl` / FlashAttention 二进制 Wheel 与驱动兼容问题

### 风险原因
目标机器的 CUDA 驱动、系统 `glibc` 版本或编译器工具链可能与 `flash-attn` 或特定 `vllm`/`verl` 预编译包冲突。

### 已内置保护与应对办法
1. **版本锁定与预编译优先**：`requirements-gpu.txt` 与 `scripts/setup_env.sh` 锁定了经过验证的 PyTorch `2.5.1+cu124`、`vllm==0.6.6.post1`、`verl==0.2.0`。
2. **解耦架构回退**：
   - SSPO 的核心数学模块（`sspo/dp.py`、`sspo/advantage.py`、`sspo/selector.py`、`sspo/checker.py`）是 **100% 纯 Python / NumPy** 实现，不依赖 `verl` C++ 扩展。
   - 同时提供了 `sspo/verl_patch.py`（注册到 `verl.trainer.ppo.core_algos`）和独立的 `sspo/trainer.py` 原生 PyTorch + HuggingFace/vLLM 训练器。即使目标机无法编译 `verl` 的 Ray/Megatron 附加组件，`sspo/trainer.py` 也能直接完成全部 5 个训练臂的实验。
3. **Docker 备选方案**：仓库根目录提供了自包含的 `Dockerfile`（基于 `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`）。若宿主机环境混乱，可直接用 Docker 构建运行：
   ```bash
   docker build -t sspo_v1 .
   docker run --gpus all --shm-size=32g -v "$PWD:/workspace/sspo_v1" -it sspo_v1 bash scripts/run_all.sh
   ```

---

## 3. Hugging Face 数据集与模型下载失败或限流

### 风险原因
`deepmind/code_contests` 完整 Parquet 分片较大（数十 GB），匿名高频请求容易触发 Hugging Face `429 Too Many Requests` 或连接中断。

### 已内置保护与应对办法
1. **内置固定 `HF_TOKEN` 与支持镜像 `HF_ENDPOINT`**：所有下载脚本与代码均已内置固定 `HF_TOKEN=`，并支持读取 `HF_ENDPOINT` 与断点续传。
2. **流式抽样支持（Streaming Mode）**：如果目标机磁盘紧张或不想下载完整 CodeContests 原始分片，可以给 `prepare_data.sh` 传入 `--streaming` 参数，直接按流式读取并过滤所需题数：
   ```bash
   python -m sspo.data_pipeline --dataset deepmind/code_contests --output-dir data --streaming
   ```

---

## 4. Linux 执行沙箱权限（`unshare -n` 与 `RLIMIT_AS`）

### 风险原因
1. **网络命名空间权限**：在 Ubuntu 24.04+（默认开启 AppArmor 非特权 userns 限制）或未开启 `CAP_SYS_ADMIN` 的 Docker 容器中，`unshare -rn` 会报 `Operation not permitted`。
2. **虚拟内存上限（`RLIMIT_AS = 2 GiB`）**：如果候选 Python 程序 `import numpy` 且宿主机 CPU 核心很多（如 128 核），OpenBLAS 默认分配数十个线程栈会导致虚拟地址空间超出 2 GiB 报错。

### 已内置保护与应对办法
1. **双层网络封锁**：`sspo/exec_worker.py` 在子进程启动时直接在 Python 层覆写 `socket.socket` 与 ` _socket` 模块，无条件拦截一切网络调用（无需 root 或 `CAP_SYS_ADMIN`）；同时在 `sspo/ec_common.py` 中提供可选的 `SSPO_UNSHARE_NET=1` 自动探测，若内核允许 `unshare -rn` 则叠加内核级网络命名空间隔离，不允许则平滑回退进程内封锁。
2. **线程栈显存控制**：`sspo/exec_worker.py` 在启动时强制设置 `OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`，确保 2 GiB `RLIMIT_AS` 全部留给被测程序本身。

---

## 5. 边界情形与实现说明

1. **签名簇不纯（Cluster Impurity）**：
   - **情形**：虽然已经过滤了多答案题，但如果某道题的正确候选程序因未定义行为或浮点边界落入多个不同签名簇，或某错误候选恰好在有限检查输入上与正确程序产生相同签名（$\text{sig}(y) \in \Sigma_{\mathcal{C}}$），动态规划的「簇纯度条件」不成立。
   - **处理**：`sspo/selector.py` 自动检测 `cluster_impure`，一旦不纯，`sspo/dp.py` 自动回退到对 $\binom{15}{7}=6435$ 个子集精确枚举计算 $J^-(K'\cup\{i\})-J^-(K')$（单题耗时 `< 5 ms`），并在 `metrics.jsonl` 中记录 `impurity_rate`。
2. **已知正确签名集 $\Sigma_x^+$ 的时序更新**：
   - **情形**：计算 $J^-(K) = J_S(K)\cdot\mathbb{I}[K\cap\mathcal{W}^*=\emptyset]$ 时，若把当前训练步刚采出的正确签名加入 $\Sigma_x^+$，再反过来判定同一步的错误样本属于 $\mathcal{W}^*$，会破坏留一估计量的无偏性。
   - **处理**：每道题的 $\Sigma_x^+$ 用参考正确解签名初始化，**在本步采样与优势计算前固定**；本步新发现的正确签名仅在本步优势计算完毕后并入，供后续步骤使用。
3. **全零优势题目（All-Zero Advantage Problems）**：
   - **情形**：当某题 16 条采样全对（$n_{\mathcal{C}}=16$）或全被检查器过滤（$n_{\mathcal{T}}=16$）时，所有候选的 $A_i = 0$。
   - **处理**：全 $A_i=0$ 的题目从梯度求和中跳过，同时批级归一化分母 $|B|\cdot L_{\max}$ 保持固定常数以保证无偏性，并在日志中记录 `active_fraction`。
4. **Qwen3 思考模式（Thinking Mode）干扰辅助程序生成**：
   - **情形**：Qwen3 默认开启 `<think>` 会消耗大量输出 token 导致代码未写完。
   - **处理**：在 `ec_gen.py` 与 `trainer.py` 中默认关闭思考（`enable_thinking=False`），并在代码提取器中自动剥离残留的 `<think>...</think>` 块。
