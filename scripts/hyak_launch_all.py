#!/usr/bin/env python3
"""Prepare Hyak dataset, cached execution checkers, container Python wrapper, and submit 8x2-GPU (16 GPU) Slurm jobs."""

import glob
import json
import os
import random
import subprocess
from pathlib import Path

ROOT = Path("/gscratch/simondu/xuanqi26/sspo_v1")


def setup_venv_and_models():
    venv_bin = ROOT / ".venv" / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    py_wrapper = venv_bin / "python"
    py_wrapper.write_text(
        """#!/usr/bin/env bash
set -eo pipefail
LAB=/gscratch/simondu/xuanqi26/a40-model-lab
ENVROOT=/tmp/xuanqi26-a40-lab-env-20261004
source /gscratch/simondu/xuanqi26/agent_kit/setup_env.sh >/dev/null
JID_TAG="${SLURM_JOB_ID:-local}"
mkdir -p "${ENVROOT}/vllm-cache-${JID_TAG}" "${ENVROOT}/triton-cache-${JID_TAG}" "${ENVROOT}/flashinfer-cache-${JID_TAG}"
exec apptainer exec --nv --no-home \\
  --bind /gscratch/simondu/xuanqi26:/gscratch/simondu/xuanqi26 \\
  --bind /mmfs1/gscratch/simondu/xuanqi26:/mmfs1/gscratch/simondu/xuanqi26 \\
  --bind "${ENVROOT}:${ENVROOT}" \\
  --env PYTHONPATH="/gscratch/simondu/xuanqi26/sspo_v1" \\
  --env HF_HOME="${LAB}/cache/huggingface" \\
  --env HF_HUB_OFFLINE=1 \\
  --env VLLM_CACHE_ROOT="${ENVROOT}/vllm-cache-${JID_TAG}" \\
  --env TRITON_CACHE_DIR="${ENVROOT}/triton-cache-${JID_TAG}" \\
  --env FLASHINFER_WORKSPACE_BASE="${ENVROOT}/flashinfer-cache-${JID_TAG}" \\
  --env TOKENIZERS_PARALLELISM=false \\
  --env PYTHONUNBUFFERED=1 \\
  "${LAB}/envs/sglang-0.5.20.sif" \\
  env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES "${ENVROOT}/verl/bin/python" "$@"
""",
        encoding="utf-8",
    )
    py_wrapper.chmod(0o755)

    pytest_wrapper = venv_bin / "pytest"
    pytest_wrapper.write_text(
        """#!/usr/bin/env bash
exec "$(dirname "$0")/python" -m pytest "$@"
""",
        encoding="utf-8",
    )
    pytest_wrapper.chmod(0o755)

    q8_src = Path(
        "/gscratch/simondu/xuanqi26/a40-model-lab/cache/huggingface/hub/"
        "models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    )
    q4_src = Path(
        "/gscratch/simondu/xuanqi26/a40-model-lab/cache/huggingface/hub/"
        "models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
    )
    for dst_name, src_path in [("Qwen--Qwen3-8B", q8_src), ("Qwen--Qwen3-4B", q4_src)]:
        dst = models_dir / dst_name
        if dst.is_symlink() or dst.exists():
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
        if src_path.exists() and not dst.exists():
            dst.symlink_to(src_path)

    (models_dir / "done").write_text(
        json.dumps({"status": "done", "models": ["Qwen/Qwen3-8B", "Qwen/Qwen3-4B"]}, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_data_and_checkers():
    done_f = ROOT / "data" / "done"
    if done_f.exists():
        return json.loads(done_f.read_text(encoding="utf-8"))
    cases_dir = ROOT / "cache" / "checkers" / "cases"
    art_dir = ROOT / "cache" / "checkers" / "artifacts"
    data_dir = ROOT / "data"
    cases_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    source_dirs = [
        "/gscratch/simondu/xuanqi26/exec-select-cf-20261010",
        "/gscratch/simondu/xuanqi26/exec-select-cfo-20261010",
        "/gscratch/simondu/xuanqi26/exec-select-lc-20261009",
        "/gscratch/simondu/xuanqi26/exec-select-usaco-20261010",
        "/gscratch/simondu/xuanqi26/exec-select-20261009",
        "/gscratch/simondu/xuanqi26/exec-select-n92-20261009",
    ]
    multi_kw = [
        "any valid",
        "if there are multiple",
        "print any",
        "return any",
        "multiple answers",
        "multiple solutions",
        "in any order",
    ]
    inter_kw = ["interactive problem", "flush the output", "sys.stdout.flush()"]

    valid_records = []
    seen_pids = set()

    for sdir in source_dirs:
        sdir_p = Path(sdir)
        pp_file = sdir_p / "data" / "public_problems.json"
        if not pp_file.exists():
            pp_file = sdir_p / "data" / "lcb_main.json"
        if not pp_file.exists():
            continue

        probs_raw = json.loads(pp_file.read_text(encoding="utf-8"))
        if isinstance(probs_raw, dict):
            probs_list = list(probs_raw.values())
        else:
            probs_list = probs_raw

        for row in probs_list:
            pid = str(row["pid"])
            if pid in seen_pids:
                continue
            hf_path = sdir_p / "grader_private" / "tests" / "{}.json".format(pid)
            cf_path = sdir_p / "cases_nothink" / "{}.json".format(pid)
            af_path = sdir_p / "artifacts_nothink" / "{}.json".format(pid)
            if not hf_path.exists() or not cf_path.exists():
                continue

            q_text = str(row.get("question") or row.get("prompt") or "")
            q_low = q_text.lower()
            if any(k in q_low for k in multi_kw) or any(k in q_low for k in inter_kw):
                continue

            seen_pids.add(pid)

            dst_cf = cases_dir / "{}.json".format(pid)
            if not dst_cf.exists():
                dst_cf.symlink_to(cf_path)

            if af_path.exists():
                dst_af = art_dir / "{}.json".format(pid)
                if not dst_af.exists():
                    dst_af.symlink_to(af_path)

            ref_sol = ""
            try:
                cdata = json.loads(cf_path.read_text(encoding="utf-8"))
                bf_codes = cdata.get("bf_codes") or []
                if bf_codes:
                    ref_sol = bf_codes[0]
            except Exception:
                pass

            rec = dict(row)
            rec["pid"] = pid
            rec["question"] = q_text
            rec["prompt"] = q_text
            rec["contest_id"] = str(row.get("contest_id") or "-".join(pid.split("-")[:2]))
            rec["hidden_file"] = str(hf_path)
            rec["reference_solution"] = ref_sol
            rec["timeout"] = float(row.get("timeout", 6.0))
            valid_records.append(rec)

    by_contest = {}
    for r in valid_records:
        by_contest.setdefault(r["contest_id"], []).append(r)

    ckeys = sorted(by_contest.keys())
    rng = random.Random(42)
    rng.shuffle(ckeys)

    train_recs, dev_recs, test_recs = [], [], []
    for ck in ckeys:
        items = by_contest[ck]
        if len(test_recs) < 200:
            test_recs.extend(items)
        elif len(dev_recs) < 200:
            dev_recs.extend(items)
        else:
            train_recs.extend(items)

    def write_jsonl(path, rows):
        with open(str(path), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(data_dir / "train.jsonl", train_recs)
    write_jsonl(data_dir / "train_active.jsonl", train_recs[:300])
    write_jsonl(data_dir / "dev.jsonl", dev_recs[:200])
    write_jsonl(data_dir / "test.jsonl", test_recs[:200])

    meta = {
        "status": "done",
        "total_valid": len(valid_records),
        "train": len(train_recs),
        "train_active": len(train_recs[:300]),
        "dev": len(dev_recs[:200]),
        "test": len(test_recs[:200]),
    }
    (data_dir / "done").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (ROOT / "cache" / "checkers" / "done").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def create_and_submit_slurm_jobs():
    slurm_dir = ROOT / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Cancel ONLY our own agent:sspo jobs so we can resubmit with fixed wrapper & flags
    sq = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", "xuanqi26"), "-h", "-o", "%i|%j|%T|%k"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    ).stdout.strip()
    to_cancel = []
    for line in sq.splitlines():
        parts = line.split("|")
        if len(parts) >= 4 and parts[3] == "agent:sspo":
            to_cancel.append(parts[0])
    if to_cancel:
        subprocess.run(["scancel"] + to_cancel, check=False)

    # 5 x 2-GPU (10 GPUs) + 6 x 1-GPU (6 GPUs) = 16 GPUs total under agent:sspo
    jobs_spec = [
        (
            "sspo-diag",
            1,
            4,
            "14G",
            """export SSPO_VLLM_ONLY=1
./scripts/run_diag.sh --config configs/pilot.yaml
./scripts/run_probe.sh
""",
        ),
        (
            "sspo-frozen",
            1,
            4,
            "14G",
            """export SSPO_VLLM_ONLY=1
./scripts/run_train.sh frozen 0 --config configs/pilot.yaml
./scripts/run_eval.sh results/frozen_seed0 latest --config configs/pilot.yaml
""",
        ),
        (
            "sspo-s0",
            2,
            5,
            "24G",
            """./scripts/run_train.sh sspo 0 --config configs/pilot.yaml
./scripts/run_eval.sh results/sspo_seed0 latest --config configs/pilot.yaml
""",
        ),
        (
            "sspo-drgrpo-s0",
            2,
            5,
            "24G",
            """./scripts/run_train.sh drgrpo_pass1 0 --config configs/pilot.yaml
./scripts/run_eval.sh results/drgrpo_pass1_seed0 latest --config configs/pilot.yaml
""",
        ),
        (
            "sspo-pkpo-s0",
            2,
            5,
            "24G",
            """./scripts/run_train.sh pkpo_pass8 0 --config configs/pilot.yaml
./scripts/run_eval.sh results/pkpo_pass8_seed0 latest --config configs/pilot.yaml
""",
        ),
        (
            "sspo-s1",
            2,
            5,
            "24G",
            """./scripts/run_train.sh sspo 1 --config configs/pilot.yaml
./scripts/run_eval.sh results/sspo_seed1 latest --config configs/pilot.yaml
""",
        ),
        (
            "sspo-drgrpo-s1",
            2,
            5,
            "24G",
            """./scripts/run_train.sh drgrpo_pass1 1 --config configs/pilot.yaml
./scripts/run_eval.sh results/drgrpo_pass1_seed1 latest --config configs/pilot.yaml
""",
        ),
        (
            "sspo-chk-s0",
            1,
            4,
            "14G",
            """./scripts/run_train.sh checker_reward 0 --config configs/pilot.yaml
./scripts/run_eval.sh results/checker_reward_seed0 latest --config configs/pilot.yaml
""",
        ),
        (
            "sspo-pkpo-s1",
            1,
            4,
            "14G",
            """./scripts/run_train.sh pkpo_pass8 1 --config configs/pilot.yaml
./scripts/run_eval.sh results/pkpo_pass8_seed1 latest --config configs/pilot.yaml
""",
        ),
        (
            "sspo-turns2-s0",
            1,
            4,
            "14G",
            """./scripts/run_train.sh sspo 0 --turns 2 --run-dir results/sspo_turns2_seed0 --config configs/pilot.yaml
./scripts/run_eval.sh results/sspo_turns2_seed0 latest --config configs/pilot.yaml
""",
        ),
        (
            "sspo-s2",
            1,
            4,
            "14G",
            """./scripts/run_train.sh sspo 2 --config configs/pilot.yaml
./scripts/run_eval.sh results/sspo_seed2 latest --config configs/pilot.yaml
""",
        ),
    ]

    submitted = []
    for job_name, n_gpus, n_cpus, mem_str, body_cmds in jobs_spec:
        script_path = slurm_dir / "{}.sbatch".format(job_name)
        script_content = """#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --account=ckpt-simondu
#SBATCH --partition=ckpt-all
#SBATCH --constraint="rtx6k|a40|l40|l40s|a100|h200"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node={n_gpus}
#SBATCH --cpus-per-task={n_cpus}
#SBATCH --mem={mem_str}
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --comment=agent:sspo
#SBATCH --output=/gscratch/simondu/xuanqi26/sspo_v1/logs/{job_name}.%j.out
#SBATCH --error=/gscratch/simondu/xuanqi26/sspo_v1/logs/{job_name}.%j.err

set -eo pipefail
cd /gscratch/simondu/xuanqi26/sspo_v1
export HF_TOKEN=
export SSPO_NCPU={n_cpus}
echo "[$(date -Is)] Starting {job_name} on $(hostname) GPUs=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L || true

{body_cmds}
echo "[$(date -Is)] Finished {job_name}"
""".format(
            job_name=job_name,
            n_gpus=n_gpus,
            n_cpus=n_cpus,
            mem_str=mem_str,
            body_cmds=body_cmds,
        )
        script_path.write_text(script_content, encoding="utf-8")
        script_path.chmod(0o755)

        budget_chk = subprocess.run(
            ["/gscratch/simondu/xuanqi26/agent_kit/gpu_budget.sh", str(n_gpus), "agent:sspo", "16"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )
        res = subprocess.run(
            ["sbatch", "--parsable", str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True,
        )
        jid = res.stdout.strip().split(";")[0]
        submitted.append(
            {
                "job_name": job_name,
                "job_id": jid,
                "gpus": n_gpus,
                "status": "submitted",
            }
        )
    return submitted


if __name__ == "__main__":
    setup_venv_and_models()
    data_meta = prepare_data_and_checkers()
    jobs = create_and_submit_slurm_jobs()
    print(json.dumps({"data": data_meta, "jobs": jobs}, indent=2, ensure_ascii=False))
