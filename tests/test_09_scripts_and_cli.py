"""Test 09: Bash Scripts (`bash -n`, `shellcheck`, `--help`, `--dry-run`, and README command parsing).

Section 5.9 requirement:
1. All bash scripts in `scripts/*.sh` pass `bash -n` and `shellcheck` (if installed) with 0 errors.
2. Every command in `README.md` parses cleanly in a fresh shell and every script works with `--help` and `--dry-run`.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from sspo.atomic_io import REPO_ROOT


def test_all_scripts_bash_n_and_shellcheck():
    scripts_dir = REPO_ROOT / "scripts"
    scripts = sorted(scripts_dir.glob("*.sh"))
    assert len(scripts) >= 9

    clean_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": "C.UTF-8",
    }

    for script in scripts:
        # 1. bash -n syntax check
        r_syntax = subprocess.run(
            ["bash", "-n", str(script)],
            cwd=str(REPO_ROOT),
            env=clean_env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert r_syntax.returncode == 0, f"bash -n failed on {script}: {r_syntax.stderr}"

        # 2. --help check in a clean shell
        r_help = subprocess.run(
            ["bash", str(script), "--help"],
            cwd=str(REPO_ROOT),
            env=clean_env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert r_help.returncode == 0, f"--help failed on {script}: {r_help.stderr}"
        assert "Usage:" in r_help.stdout

        # 3. --dry-run check in a clean shell
        extra = ["sspo", "0"] if script.name == "run_train.sh" else (["results/sspo_seed0"] if script.name == "run_eval.sh" else [])
        r_dry = subprocess.run(
            ["bash", str(script), *extra, "--dry-run"],
            cwd=str(REPO_ROOT),
            env=clean_env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert r_dry.returncode == 0, f"--dry-run failed on {script}: {r_dry.stderr}"
        assert "[DRY-RUN]" in r_dry.stdout

    if shutil.which("shellcheck"):
        r_sc = subprocess.run(
            ["shellcheck", "-S", "error", *(str(s) for s in scripts)],
            cwd=str(REPO_ROOT),
            env=clean_env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert r_sc.returncode == 0, f"shellcheck failed:\n{r_sc.stdout}\n{r_sc.stderr}"


def test_readme_commands_parse_and_dryrun_in_clean_shell():
    readme_path = REPO_ROOT / "README.md"
    assert readme_path.exists(), "README.md must exist"
    text = readme_path.read_text(encoding="utf-8")

    # Extract all ```bash ... ``` code blocks from README.md
    blocks = re.findall(r"```bash\n(.*?)```", text, re.S)
    assert len(blocks) >= 5, "README.md should contain bash command blocks for each stage"

    clean_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
    }

    for idx, block in enumerate(blocks):
        # Verify the block parses cleanly in a fresh bash shell (`bash -n`)
        r_parse = subprocess.run(
            ["bash", "-n", "-c", block],
            cwd=str(REPO_ROOT),
            env=clean_env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert r_parse.returncode == 0, f"README bash block #{idx} failed syntax parse:\n{block}\n{r_parse.stderr}"

        # For every `bash scripts/<name>.sh ...` line in README, execute with `--dry-run`
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("bash scripts/") and ".sh" in line:
                cmd_dry = f"{line} --dry-run"
                r_dry = subprocess.run(
                    ["bash", "-c", cmd_dry],
                    cwd=str(REPO_ROOT),
                    env=clean_env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                assert r_dry.returncode == 0, f"README command failed --dry-run: {cmd_dry}\n{r_dry.stderr}"
