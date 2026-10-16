"""Encrypted Hidden Test Vault for SSPO-Bench (`benchmark/hidden_test_vault.py`).

Protects the held-out test split (`data/test.jsonl`) against direct inspection or
data leakage by coding agents:
- On disk, `data/test.jsonl` contains ONLY `pid`, `contest_id`, `question`, and `public`
  examples (`"hidden": []`, `"reference_solution": null`).
- The true hidden test cases (`40` per problem) are compressed and encrypted in
  `data/test_hidden.enc` using HMAC-SHA256 authenticated stream encryption.
- During official test evaluation (`eval_split == "test"`), `inject_hidden_tests_in_memory`
  decrypts `data/test_hidden.enc` strictly in memory without ever writing plaintext
  hidden tests to disk.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import zlib
from pathlib import Path
from typing import Any, Dict, List

from sspo.atomic_io import REPO_ROOT, atomic_write_jsonl, read_jsonl

DEFAULT_VAULT_KEY = "sspo-bench-v1-public-eval-key-2026"
VAULT_MAGIC = b"SSPOVAULT1"


def _derive_keys(secret: str, salt: bytes) -> tuple[bytes, bytes]:
    master = hashlib.pbkdf2_hmac("sha256", secret.encode("utf-8"), salt, 10000, dklen=64)
    return master[:32], master[32:]


def _xor_keystream(data: bytes, enc_key: bytes, nonce: bytes) -> bytes:
    out = bytearray(len(data))
    block_size = 32
    n_blocks = (len(data) + block_size - 1) // block_size
    for blk in range(n_blocks):
        counter = blk.to_bytes(8, "big")
        ks = hmac.new(enc_key, nonce + counter, hashlib.sha256).digest()
        start = blk * block_size
        end = min(len(data), start + block_size)
        for i in range(start, end):
            out[i] = data[i] ^ ks[i - start]
    return bytes(out)


def encrypt_payload(payload_obj: Dict[str, Any], secret: str = DEFAULT_VAULT_KEY) -> bytes:
    """Compress and encrypt a JSON-serializable dictionary with HMAC-SHA256 authentication."""
    raw_json = json.dumps(payload_obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    compressed = zlib.compress(raw_json, level=9)
    # Deterministic salt from content hash so builds are reproducible
    salt = hashlib.sha256(b"sspo-salt:" + compressed).digest()[:16]
    nonce = hashlib.sha256(b"sspo-nonce:" + compressed).digest()[:16]
    enc_key, mac_key = _derive_keys(secret, salt)
    ciphertext = _xor_keystream(compressed, enc_key, nonce)
    mac = hmac.new(mac_key, VAULT_MAGIC + salt + nonce + ciphertext, hashlib.sha256).digest()
    blob = VAULT_MAGIC + salt + nonce + mac + ciphertext
    return base64.b64encode(blob)


def decrypt_payload(encoded_blob: bytes, secret: str = DEFAULT_VAULT_KEY) -> Dict[str, Any]:
    """Authenticate, decrypt, and decompress a vault payload in memory."""
    blob = base64.b64decode(encoded_blob)
    magic_len = len(VAULT_MAGIC)
    if len(blob) < magic_len + 16 + 16 + 32 or blob[:magic_len] != VAULT_MAGIC:
        raise ValueError("Invalid or corrupted hidden test vault header.")
    pos = magic_len
    salt = blob[pos : pos + 16]
    pos += 16
    nonce = blob[pos : pos + 16]
    pos += 16
    expected_mac = blob[pos : pos + 32]
    pos += 32
    ciphertext = blob[pos:]

    enc_key, mac_key = _derive_keys(secret, salt)
    actual_mac = hmac.new(mac_key, VAULT_MAGIC + salt + nonce + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(expected_mac, actual_mac):
        raise PermissionError("Hidden test vault authentication failed (invalid SSPO_BENCH_KEY).")

    compressed = _xor_keystream(ciphertext, enc_key, nonce)
    raw_json = zlib.decompress(compressed)
    return json.loads(raw_json.decode("utf-8"))


def Seal_test_split(
    test_jsonl_path: str | Path = REPO_ROOT / "data" / "test.jsonl",
    vault_path: str | Path = REPO_ROOT / "data" / "test_hidden.enc",
    secret: str | None = None,
) -> Dict[str, Any]:
    """Extract hidden tests from `test.jsonl` into `test_hidden.enc` and strip `test.jsonl`."""
    key = secret or os.environ.get("SSPO_BENCH_KEY", DEFAULT_VAULT_KEY)
    t_path = Path(test_jsonl_path)
    v_path = Path(vault_path)
    rows = read_jsonl(t_path)

    hidden_map: Dict[str, List[List[str]]] = {}
    public_rows: List[Dict[str, Any]] = []
    for r in rows:
        pid = str(r["pid"])
        hid = r.get("hidden") or []
        if hid:
            hidden_map[pid] = [[str(i), str(o)] for i, o in hid]
        stripped = dict(r)
        stripped["hidden"] = []
        stripped["reference_solution"] = None
        stripped.pop("reference_solutions", None)
        public_rows.append(stripped)

    if hidden_map:
        v_path.parent.mkdir(parents=True, exist_ok=True)
        encrypted_bytes = encrypt_payload({"hidden_by_pid": hidden_map}, secret=key)
        v_path.write_bytes(encrypted_bytes)
        atomic_write_jsonl(t_path, public_rows)

    return {
        "n_problems": len(public_rows),
        "n_sealed_in_vault": len(hidden_map),
        "vault_file": str(v_path),
        "test_jsonl_file": str(t_path),
    }


def inject_hidden_tests_in_memory(
    problems: List[Dict[str, Any]],
    vault_path: str | Path = REPO_ROOT / "data" / "test_hidden.enc",
    secret: str | None = None,
) -> List[Dict[str, Any]]:
    """Return a copy of `problems` with `hidden` tests populated in-memory from `test_hidden.enc`."""
    need_decrypt = any(not (p.get("hidden")) for p in problems)
    v_path = Path(vault_path)
    if not need_decrypt or not v_path.exists():
        return problems

    key = secret or os.environ.get("SSPO_BENCH_KEY", DEFAULT_VAULT_KEY)
    payload = decrypt_payload(v_path.read_bytes(), secret=key)
    hidden_by_pid = payload.get("hidden_by_pid", {})

    enriched: List[Dict[str, Any]] = []
    for p in problems:
        cp = dict(p)
        pid = str(cp["pid"])
        if not cp.get("hidden") and pid in hidden_by_pid:
            cp["hidden"] = hidden_by_pid[pid]
        enriched.append(cp)
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage SSPO-Bench Encrypted Hidden Test Vault.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_seal = sub.add_parser("seal", help="Encrypt hidden tests from test.jsonl into test_hidden.enc and strip test.jsonl")
    p_seal.add_argument("--test-file", type=str, default=str(REPO_ROOT / "data" / "test.jsonl"))
    p_seal.add_argument("--vault-file", type=str, default=str(REPO_ROOT / "data" / "test_hidden.enc"))

    p_check = sub.add_parser("verify-vault", help="Verify test_hidden.enc can be decrypted in memory")
    p_check.add_argument("--test-file", type=str, default=str(REPO_ROOT / "data" / "test.jsonl"))
    p_check.add_argument("--vault-file", type=str, default=str(REPO_ROOT / "data" / "test_hidden.enc"))

    args = parser.parse_args()
    if args.cmd == "seal":
        res = Seal_test_split(args.test_file, args.vault_file)
        print(json.dumps(res, indent=2))
    elif args.cmd == "verify-vault":
        rows = read_jsonl(args.test_file)
        enriched = inject_hidden_tests_in_memory(rows, args.vault_file)
        n_with_hid = sum(1 for r in enriched if len(r.get("hidden") or []) >= 5)
        print(json.dumps({"total": len(enriched), "verified_with_hidden_tests": n_with_hid}, indent=2))


if __name__ == "__main__":
    main()
