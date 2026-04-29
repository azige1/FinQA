"""Watch remote DPO v2 checkpoint-50 eval and pull artifacts locally."""

from __future__ import annotations

import os
import posixpath
import stat
import time
from pathlib import Path

import paramiko


def required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(f"{name} is required")
    return value


HOST = required_env("FINGROUND_REMOTE_HOST")
PORT = int(os.environ.get("FINGROUND_REMOTE_PORT", "22"))
USER = required_env("FINGROUND_REMOTE_USER")
PASSWORD = os.environ.get("FINGROUND_REMOTE_PASSWORD", "")
REMOTE_ROOT = os.environ.get("FINGROUND_REMOTE_ROOT", "/root/FinGround-QA")
LOCAL_ROOT = Path(os.environ.get("FINGROUND_LOCAL_ROOT", r"E:\MedicalGPT\FinGround-QA"))
POLL_SECONDS = int(os.environ.get("FINGROUND_S50_PULL_POLL_SECONDS", "90"))
LOG_PATH = Path(os.environ.get("FINGROUND_S50_PULL_LOG", LOCAL_ROOT / "logs" / "dpo_v2_s50_pull.log"))

REMOTE_FILES = [
    "logs/dpo_v2_reweighted_s50_eval_pipeline.log",
    "logs/dpo_v2_reweighted_s50_eval.log",
    "results/dpo_v2_reweighted_s50_metrics.json",
    "results/dpo_v2_reweighted_s50_predictions.jsonl",
    "results/dpo_v2_reweighted_s50_scored.jsonl",
]


def log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def connect() -> paramiko.SSHClient:
    if not PASSWORD:
        raise SystemExit("FINGROUND_REMOTE_PASSWORD is required")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST,
        port=PORT,
        username=USER,
        password=PASSWORD,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    return client


def run(client: paramiko.SSHClient, command: str, timeout: int = 60) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    return stdout.channel.recv_exit_status(), out, err


def remote_file_exists(sftp: paramiko.SFTPClient, rel_path: str) -> bool:
    try:
        return stat.S_ISREG(sftp.stat(posixpath.join(REMOTE_ROOT, rel_path)).st_mode)
    except FileNotFoundError:
        return False


def download_file(sftp: paramiko.SFTPClient, rel_path: str) -> None:
    remote_path = posixpath.join(REMOTE_ROOT, rel_path)
    local_path = LOCAL_ROOT / Path(rel_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_name(local_path.name + ".pull_tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    sftp.get(remote_path, str(tmp_path))
    attrs = sftp.stat(remote_path)
    os.utime(tmp_path, (attrs.st_mtime, attrs.st_mtime))
    os.replace(tmp_path, local_path)
    log(f"pulled file {rel_path} ({attrs.st_size} bytes)")


def eval_done(client: paramiko.SSHClient) -> tuple[bool, str]:
    command = (
        f"cd {REMOTE_ROOT} || exit 2; "
        "grep -n 'DPO_V2_REWEIGHTED_S50_EVAL_DONE\\|eval finished\\|Traceback\\|RuntimeError\\|OutOfMemory' "
        "logs/dpo_v2_reweighted_s50_eval_pipeline.log logs/dpo_v2_reweighted_s50_eval.log 2>/dev/null | tail -n 80 || true; "
        "ps -eo cmd | grep -E 'run_dpo_v2_reweighted_s50|run_dpo_eval|generate' | grep -v grep || true"
    )
    rc, out, err = run(client, command)
    if rc != 0:
        return False, f"status command failed rc={rc} err={err.strip()}"
    done = "DPO_V2_REWEIGHTED_S50_EVAL_DONE rc=0" in out
    still_running = any(token in out for token in ["run_dpo_v2_reweighted_s50", "run_dpo_eval", "generate"])
    return done and not still_running, out.strip()


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log("local DPO v2 checkpoint-50 pull watcher started")
    log(f"remote={USER}@{HOST}:{PORT}:{REMOTE_ROOT}")
    log(f"local={LOCAL_ROOT}")
    while True:
        try:
            client = connect()
            try:
                done, status = eval_done(client)
                log("remote status:\n" + status[-4000:])
                if done:
                    log("DPO v2 checkpoint-50 eval is done; pulling artifacts")
                    sftp = client.open_sftp()
                    try:
                        for rel_path in REMOTE_FILES:
                            if remote_file_exists(sftp, rel_path):
                                download_file(sftp, rel_path)
                            else:
                                log(f"remote file missing, skipped {rel_path}")
                    finally:
                        sftp.close()
                    log("DPO v2 checkpoint-50 artifact pull complete")
                    return 0
            finally:
                client.close()
        except Exception as exc:
            log(f"watcher error: {exc!r}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
