"""
checkpoint.py -- resume-safe GitHub checkpointing for CURE.

Force-adds results and logs (but never results/dryrun) and pushes to the branch
every PUSH_INTERVAL_S seconds and after each unit. A released VM restarts from the
last pushed, non-corrupt, de-duplicated results.

The remote token is read from the environment (Github_Classic_Token); it is never
written into a tracked file or a log.
"""

import logging
import subprocess
import threading
import time

import config_cure as C

log = logging.getLogger("cure.checkpoint")

GIT_BRANCH = "main"
PUSH_INTERVAL_S = 15 * 60
REL_RESULTS = "Code/CURE/results"
REL_LOGS = "Code/CURE/logs"


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(C.REPO), capture_output=True, text=True)


def git_configure(token: str, remote_url: str) -> None:
    """Point origin at a token URL. The token is only used here, never written out."""
    _run(["git", "config", "user.name", "CURE Runner"])
    _run(["git", "config", "user.email", "koushikdeb2009@gmail.com"])
    _run(["git", "config", "pull.rebase", "true"])
    if token and remote_url:
        auth = remote_url.replace("https://", f"https://{token}@")
        _run(["git", "remote", "set-url", "origin", auth])


def push_checkpoint(message: str) -> bool:
    # Push only real results (parquets, STATUS, DONE). Never push logs, and never push
    # the dry-run test results -- per the run contract.
    _run(["git", "add", "-f", REL_RESULTS])
    _run(["git", "reset", "-q", "--", REL_RESULTS + "/dryrun", REL_RESULTS + "/quarantine"])
    if not _run(["git", "status", "--porcelain", "--", REL_RESULTS]).stdout.strip():
        return False
    _run(["git", "commit", "-q", "-m", message])
    _run(["git", "pull", "--rebase", "-q", "origin", GIT_BRANCH])
    ok = _run(["git", "push", "-q", "origin", GIT_BRANCH]).returncode == 0
    if not ok:
        _run(["git", "pull", "--rebase", "-q", "origin", GIT_BRANCH])
        ok = _run(["git", "push", "-q", "origin", GIT_BRANCH]).returncode == 0
    log.info("checkpoint push: %s", "ok" if ok else "no-op/failed")
    return ok


class CheckpointPusher(threading.Thread):
    def __init__(self, interval_s: int = PUSH_INTERVAL_S):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                push_checkpoint(f"cure-results: checkpoint {ts}")
            except Exception as exc:
                log.warning("pusher error: %s", exc)

    def stop_and_flush(self, message: str) -> None:
        self._stop.set()
        try:
            push_checkpoint(message)
        except Exception as exc:
            log.warning("final push failed: %s", exc)
