#!/usr/bin/env python3
"""Foam-Agent doctor: validate the local setup without needing an AI agent.

Read-only preflight — it never changes anything, it tells you exactly what to
run. The agent-driven equivalent is the `foam-setup` skill
(agents/skills/foam-setup/SKILL.md); both check the same things.

Usage:
    python scripts/doctor.py
    python scripts/doctor.py --json    # machine-readable output

Exit code 0 when the MCP server is reachable, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MCP_URL = "http://localhost:7860/mcp"

PULL_CMDS = (
    "docker pull ghcr.io/kasperhonore/foamagent:latest\n"
    "  docker tag ghcr.io/kasperhonore/foamagent:latest foamagent:latest"
)
RUN_CMD = (
    "docker run -d --name foamagent-mcp --restart unless-stopped -p 7860:7860 \\\n"
    "    foamagent:latest python -m src.mcp.fastmcp_server --transport http --host 0.0.0.0 --port 7860"
)


class Check:
    def __init__(self, name: str, ok: bool, detail: str, fix: str | None = None, warn: bool = False):
        self.name, self.ok, self.detail, self.fix, self.warn = name, ok, detail, fix, warn


def _run(cmd: list[str], timeout: int = 15) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def _docker() -> str | None:
    """Find the docker CLI, including the Docker Desktop path Windows shells miss."""
    path = shutil.which("docker")
    if path:
        return path
    if sys.platform == "win32":
        candidate = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Docker" / "Docker" / "resources" / "bin" / "docker.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def check_git_lfs() -> Check:
    """FAISS indices must be real files, not LFS pointers (only matters for source builds)."""
    faiss_dir = REPO / "database" / "faiss"
    files = list(faiss_dir.rglob("*.faiss"))
    if not files:
        return Check("git-lfs FAISS indices", False, "no .faiss files found under database/faiss",
                     "git lfs install && git lfs pull", warn=True)
    pointers = [f for f in files if f.stat().st_size < 10_000]
    if pointers:
        return Check("git-lfs FAISS indices", False,
                     f"{len(pointers)}/{len(files)} files are LFS pointers, not real indices",
                     "git lfs install && git lfs pull", warn=True)
    return Check("git-lfs FAISS indices", True, f"{len(files)} real index files (needed only for source builds)")


def check_docker_daemon(docker: str | None) -> Check:
    if docker is None:
        return Check("Docker CLI", False, "docker not found on PATH",
                     "install Docker Desktop (https://docker.com) and start it")
    code, out = _run([docker, "version", "--format", "{{.Server.Version}}"])
    if code != 0:
        return Check("Docker daemon", False, "docker CLI found but the daemon is not responding",
                     "start Docker Desktop (or the docker service) and re-run")
    return Check("Docker daemon", True, f"server version {out.splitlines()[-1]}")


def check_image(docker: str) -> Check:
    code, out = _run([docker, "images", "foamagent", "--format", "{{.Repository}}:{{.Tag}} {{.Size}}"])
    if code != 0 or not out:
        return Check("foamagent image", False, "image foamagent:latest not present", PULL_CMDS)
    return Check("foamagent image", True, out.splitlines()[0])


def check_container(docker: str) -> Check:
    code, out = _run([docker, "ps", "-a", "--filter", "name=foamagent-mcp", "--format", "{{.Status}}"])
    if code != 0 or not out:
        return Check("foamagent-mcp container", False, "container does not exist", RUN_CMD)
    status = out.splitlines()[0]
    if not status.startswith("Up"):
        return Check("foamagent-mcp container", False, f"container exists but is not running ({status})",
                     "docker start foamagent-mcp")
    return Check("foamagent-mcp container", True, status)


def check_endpoint() -> Check:
    """Any HTTP response from the MCP endpoint means the server is listening."""
    req = urllib.request.Request(MCP_URL, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5):
            pass
        return Check("MCP endpoint", True, f"{MCP_URL} responding")
    except urllib.error.HTTPError:
        # Streamable-HTTP MCP answers GET with 405/406 — still proof of life
        return Check("MCP endpoint", True, f"{MCP_URL} responding")
    except OSError as exc:
        return Check("MCP endpoint", False, f"{MCP_URL} unreachable ({exc})",
                     "start the container (see checks above), wait ~15 s, re-run")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    checks: list[Check] = [check_git_lfs()]
    docker = _docker()
    daemon = check_docker_daemon(docker)
    checks.append(daemon)
    if daemon.ok and docker:
        image = check_image(docker)
        checks.append(image)
        if image.ok:
            checks.append(check_container(docker))
    checks.append(check_endpoint())

    healthy = checks[-1].ok

    if args.json:
        print(json.dumps({"healthy": healthy, "checks": [vars(c) for c in checks]}, indent=2))
        return 0 if healthy else 1

    print("Foam-Agent doctor\n" + "=" * 17)
    for c in checks:
        mark = "OK  " if c.ok else ("WARN" if c.warn else "FAIL")
        print(f"[{mark}] {c.name}: {c.detail}")
        if c.fix and not c.ok:
            print(f"       fix: {c.fix}")
    print()
    if healthy:
        print("Server is up. Open this repo in your AI agent (claude / cursor / codex / opencode)")
        print("and ask for a simulation, e.g.:  /foam Simulate lid-driven cavity flow at Re=1000")
        print("Note: the FIRST retrieval call downloads a ~1.2 GB embedding model -- minutes of")
        print("silence there is normal, not a hang.")
    else:
        print("Not ready yet. Run the fixes above in order, or open this repo in your AI agent")
        print("and say: \"set up foam-agent for me\" — the foam-setup skill does this interactively.")
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
