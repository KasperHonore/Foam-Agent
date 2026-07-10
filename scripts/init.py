#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Foam-Agent one-command init: nothing to onboarding hand-off in one command.

Validates prerequisites (git, git-lfs, Docker daemon) BEFORE cloning and
prints the exact fix for anything missing, clones the latest release tag
(falling back to the default branch while no releases exist, and saying so),
verifies the clone's FAISS indices are real LFS content, then prints the one
line to open the clone in your AI agent CLI and let the foam-onboard skill
finish setup conversationally. Read-only until the clone step; it never
touches an existing installation. Deterministic and key-free like the doctor.

Usage -- from nothing, no clone needed:

    uv run https://raw.githubusercontent.com/KasperHonore/Foam-Agent/main/scripts/init.py

  or with plain Python (3.10+, standard library only):

    curl -sLO https://raw.githubusercontent.com/KasperHonore/Foam-Agent/main/scripts/init.py
    python init.py

(`uvx --from git+...` is deliberately not the documented form: it would build
the whole server package and its ML dependency stack just to run this
stdlib-only script. `uv run` fetches and runs the single file.)

Options:
    --target DIR    clone destination (default: ./Foam-Agent)
    --repo URL      repository to clone (default: the official repo)

Exit code 0 when the clone is ready (or one already exists), 1 otherwise.
The manual path remains in the README Quick start; the health check inside
the clone is scripts/doctor.py.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/KasperHonore/Foam-Agent.git"
AGENT_CLIS = ("claude", "cursor", "codex", "opencode")  # the supported harnesses


def emit(line: str = "") -> None:
    """Print ASCII no matter what the console or the externals hand us."""
    sys.stdout.write(line.encode("ascii", "replace").decode("ascii") + "\n")


def report(name: str, ok: bool, detail: str, fix: str | None = None, warn: bool = False) -> bool:
    """One doctor-style check line: status, what was seen, the exact fix."""
    mark = "OK  " if ok else ("WARN" if warn else "FAIL")
    emit(f"[{mark}] {name}: {detail}")
    if fix and not ok:
        emit(f"       fix: {fix}")
    return ok


def _run(cmd: list[str], timeout: int = 60, cwd: str | None = None) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def _docker() -> str | None:
    """Find the docker CLI, including the Docker Desktop path Windows shells miss."""
    path = shutil.which("docker")
    if path:
        return path
    # Test seam: lets the CLI-seam tests simulate a machine without Docker
    # Desktop (the Windows loader restores ProgramFiles in child processes,
    # so the fallback cannot be steered through the environment block).
    if os.environ.get("FOAM_INIT_DISABLE_DOCKER_FALLBACK"):
        return None
    if sys.platform == "win32":
        candidate = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Docker" / "Docker" / "resources" / "bin" / "docker.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def check_prerequisites() -> tuple[str | None, bool]:
    """Validate everything the clone-and-onboard path needs, before cloning.

    Returns (git_path, all_ok). Every check prints doctor-style; all failures
    are reported in one run so the user fixes them in one round-trip.
    """
    ok = True

    git = shutil.which("git")
    if git:
        report("git", True, git)
    else:
        ok = report("git", False, "git not found on PATH",
                    "install git (https://git-scm.com/downloads) and re-run") and ok

    if git:
        code, out = _run([git, "lfs", "version"])
        if code == 0:
            report("git-lfs", True, out.splitlines()[0] if out else "installed")
        else:
            ok = report("git-lfs", False, "git-lfs is not installed",
                        "install git-lfs (https://git-lfs.com) and re-run -- "
                        "the FAISS tutorial indices ship via LFS") and ok

    docker = _docker()
    if docker is None:
        ok = report("Docker CLI", False, "docker not found on PATH",
                    "install Docker Desktop (https://docker.com) and start it") and ok
    else:
        code, out = _run([docker, "version", "--format", "{{.Server.Version}}"])
        if code != 0:
            ok = report("Docker daemon", False, "docker CLI found but the daemon is not responding",
                        "start Docker Desktop (or the docker service) and re-run") and ok
        else:
            version = out.splitlines()[-1] if out else "unknown"
            report("Docker daemon", True, f"server version {version}")

    return git, ok


def is_clone(path: Path) -> bool:
    """Does this directory already hold a Foam-Agent clone?"""
    return (path / "scripts" / "doctor.py").is_file()


def latest_release_tag(ls_remote_output: str) -> str | None:
    """The highest version-shaped tag (v1.10 > v1.2: numeric, not lexicographic)."""
    versions: list[tuple[tuple[int, ...], str]] = []
    for line in ls_remote_output.splitlines():
        tag = line.rpartition("refs/tags/")[2].strip()
        match = re.fullmatch(r"v?(\d+(?:\.\d+)*)", tag)
        if match:
            versions.append((tuple(int(p) for p in match.group(1).split(".")), tag))
    return max(versions)[1] if versions else None


def clone(git: str, repo_url: str, target: Path) -> bool:
    """Clone the latest release tag, or the default branch when none exist."""
    code, out = _run([git, "ls-remote", "--tags", "--refs", repo_url], timeout=60)
    if code != 0:
        report("release lookup", False, f"could not reach {repo_url}",
               "check your network connection and re-run")
        return False

    tag = latest_release_tag(out)
    if tag:
        report("latest release", True, tag)
        clone_cmd = [git, "clone", "--branch", tag, repo_url, str(target)]
        what = f"release {tag}"
    else:
        report("releases", True, "no release tags published yet -- cloning the default branch instead")
        clone_cmd = [git, "clone", repo_url, str(target)]
        what = "the default branch"

    emit(f"Cloning {what} into {target} ...")
    code, out = _run(clone_cmd, timeout=1800)
    if code != 0:
        report("clone", False, "git clone failed",
               "read the git output below, fix the cause, and re-run")
        for line in out.splitlines():
            emit(f"       {line}")
        return False
    report("clone", True, str(target))
    return True


def _faiss_state(target: Path) -> tuple[list[Path], list[Path]]:
    """All .faiss files in the clone, and the ones that are still LFS pointers."""
    files = list((target / "database" / "faiss").rglob("*.faiss"))
    pointers = [f for f in files if f.stat().st_size < 10_000]
    return files, pointers


def validate_faiss(git: str, target: Path) -> None:
    """The FAISS tutorial indices ship via LFS; make sure the clone got real ones.

    Warn-only, like the doctor's check: the Docker image ships the indices
    baked in, so onboarding works either way -- but say exactly how to fix it.
    """
    files, pointers = _faiss_state(target)
    if not files or pointers:
        emit("LFS content not materialized yet -- running git lfs pull ...")
        _run([git, "lfs", "pull"], timeout=1800, cwd=str(target))
        files, pointers = _faiss_state(target)
    if not files or pointers:
        detail = (f"{len(pointers)}/{len(files)} files are LFS pointers, not real indices"
                  if files else "no .faiss files found under database/faiss")
        report("FAISS indices", False, detail,
               f"cd {target} && git lfs install && git lfs pull", warn=True)
        emit("       (needed only for source builds -- the Docker image ships them baked in)")
    else:
        report("FAISS indices", True, f"{len(files)} real index files")


def hand_off(target: Path) -> None:
    """Point at the next step: open the clone in an agent CLI, say 'onboard me'.

    Onboarding itself (health check, warm-up, demo, tour) belongs to the
    foam-onboard skill inside the clone -- the init only hands over to it.
    """
    found = [cli for cli in AGENT_CLIS if shutil.which(cli)]
    emit()
    if found:
        emit(f"Detected agent CLI{'s' if len(found) > 1 else ''}: {', '.join(found)}")
        emit("Next -- finish setup conversationally:")
        if target != Path.cwd():
            emit(f"  cd {target}")
        emit(f"  {found[0]}")
        emit("  > onboard me")
    else:
        emit("No supported agent CLI found on PATH (claude / cursor / codex / opencode).")
        emit(f"Install one, open {target} in it, and say: onboard me")
    emit()
    emit("Manual fallback: the Quick start in the README. Health check without an agent:")
    emit(f"  python scripts/doctor.py   (from inside {target})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Foam-Agent one-command init: clone, validate, hand off to onboarding.")
    parser.add_argument("--target", default=None, help="clone destination (default: ./Foam-Agent)")
    parser.add_argument("--repo", default=REPO_URL, help="repository to clone (default: the official repo)")
    args = parser.parse_args()

    emit("Foam-Agent init")
    emit("=" * 15)

    target = Path(args.target or "Foam-Agent").resolve()

    # Idempotence guard: an existing installation is never touched.
    if args.target is None and is_clone(Path.cwd()):
        emit("You are already inside a Foam-Agent clone -- nothing to do.")
        emit("To update it: git pull   (your runs, prompts and settings are never touched)")
        hand_off(Path.cwd())
        return 0
    if is_clone(target):
        emit(f"{target} already contains a Foam-Agent clone -- leaving it untouched.")
        emit(f"To update it: cd {target} && git pull")
        hand_off(target)
        return 0
    if target.exists() and any(target.iterdir()):
        emit(f"Target directory {target} exists, is not empty, and does not look like a Foam-Agent clone.")
        emit("Pick another destination:  --target <dir>")
        return 1

    git, prereqs_ok = check_prerequisites()
    if not prereqs_ok:
        emit()
        emit("Not ready yet -- nothing was cloned. Run the fixes above, then re-run this script.")
        emit("Manual fallback: the Quick start in the README covers the same steps by hand.")
        return 1
    assert git is not None  # a failed git check exits above

    if not clone(git, args.repo, target):
        return 1

    validate_faiss(git, target)

    emit()
    emit(f"Foam-Agent is ready at {target}.")
    hand_off(target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
