#!/usr/bin/env python3
"""Foam-Agent doctor: validate the local setup without needing an AI agent.

Read-only preflight — it never changes anything, it tells you exactly what to
run. The agent-driven equivalent is the `foam-setup` skill
(agents/skills/foam-setup/SKILL.md); both check the same things.

Usage:
    python scripts/doctor.py
    python scripts/doctor.py --json    # machine-readable output

Green is the install contract (spec #29): exit code 0 when every hard check
passes — warnings never gate. Hard checks include the version lockstep
(clone pyproject.toml vs the image's baked /etc/foamagent-version stamp,
issue #48) and the runs/ mount source matching this clone (issue #26).

The --json output also carries `onboardingNeeded`: true only on first-run
signals — no seeded config/user.yml AND no rows in runs/ledger.md. The
onboarding skill branches on it.

Env overrides (config knobs, and the faked boundary in the CLI tests):
    FOAMAGENT_MCP_URL       MCP endpoint  (default http://localhost:7860/mcp)
    FOAMAGENT_RELEASES_URL  release feed  (default the GitHub releases API)

Stays key-free, stdlib-only, ASCII-output.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import ledger  # noqa: E402  (stdlib-only, ships with the clone)

# Env overrides double as config knobs and as the faked boundary in the CLI
# tests (tests/test_doctor_unit.py) -- both accept file:// URLs.
MCP_URL = os.environ.get("FOAMAGENT_MCP_URL", "http://localhost:7860/mcp")
RELEASES_URL = os.environ.get(
    "FOAMAGENT_RELEASES_URL",
    "https://api.github.com/repos/KasperHonore/Foam-Agent/releases?per_page=1",
)

PULL_CMDS = (
    "docker pull ghcr.io/kasperhonore/foamagent:latest\n"
    "  docker tag ghcr.io/kasperhonore/foamagent:latest foamagent:latest"
)
RUNS_MOUNT_DST = "/home/openfoam/Foam-Agent/runs"
RUN_CMD = (
    "docker run -d --name foamagent-mcp --restart unless-stopped -p 7860:7860 \\\n"
    f'    -v "{REPO / "runs"}:{RUNS_MOUNT_DST}" \\\n'
    "    foamagent:latest python -m src.mcp.fastmcp_server --transport http --host 0.0.0.0 --port 7860"
)

# Version stamp contract (issue #48): the single source of truth is
# [project].version in pyproject.toml. docker/Dockerfile bakes that value
# into the image at build time as /etc/foamagent-version, so the doctor can
# compare clone vs image with one cheap `docker exec`. The release workflow
# inherits the stamp for free: any image built from a tagged checkout
# carries that tag's pyproject version.
VERSION_STAMP_PATH = "/etc/foamagent-version"
LOCKSTEP_FIX = (
    "update both sides to the same version, then recreate the container:\n"
    "  git pull\n"
    f"  {PULL_CMDS}\n"
    f"  docker rm -f foamagent-mcp && {RUN_CMD}"
)


def product_version() -> str | None:
    """The clone's version, read from the packaging manifest."""
    try:
        text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else None


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


def check_version_lockstep(docker: str) -> Check:
    """Clone and image must carry the same version stamp (issue #48).

    Skills (the clone) and the container image travel separately; silent
    drift between them is the version-mismatch bug class the product exists
    to prevent, so a mismatch is a hard failure.
    """
    product = product_version()
    if product is None:
        return Check("version lockstep", False,
                     "could not read the product version from pyproject.toml", warn=True)
    code, out = _run([docker, "exec", "foamagent-mcp", "cat", VERSION_STAMP_PATH])
    # _run concatenates stdout+stderr; the stamp is stdout's (single) first line
    stamp = out.strip().splitlines()[0].strip() if out.strip() else ""
    if code != 0 or not stamp:
        return Check("version lockstep", False,
                     f"image carries no version stamp ({VERSION_STAMP_PATH} missing -- "
                     "built before the stamp mechanism)",
                     PULL_CMDS, warn=True)
    if stamp != product:
        return Check("version lockstep", False,
                     f"clone is {product} but the running image is {stamp} -- "
                     "skills and container have drifted out of lockstep",
                     LOCKSTEP_FIX)
    return Check("version lockstep", True, f"clone {product} == image {stamp}")


def _canonical_mount_path(raw: str) -> str | None:
    """Best-effort canonical form of a mount path for comparison.

    Docker reports the same host directory in several spellings (backslashes,
    \\\\?\\ long-path prefixes, Docker Desktop's /run/desktop/mnt/host/c/...,
    WSL's /mnt/c/...); fold them all to one form. Drive-letter paths are
    casefolded (Windows filesystems are case-insensitive). Returns None for
    values that are not absolute paths at all (e.g. named volumes) -- those
    are not comparable.
    """
    path = raw.strip().replace("\\", "/")
    for prefix in ("//?/", "//./"):
        if path.startswith(prefix):
            path = path[len(prefix):]
    m = re.match(r"^/(?:run/desktop/mnt/host|host_mnt|mnt/host|mnt)/([A-Za-z])(/.*)?$", path)
    if m:
        path = f"{m.group(1)}:{m.group(2) or '/'}"
    if len(path) > 1:
        path = path.rstrip("/")
    if re.match(r"^[A-Za-z]:(/|$)", path):
        return path.casefold()
    if path.startswith("/"):
        return path
    return None


def check_runs_mount(docker: str) -> Check:
    """runs/ must be bind-mounted from THIS clone's runs directory.

    Without the mount, results are lost on container recreation; mounted
    from a different clone (issue #26), every run lands in the wrong
    repository -- so a wrong source is a hard failure, while an
    uncomparable source is only a warning.
    """
    recreate = f"docker rm -f foamagent-mcp && {RUN_CMD}"
    code, out = _run([docker, "inspect", "foamagent-mcp", "--format", "{{json .Mounts}}"])
    try:
        # _run concatenates stdout+stderr; the JSON is stdout's first line
        mounts = json.loads(out.strip().splitlines()[0]) if out.strip() else []
    except json.JSONDecodeError:
        mounts = None
    if code != 0 or not isinstance(mounts, list):
        return Check("runs/ mount", False, "could not inspect container mounts", warn=True)
    mount = next((m for m in mounts if m.get("Destination") == RUNS_MOUNT_DST), None)
    if mount is None:
        return Check("runs/ mount", False,
                     "container has no runs/ bind mount -- simulation results will be "
                     "lost when the container is recreated",
                     recreate, warn=True)
    expected_runs = str((REPO / "runs").resolve())
    source = str(mount.get("Source") or "")
    canonical_source = _canonical_mount_path(source)
    if canonical_source is None:
        return Check("runs/ mount", False,
                     f"could not compare the mount source ({source or 'unknown'}) "
                     f"against this clone's runs directory ({expected_runs})",
                     warn=True)
    if canonical_source != _canonical_mount_path(expected_runs):
        return Check("runs/ mount", False,
                     f"container mounts runs/ from a DIFFERENT clone: {source} "
                     f"instead of {expected_runs} -- simulations would land in the "
                     "wrong repository (issue #26)",
                     recreate)
    return Check("runs/ mount", True,
                 f"{RUNS_MOUNT_DST} bind-mounted from this clone ({source})")


def _version_tuple(version: str) -> tuple[int, ...] | None:
    match = re.match(r"v?(\d+(?:\.\d+)*)", version.strip())
    return tuple(int(p) for p in match.group(1).split(".")) if match else None


def check_release() -> Check | None:
    """Notify-only release check (issue #48): reports a newer upstream
    release, never applies anything, and degrades to a silent skip when
    offline, rate-limited, or when no releases exist (none do today)."""
    product = product_version()
    if product is None:
        return None
    req = urllib.request.Request(RELEASES_URL, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "foamagent-doctor",
    })
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 -- ANY failure means silent skip, by design
        return None
    if isinstance(data, dict):  # a /releases/latest-style single object
        data = [data]
    if not isinstance(data, list) or not data:
        return None
    tag = str(data[0].get("tag_name") or "") if isinstance(data[0], dict) else ""
    latest, current = _version_tuple(tag), _version_tuple(product)
    if latest is None or current is None:
        return None
    if latest > current:
        return Check("release check", False,
                     f"a newer release {tag} is available (installed: {product}) -- "
                     "notify only, nothing has been changed",
                     LOCKSTEP_FIX, warn=True)
    return Check("release check", True, f"installed {product}, latest release {tag}")


def check_host_openfoam() -> Check | None:
    """Detect-and-name for ESI OpenFOAM installs (openfoam.com), issue #48.

    A sourced OpenFOAM environment leaves WM_PROJECT_DIR/WM_PROJECT_VERSION
    in the shell; ESI releases are calendar-versioned (v2312 etc.) where
    Foundation's are plain integers. No host install at all is the common
    case and reports nothing.
    """
    wm_dir = os.environ.get("WM_PROJECT_DIR", "")
    if not wm_dir:
        return None
    wm_version = os.environ.get("WM_PROJECT_VERSION", "")
    dir_hint = wm_dir.replace("\\", "/").lower()
    is_esi = bool(re.match(r"v\d{4}", wm_version)) \
        or "openfoam.com" in dir_hint \
        or re.search(r"openfoam-?v?\d{4}", dir_hint)
    if is_esi:
        return Check("host OpenFOAM", False,
                     f"ESI OpenFOAM (openfoam.com) {wm_version or wm_dir} detected on "
                     "this machine; Foam-Agent targets Foundation OpenFOAM v10 "
                     "(openfoam.org), which the container provides -- generated cases "
                     "use Foundation dictionaries",
                     "nothing to change for container runs; to adapt a generated case "
                     "to your ESI install, use the translate_case_to_esi tool "
                     "(best-effort translation)",
                     warn=True)
    return Check("host OpenFOAM", True,
                 f"Foundation OpenFOAM {wm_version or wm_dir} on host; the container "
                 "provides the pinned v10 toolchain either way")


def onboarding_needed() -> bool:
    """First-run signal for the onboarding skill (spec #29).

    True only when both first-run signals are present: no seeded
    user-preferences file (config/user.yml, template-seeded during
    onboarding) and no prior runs in the ledger (no runs/ledger.md, or one
    with zero rows).
    """
    if (REPO / "config" / "user.yml").is_file():
        return False
    return not ledger.read_rows(str(REPO / "runs"))


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
            container = check_container(docker)
            checks.append(container)
            if container.ok:
                checks.append(check_version_lockstep(docker))
                checks.append(check_runs_mount(docker))
    host_openfoam = check_host_openfoam()
    if host_openfoam is not None:
        checks.append(host_openfoam)
    checks.append(check_endpoint())
    release = check_release()
    if release is not None:
        checks.append(release)

    # Green = the install contract holds: every hard check passes (warns
    # never gate). A lockstep or wrong-clone-mount failure is red even
    # while the endpoint answers.
    healthy = all(c.ok or c.warn for c in checks)

    if args.json:
        print(json.dumps({"healthy": healthy,
                          "onboardingNeeded": onboarding_needed(),
                          "checks": [vars(c) for c in checks]}, indent=2))
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
        print("and say: \"set up foam-agent for me\" -- the foam-setup skill does this interactively.")
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
