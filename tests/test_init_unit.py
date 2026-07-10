"""CLI tests for the one-command init (scripts/init.py, issue #49).

The seam (confirmed in the parent spec #29): the init is tested as a CLI --
subprocess invocations in tmp directories, asserting stdout, exit codes, and
what appeared (or did not appear) on disk. External probes are faked at the
process boundary: fake git / docker / agent-CLI executables placed on a
controlled PATH, behavior driven by FAKE_* environment variables. Prior art:
tests/test_ledger_check_unit.py, tests/test_runs_list_unit.py, and the mock
sbatch script. Key-free, stdlib-only, no network, no real clone in CI.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "init.py"
WINDOWS = os.name == "nt"

# ---------------------------------------------------------------------------
# Fake externals: python implementations behind platform-appropriate shims
# ---------------------------------------------------------------------------

FAKE_GIT = """\
import os, sys
from pathlib import Path

args = sys.argv[1:]
log = os.environ.get("FAKE_GIT_LOG")
if log:
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(" ".join(args) + "\\n")

def faiss_bytes():
    kind = os.environ.get("FAKE_CLONE_FAISS", "real")
    return b"x" * (130 if kind == "pointer" else 20000)

cmd = args[0] if args else ""
if cmd == "lfs":
    if os.environ.get("FAKE_GIT_LFS") == "missing":
        sys.stderr.write("git: 'lfs' is not a git command. See 'git --help'.\\n")
        sys.exit(1)
    sub = args[1] if len(args) > 1 else ""
    if sub == "version":
        print("git-lfs/3.4.0 (fake)")
    elif sub == "pull" and os.environ.get("FAKE_LFS_PULL") == "fixes":
        for f in Path(".").rglob("*.faiss"):
            f.write_bytes(b"x" * 20000)
    sys.exit(0)
if cmd == "ls-remote":
    tags = os.environ.get("FAKE_GIT_TAGS", "")
    if tags == "NETFAIL":
        sys.stderr.write("fatal: unable to access remote\\n")
        sys.exit(128)
    for tag in [t for t in tags.split(",") if t]:
        print("deadbeef\\trefs/tags/" + tag)
    sys.exit(0)
if cmd == "clone":
    if os.environ.get("FAKE_GIT_CLONE") == "fail":
        sys.stderr.write("fatal: could not clone\\n")
        sys.exit(128)
    target = Path(args[-1])
    (target / "scripts").mkdir(parents=True)
    (target / "scripts" / "doctor.py").write_text("# stub\\n", encoding="utf-8")
    faiss = target / "database" / "faiss" / "openfoam_tutorials_details"
    faiss.mkdir(parents=True)
    (faiss / "index.faiss").write_bytes(faiss_bytes())
    sys.exit(0)
sys.exit(0)
"""

FAKE_DOCKER = """\
import os, sys
if os.environ.get("FAKE_DOCKER") == "down":
    sys.stderr.write("error during connect: the docker daemon is not running\\n")
    sys.exit(1)
print("24.0.7")
sys.exit(0)
"""

FAKE_CLI = "import sys\nsys.exit(0)\n"


def make_exe(bin_dir: Path, name: str, body: str) -> None:
    """Place a fake executable named `name` on the controlled PATH."""
    impl = bin_dir / ("_" + name + "_impl.py")
    impl.write_text(body, encoding="utf-8")
    if WINDOWS:
        (bin_dir / (name + ".bat")).write_text(
            '@echo off\r\n"%s" "%s" %%*\r\n' % (sys.executable, impl),
            encoding="utf-8",
        )
    else:
        exe = bin_dir / name
        exe.write_text('#!/bin/sh\nexec "%s" "%s" "$@"\n' % (sys.executable, impl))
        exe.chmod(0o755)


def setup(tmp_path: Path, *, git=True, docker=True, clis=()):
    """A working directory plus a PATH containing only the requested fakes.

    Returns (workdir, base_env, git_log_path). The git log records every
    fake-git invocation so tests can assert what was (not) run.
    """
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    if git:
        make_exe(bin_dir, "git", FAKE_GIT)
    if docker:
        make_exe(bin_dir, "docker", FAKE_DOCKER)
    for cli in clis:
        make_exe(bin_dir, cli, FAKE_CLI)
    log = tmp_path / "git_calls.log"
    env = dict(os.environ)
    env["PATH"] = str(bin_dir)
    env["FAKE_GIT_LOG"] = str(log)
    # The harshest console: init output must survive ascii:strict (issue #18).
    env["PYTHONIOENCODING"] = "ascii:strict"
    # Simulate a machine without Docker Desktop: the Windows loader restores
    # ProgramFiles in child processes, so the fallback needs its own switch.
    env["FOAM_INIT_DISABLE_DOCKER_FALLBACK"] = "1"
    return work, env, log


def run_init(work: Path, env: dict, *args: str) -> subprocess.CompletedProcess:
    """Invoke the init exactly as a user would: command line in, text out."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=work,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )


def git_calls(log: Path) -> str:
    return log.read_text(encoding="utf-8") if log.exists() else ""


# ---------------------------------------------------------------------------
# Prerequisite gate: validated before cloning, doctor's manners on failure
# ---------------------------------------------------------------------------

def test_missing_docker_cli_fails_with_fix_and_clones_nothing(tmp_path):
    work, env, log = setup(tmp_path, docker=False)

    result = run_init(work, env)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "docker not found" in result.stdout
    assert "docker.com" in result.stdout  # the exact fix, doctor's manners
    assert "clone" not in git_calls(log)  # read-only: nothing was cloned
    assert not (work / "Foam-Agent").exists()


def test_docker_daemon_down_fails_with_fix_and_clones_nothing(tmp_path):
    work, env, log = setup(tmp_path)
    env["FAKE_DOCKER"] = "down"

    result = run_init(work, env)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "daemon is not responding" in result.stdout
    assert "start Docker Desktop" in result.stdout
    assert "clone" not in git_calls(log)
    assert not (work / "Foam-Agent").exists()


def test_missing_git_lfs_fails_with_fix_and_clones_nothing(tmp_path):
    work, env, log = setup(tmp_path)
    env["FAKE_GIT_LFS"] = "missing"

    result = run_init(work, env)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "git-lfs" in result.stdout
    assert "git-lfs.com" in result.stdout
    assert "clone" not in git_calls(log)
    assert not (work / "Foam-Agent").exists()


def test_missing_git_fails_with_fix(tmp_path):
    work, env, log = setup(tmp_path, git=False)

    result = run_init(work, env)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "git not found" in result.stdout
    assert "git-scm.com" in result.stdout
    assert not (work / "Foam-Agent").exists()


def test_all_prerequisites_are_reported_in_one_run(tmp_path):
    """Doctor's manners: report every failed check at once, not one per run."""
    work, env, log = setup(tmp_path, docker=False)
    env["FAKE_GIT_LFS"] = "missing"

    result = run_init(work, env)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "git-lfs.com" in result.stdout
    assert "docker.com" in result.stdout


def test_prerequisite_failure_leaves_the_working_directory_untouched(tmp_path):
    work, env, log = setup(tmp_path, docker=False)

    run_init(work, env)

    assert list(work.iterdir()) == []  # read-only until the clone step


# ---------------------------------------------------------------------------
# Clone: latest release tag, numeric ordering; default branch when no releases
# ---------------------------------------------------------------------------

def test_clones_latest_release_tag_by_numeric_order(tmp_path):
    work, env, log = setup(tmp_path)
    env["FAKE_GIT_TAGS"] = "v1.2.0,v1.10.0,legacy-multiagent,v0.9"

    result = run_init(work, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "v1.10.0" in result.stdout  # numeric, not lexicographic (v1.2 < v1.10)
    calls = git_calls(log)
    assert "clone --branch v1.10.0" in calls
    assert (work / "Foam-Agent" / "scripts" / "doctor.py").is_file()


def test_no_releases_falls_back_to_default_branch_saying_so(tmp_path):
    work, env, log = setup(tmp_path)
    env["FAKE_GIT_TAGS"] = ""  # none published today (issue #49)

    result = run_init(work, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "no release" in result.stdout.lower()
    assert "default branch" in result.stdout.lower()
    calls = git_calls(log)
    assert "clone" in calls
    assert "--branch" not in calls
    assert (work / "Foam-Agent" / "scripts" / "doctor.py").is_file()


def test_target_flag_controls_the_clone_destination(tmp_path):
    work, env, log = setup(tmp_path)

    result = run_init(work, env, "--target", "my-foam")

    assert result.returncode == 0, result.stdout + result.stderr
    assert (work / "my-foam" / "scripts" / "doctor.py").is_file()
    assert not (work / "Foam-Agent").exists()


def test_unreachable_remote_fails_before_cloning(tmp_path):
    work, env, log = setup(tmp_path)
    env["FAKE_GIT_TAGS"] = "NETFAIL"

    result = run_init(work, env)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "could not reach" in result.stdout
    assert "network" in result.stdout
    assert "clone" not in git_calls(log)
    assert not (work / "Foam-Agent").exists()


def test_failed_clone_reports_and_exits_one(tmp_path):
    work, env, log = setup(tmp_path)
    env["FAKE_GIT_CLONE"] = "fail"

    result = run_init(work, env)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "clone" in result.stdout.lower()
    assert "FAIL" in result.stdout


# ---------------------------------------------------------------------------
# Idempotence guard: never touch an existing installation
# ---------------------------------------------------------------------------

def existing_clone(parent: Path, name: str) -> Path:
    clone = parent / name
    (clone / "scripts").mkdir(parents=True)
    (clone / "scripts" / "doctor.py").write_text("# real doctor\n", encoding="utf-8")
    return clone


def test_target_with_existing_clone_is_refused_politely_and_untouched(tmp_path):
    work, env, log = setup(tmp_path)
    clone = existing_clone(work, "Foam-Agent")
    before = (clone / "scripts" / "doctor.py").read_bytes()

    result = run_init(work, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "already contains a Foam-Agent clone" in result.stdout
    assert "git pull" in result.stdout  # how to update instead
    assert git_calls(log) == ""  # not even a read-only git call
    assert (clone / "scripts" / "doctor.py").read_bytes() == before


def test_running_inside_a_clone_is_refused_politely(tmp_path):
    work, env, log = setup(tmp_path)
    existing_clone(tmp_path, "work2")  # make the cwd itself a clone
    work = tmp_path / "work2"

    result = run_init(work, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "already inside a Foam-Agent clone" in result.stdout
    assert git_calls(log) == ""
    assert not (work / "Foam-Agent").exists()


def test_nonempty_target_that_is_not_a_clone_is_refused(tmp_path):
    work, env, log = setup(tmp_path)
    target = work / "Foam-Agent"
    target.mkdir()
    (target / "my-thesis.txt").write_text("precious\n", encoding="utf-8")

    result = run_init(work, env)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "not empty" in result.stdout
    assert "--target" in result.stdout  # points at the way out
    assert "clone" not in git_calls(log)
    assert (target / "my-thesis.txt").read_text(encoding="utf-8") == "precious\n"


def test_empty_existing_target_directory_is_fine(tmp_path):
    work, env, log = setup(tmp_path)
    (work / "Foam-Agent").mkdir()

    result = run_init(work, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (work / "Foam-Agent" / "scripts" / "doctor.py").is_file()


# ---------------------------------------------------------------------------
# Hand-off: detect agent CLIs, print the line that starts foam-onboard
# ---------------------------------------------------------------------------

def test_detected_cli_gets_a_tailored_handoff_line(tmp_path):
    work, env, log = setup(tmp_path, clis=("claude",))

    result = run_init(work, env)

    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert "claude" in out
    assert "onboard me" in out  # the foam-onboard trigger phrase
    assert "cd " in out  # open the clone first
    assert "Foam-Agent" in out.split("onboard me")[0]  # the clone is named before the trigger


def test_all_detected_clis_are_named(tmp_path):
    work, env, log = setup(tmp_path, clis=("claude", "cursor"))

    result = run_init(work, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "claude" in result.stdout
    assert "cursor" in result.stdout
    assert "codex" not in result.stdout.replace(
        "claude / cursor / codex / opencode", "")  # not claimed as installed


def test_no_cli_detected_gets_the_generic_handoff_line(tmp_path):
    work, env, log = setup(tmp_path)

    result = run_init(work, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "No supported agent CLI" in result.stdout
    assert "claude / cursor / codex / opencode" in result.stdout
    assert "onboard me" in result.stdout


def test_existing_clone_refusal_still_prints_the_handoff(tmp_path):
    """Idempotence: the second run ends at the same 'what next' as the first."""
    work, env, log = setup(tmp_path, clis=("claude",))
    existing_clone(work, "Foam-Agent")

    result = run_init(work, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "onboard me" in result.stdout
    assert "claude" in result.stdout


# ---------------------------------------------------------------------------
# Post-clone LFS validation: real FAISS indices, not pointers
# ---------------------------------------------------------------------------

def test_real_faiss_indices_validate_without_an_lfs_pull(tmp_path):
    work, env, log = setup(tmp_path)

    result = run_init(work, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAISS indices" in result.stdout
    assert "lfs pull" not in git_calls(log)


def test_pointer_faiss_indices_are_repaired_by_lfs_pull(tmp_path):
    work, env, log = setup(tmp_path)
    env["FAKE_CLONE_FAISS"] = "pointer"
    env["FAKE_LFS_PULL"] = "fixes"

    result = run_init(work, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "lfs pull" in git_calls(log)
    assert "WARN" not in result.stdout  # repaired, nothing left to warn about
    faiss = work / "Foam-Agent" / "database" / "faiss" / "openfoam_tutorials_details" / "index.faiss"
    assert faiss.stat().st_size >= 10_000  # real content, not a pointer


def test_unrepaired_pointers_warn_with_the_exact_fix(tmp_path):
    work, env, log = setup(tmp_path)
    env["FAKE_CLONE_FAISS"] = "pointer"  # and FAKE_LFS_PULL unset: pull is a no-op

    result = run_init(work, env)

    # Warn-only: the Docker image ships the indices baked in, so onboarding
    # still works -- but the user is told exactly how to materialize them.
    assert result.returncode == 0, result.stdout + result.stderr
    assert "WARN" in result.stdout
    assert "LFS pointers" in result.stdout
    assert "git lfs pull" in result.stdout
    assert "onboard me" in result.stdout  # the hand-off still happens


# ---------------------------------------------------------------------------
# ASCII output on the harshest console (every run above already uses
# PYTHONIOENCODING=ascii:strict; this pins the contract explicitly)
# ---------------------------------------------------------------------------

def test_output_is_ascii_and_stderr_is_quiet_on_the_happy_path(tmp_path):
    work, env, log = setup(tmp_path, clis=("claude",))
    env["FAKE_GIT_TAGS"] = "v2.0.0"

    result = run_init(work, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.isascii()
    assert result.stderr == ""  # no tracebacks, no encoding accidents
