"""CLI tests for scripts/doctor.py (issue #48).

The seam (confirmed in the parent spec #29): the doctor is tested as a CLI --
subprocess invocations over a staged fake clone in tmp_path, asserting JSON
fields, exit codes, and printed fix commands. External probes are faked at
the process boundary: a fake docker executable placed on PATH (prior art:
scripts/mock_sbatch.sh, tests/test_run_sourced.py), the MCP endpoint and the
release feed pointed at file:// URLs via their env overrides. Key-free,
stdlib-only, no OpenFOAM, no real docker, no network.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

RUNS_MOUNT_DST = "/home/openfoam/Foam-Agent/runs"

# The fake docker CLI: a tiny dispatcher that answers each subcommand from a
# canned-responses file written by the test. Installed on PATH as both a
# POSIX script and a .bat wrapper so the same tests run on Linux CI and
# Windows dev machines.
FAKE_DOCKER_IMPL = """\
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "docker_responses.json"), encoding="utf-8") as fh:
    responses = json.load(fh)
sub = sys.argv[1] if len(sys.argv) > 1 else ""
r = responses.get(sub)
if r is None:
    sys.stderr.write("fake docker: unhandled subcommand %r\\n" % (sub,))
    sys.exit(9)
sys.stdout.write(r.get("out", ""))
sys.exit(r.get("rc", 0))
"""


def make_clone(tmp_path, version="3.0.0"):
    """A minimal fake Foam-Agent clone with the real doctor installed in it.

    doctor.py locates its clone from its own path, so copying the real script
    into tmp_path/scripts makes every filesystem probe (pyproject version,
    config/user.yml, runs/ledger.md, database/faiss) hermetic.
    """
    (tmp_path / "scripts").mkdir()
    shutil.copy(REPO / "scripts" / "doctor.py", tmp_path / "scripts" / "doctor.py")
    (tmp_path / "src").mkdir()
    shutil.copy(REPO / "src" / "ledger.py", tmp_path / "src" / "ledger.py")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "foamagent"\nversion = "%s"\n' % version,
        encoding="utf-8",
    )
    faiss = tmp_path / "database" / "faiss"
    faiss.mkdir(parents=True)
    (faiss / "index.faiss").write_bytes(b"\0" * 20_000)  # a real (non-pointer) index
    (tmp_path / "runs").mkdir()
    return tmp_path


def seed_preferences(clone):
    (clone / "config").mkdir(exist_ok=True)
    (clone / "config" / "user.yml").write_text("units: SI\n", encoding="utf-8")


def seed_run(clone, name="cavity"):
    """A prior run: one ledger row written by the same producer the server uses."""
    sys.path.insert(0, str(REPO / "src"))
    import ledger  # noqa: E402

    case = clone / "runs" / name
    case.mkdir(parents=True)
    ledger.track_planned(str(clone / "runs"), str(case))


def default_docker_responses(clone, stamp="3.0.0"):
    mounts = [{"Source": str(clone / "runs"), "Destination": RUNS_MOUNT_DST}]
    return {
        "version": {"out": "27.0.1\n", "rc": 0},
        "images": {"out": "foamagent:latest 10.4GB\n", "rc": 0},
        "ps": {"out": "Up 2 hours\n", "rc": 0},
        "inspect": {"out": json.dumps(mounts) + "\n", "rc": 0},
        "exec": {"out": stamp + "\n", "rc": 0},
    }


def install_fake_docker(clone, responses):
    fakebin = clone / "fakebin"
    fakebin.mkdir(exist_ok=True)
    (fakebin / "docker_impl.py").write_text(FAKE_DOCKER_IMPL, encoding="utf-8")
    (fakebin / "docker_responses.json").write_text(
        json.dumps(responses), encoding="utf-8"
    )
    (fakebin / "docker.bat").write_text(
        '@echo off\n"%s" "%%~dp0docker_impl.py" %%*\nexit /b %%ERRORLEVEL%%\n'
        % sys.executable,
        encoding="utf-8",
    )
    posix = fakebin / "docker"
    posix.write_text(
        '#!/bin/sh\nexec "%s" "$(dirname "$0")/docker_impl.py" "$@"\n'
        % sys.executable,
        encoding="utf-8",
        newline="\n",
    )
    posix.chmod(0o755)
    return fakebin


def run_doctor(clone, *args, docker=None, endpoint_up=True, releases="offline",
               env_extra=None):
    """Invoke the doctor exactly as a user or skill would (CLI in, JSON out).

    External boundaries are faked: docker via a PATH executable, the MCP
    endpoint and the release feed via their file://-capable env overrides
    ("offline" points the release check at a URL that errors).
    """
    responses = default_docker_responses(clone)
    responses.update(docker or {})
    fakebin = install_fake_docker(clone, responses)

    env = {k: v for k, v in os.environ.items() if not k.startswith("WM_PROJECT")}
    env["PATH"] = str(fakebin) + os.pathsep + env.get("PATH", "")

    endpoint = clone / "endpoint.ok"
    if endpoint_up:
        endpoint.write_text("ok", encoding="utf-8")
    env["FOAMAGENT_MCP_URL"] = endpoint.as_uri()

    releases_file = clone / "releases.json"
    if releases != "offline":
        releases_file.write_text(json.dumps(releases), encoding="utf-8")
    env["FOAMAGENT_RELEASES_URL"] = releases_file.as_uri()

    env.update(env_extra or {})

    result = subprocess.run(
        [sys.executable, str(clone / "scripts" / "doctor.py"), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
    env=env,
    )
    result.stdout.encode("ascii")  # ASCII-output contract (issues #18/#20)
    result.stderr.encode("ascii")
    return result


def doctor_json(clone, **kwargs):
    result = run_doctor(clone, "--json", **kwargs)
    return result, json.loads(result.stdout)


def check_named(data, name):
    matches = [c for c in data["checks"] if c["name"] == name]
    assert len(matches) == 1, "expected exactly one %r check, got %r" % (
        name, [c["name"] for c in data["checks"]])
    return matches[0]


def mounts_response(source, destination=RUNS_MOUNT_DST):
    return {"inspect": {
        "out": json.dumps([{"Source": source, "Destination": destination}]) + "\n",
        "rc": 0,
    }}


# ---------------------------------------------------------------------------
# runs/ mount: source must be THIS clone's runs directory (issue #26)
# ---------------------------------------------------------------------------

def test_mount_from_this_clone_is_green(tmp_path):
    clone = make_clone(tmp_path)

    result, data = doctor_json(clone, docker=mounts_response(str(clone / "runs")))

    check = check_named(data, "runs/ mount")
    assert check["ok"] is True


def test_mount_from_a_different_clone_fails_with_recreate_fix(tmp_path):
    """The live #26 evidence: a container bind-mounting another clone's runs/
    passed the old destination-only check."""
    clone = make_clone(tmp_path)
    elsewhere = str(tmp_path.parent / "another-clone" / "runs")

    result, data = doctor_json(clone, docker=mounts_response(elsewhere))

    check = check_named(data, "runs/ mount")
    assert check["ok"] is False
    assert check["warn"] is False  # a hard gate: runs land in the wrong repo
    assert "another-clone" in check["detail"]  # names the wrong source
    assert str(clone / "runs") in check["detail"]  # and the expected one
    assert "docker rm -f foamagent-mcp" in check["fix"]
    assert data["healthy"] is False
    assert result.returncode == 1


@pytest.mark.skipif(sys.platform != "win32",
                    reason="drive-letter mount translation is a Windows-host concern")
def test_docker_desktop_mount_source_format_matches(tmp_path):
    """Docker Desktop on Windows reports the source as /run/desktop/mnt/host/c/...;
    that is still this clone's runs directory."""
    clone = make_clone(tmp_path)
    runs = (clone / "runs").resolve()
    desktop = "/run/desktop/mnt/host/" + runs.drive[0].lower() + runs.as_posix()[2:]

    result, data = doctor_json(clone, docker=mounts_response(desktop))

    check = check_named(data, "runs/ mount")
    assert check["ok"] is True, check["detail"]


@pytest.mark.skipif(sys.platform != "win32",
                    reason="case-insensitive paths are a Windows-host concern")
def test_mount_source_comparison_ignores_case_and_slashes_on_windows(tmp_path):
    clone = make_clone(tmp_path)
    shouty = str((clone / "runs").resolve()).upper()

    result, data = doctor_json(clone, docker=mounts_response(shouty))

    check = check_named(data, "runs/ mount")
    assert check["ok"] is True, check["detail"]


def test_unparseable_mount_source_warns_not_fails(tmp_path):
    """A named volume is not a path we can compare against the clone."""
    clone = make_clone(tmp_path)

    result, data = doctor_json(clone, docker=mounts_response("foamagent_data"))

    check = check_named(data, "runs/ mount")
    assert check["ok"] is False
    assert check["warn"] is True
    assert data["healthy"] is True
    assert result.returncode == 0


def test_missing_runs_mount_still_warns(tmp_path):
    clone = make_clone(tmp_path)

    result, data = doctor_json(
        clone, docker={"inspect": {"out": "[]\n", "rc": 0}})

    check = check_named(data, "runs/ mount")
    assert check["ok"] is False
    assert check["warn"] is True
    assert "docker rm -f foamagent-mcp" in check["fix"]


# ---------------------------------------------------------------------------
# ESI detection: name the fork mismatch before it bites
# ---------------------------------------------------------------------------

def test_esi_host_environment_is_detected_and_named(tmp_path):
    clone = make_clone(tmp_path)

    result, data = doctor_json(clone, env_extra={
        "WM_PROJECT_DIR": "/usr/lib/openfoam/openfoam2312",
        "WM_PROJECT_VERSION": "v2312",
    })

    check = check_named(data, "host OpenFOAM")
    assert check["ok"] is False
    assert check["warn"] is True  # detect-and-name, not a gate
    assert "ESI" in check["detail"]
    assert "v2312" in check["detail"]
    assert "v10" in check["detail"]  # the Foundation pin, named up front
    assert "translate_case_to_esi" in check["fix"]  # the escape hatch
    assert "best-effort" in check["fix"]
    assert data["healthy"] is True
    assert result.returncode == 0


def test_foundation_host_environment_is_green(tmp_path):
    clone = make_clone(tmp_path)

    result, data = doctor_json(clone, env_extra={
        "WM_PROJECT_DIR": "/opt/openfoam10",
        "WM_PROJECT_VERSION": "10",
    })

    check = check_named(data, "host OpenFOAM")
    assert check["ok"] is True


def test_no_host_openfoam_skips_the_fork_check(tmp_path):
    """Most users have no OpenFOAM on the host at all -- nothing to detect."""
    clone = make_clone(tmp_path)

    result, data = doctor_json(clone)

    assert "host OpenFOAM" not in [c["name"] for c in data["checks"]]


# ---------------------------------------------------------------------------
# Release check: notify-only, degrades to a silent skip
# ---------------------------------------------------------------------------

def test_newer_release_is_notified_without_gating(tmp_path):
    clone = make_clone(tmp_path, version="3.0.0")

    result, data = doctor_json(clone, releases=[{"tag_name": "v3.2.0"}])

    check = check_named(data, "release check")
    assert check["ok"] is False
    assert check["warn"] is True  # notify-only: never gates
    assert "v3.2.0" in check["detail"]
    assert "3.0.0" in check["detail"]
    assert data["healthy"] is True
    assert result.returncode == 0


def test_installed_version_matching_latest_release_is_green(tmp_path):
    clone = make_clone(tmp_path, version="3.0.0")

    result, data = doctor_json(clone, releases=[{"tag_name": "v3.0.0"}])

    check = check_named(data, "release check")
    assert check["ok"] is True


def test_release_check_silently_skips_when_offline(tmp_path):
    clone = make_clone(tmp_path)

    result, data = doctor_json(clone, releases="offline")

    assert "release check" not in [c["name"] for c in data["checks"]]
    assert data["healthy"] is True


def test_release_check_silently_skips_when_no_releases_exist(tmp_path):
    """None exist today -- an empty release list must not produce noise."""
    clone = make_clone(tmp_path)

    result, data = doctor_json(clone, releases=[])

    assert "release check" not in [c["name"] for c in data["checks"]]
    assert data["healthy"] is True


# ---------------------------------------------------------------------------
# onboardingNeeded: first-run signal for the onboarding skill
# ---------------------------------------------------------------------------

def test_fresh_clone_reports_onboarding_needed(tmp_path):
    clone = make_clone(tmp_path)  # no config/user.yml, no ledger

    result, data = doctor_json(clone)

    assert data["onboardingNeeded"] is True


def test_seeded_preferences_clear_onboarding(tmp_path):
    clone = make_clone(tmp_path)
    seed_preferences(clone)

    result, data = doctor_json(clone)

    assert data["onboardingNeeded"] is False


def test_prior_runs_clear_onboarding(tmp_path):
    clone = make_clone(tmp_path)
    seed_run(clone)

    result, data = doctor_json(clone)

    assert data["onboardingNeeded"] is False


def test_version_lockstep_match_is_green(tmp_path):
    clone = make_clone(tmp_path, version="3.0.0")

    result, data = doctor_json(clone, docker={"exec": {"out": "3.0.0\n", "rc": 0}})

    check = check_named(data, "version lockstep")
    assert check["ok"] is True
    assert "3.0.0" in check["detail"]


def test_version_lockstep_mismatch_fails_loud_naming_both_versions(tmp_path):
    clone = make_clone(tmp_path, version="3.1.0")

    result, data = doctor_json(clone, docker={"exec": {"out": "3.0.0\n", "rc": 0}})

    check = check_named(data, "version lockstep")
    assert check["ok"] is False
    assert check["warn"] is False  # a hard gate, not a warning
    assert "3.1.0" in check["detail"] and "3.0.0" in check["detail"]
    # the fix is the matching pull/update pair
    assert "git pull" in check["fix"]
    assert "docker pull" in check["fix"]
    # mismatch makes the doctor red even though the endpoint is up
    assert data["healthy"] is False
    assert result.returncode == 1


def test_version_lockstep_mismatch_prints_fail_and_fix_in_text_mode(tmp_path):
    clone = make_clone(tmp_path, version="3.1.0")

    result = run_doctor(clone, docker={"exec": {"out": "3.0.0\n", "rc": 0}})

    assert result.returncode == 1
    assert "[FAIL] version lockstep" in result.stdout
    assert "docker pull" in result.stdout


def test_missing_image_stamp_is_warn_not_fail(tmp_path):
    """Images built before the stamp mechanism have no version file yet."""
    clone = make_clone(tmp_path)

    result, data = doctor_json(clone, docker={"exec": {
        "out": "cat: /etc/foamagent-version: No such file or directory\n",
        "rc": 1,
    }})

    check = check_named(data, "version lockstep")
    assert check["ok"] is False
    assert check["warn"] is True
    assert data["healthy"] is True  # warn never gates
    assert result.returncode == 0


def test_empty_ledger_is_still_a_first_run(tmp_path):
    """A ledger file with zero rows carries no evidence of a prior run."""
    clone = make_clone(tmp_path)
    (clone / "runs" / "ledger.md").write_text(
        "# Foam-Agent run ledger\n\n"
        "| ID | Case | Created | Solver | Mesh | Status | Result | Key result | Notes |\n"
        "|----|------|---------|--------|------|--------|--------|------------|-------|\n",
        encoding="utf-8",
    )

    result, data = doctor_json(clone)

    assert data["onboardingNeeded"] is True


# ---------------------------------------------------------------------------
# Whole contract: green when everything holds, red when the daemon is down
# ---------------------------------------------------------------------------

def test_healthy_setup_is_green(tmp_path):
    clone = make_clone(tmp_path)
    seed_preferences(clone)

    result, data = doctor_json(clone, releases=[{"tag_name": "v3.0.0"}])

    assert data["healthy"] is True
    assert result.returncode == 0
    names = [c["name"] for c in data["checks"]]
    for expected in ("Docker daemon", "foamagent image", "foamagent-mcp container",
                     "version lockstep", "runs/ mount", "MCP endpoint",
                     "release check"):
        assert expected in names
    assert all(c["ok"] or c["warn"] for c in data["checks"])


def test_daemon_down_fails_without_reaching_image_checks(tmp_path):
    clone = make_clone(tmp_path)

    result, data = doctor_json(
        clone,
        docker={"version": {"out": "error during connect\n", "rc": 1}},
        endpoint_up=False,
    )

    daemon = check_named(data, "Docker daemon")
    assert daemon["ok"] is False
    assert daemon["fix"]  # tells the user what to start
    names = [c["name"] for c in data["checks"]]
    assert "foamagent image" not in names  # later checks are gated off
    assert "version lockstep" not in names
    assert data["healthy"] is False
    assert result.returncode == 1
