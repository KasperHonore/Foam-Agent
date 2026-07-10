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
import re
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
    config/user.yml, runs/ledger.md, database/faiss) hermetic. The doctor is
    deliberately self-contained (it travels with the product, issue #51), so
    nothing else from the clone is needed.
    """
    (tmp_path / "scripts").mkdir()
    shutil.copy(REPO / "scripts" / "doctor.py", tmp_path / "scripts" / "doctor.py")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "foamagent"\nversion = "%s"\n' % version,
        encoding="utf-8",
    )
    faiss = tmp_path / "database" / "faiss"
    faiss.mkdir(parents=True)
    (faiss / "index.faiss").write_bytes(b"\0" * 20_000)  # a real (non-pointer) index
    (tmp_path / "runs").mkdir()
    return tmp_path


def make_global_install(tmp_path, stamp="3.0.0"):
    """The bundled doctor as `npx skills add` lands it (issue #51).

    references/doctor.py inside an installed skill directory, with no
    Foam-Agent clone anywhere above it, plus a faked central root
    (FOAMAGENT_HOME) whose runs/ is the central global mount. stamp mirrors
    the version line scripts/sync_agent_assets.py writes at bundle time;
    stamp=None stages a hand-copied canonical file (no stamp).

    Returns (script path, central home directory).
    """
    refs = tmp_path / "proj" / ".claude" / "skills" / "foam-setup" / "references"
    refs.mkdir(parents=True)
    content = (REPO / "scripts" / "doctor.py").read_text(encoding="utf-8")
    if stamp is not None:
        content, n = re.subn(
            r"^INSTALLED_PRODUCT_VERSION = .*$",
            'INSTALLED_PRODUCT_VERSION = "%s"' % stamp,
            content, count=1, flags=re.MULTILINE,
        )
        assert n == 1, "doctor.py lost its INSTALLED_PRODUCT_VERSION stamp line"
    (refs / "doctor.py").write_text(content, encoding="utf-8")
    home = tmp_path / "foamhome"
    (home / "runs").mkdir(parents=True)
    return refs / "doctor.py", home


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
               env_extra=None, script=None, cwd=None):
    """Invoke the doctor exactly as a user or skill would (CLI in, JSON out).

    External boundaries are faked: docker via a PATH executable, the MCP
    endpoint and the release feed via their file://-capable env overrides
    ("offline" points the release check at a URL that errors). script
    defaults to the staged clone's scripts/doctor.py; global-install tests
    pass the staged bundled copy (and a cwd, when the invocation directory
    itself is under test).
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
        [sys.executable, str(script or clone / "scripts" / "doctor.py"), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=cwd,
    )
    result.stdout.encode("ascii")  # ASCII-output contract (issues #18/#20)
    result.stderr.encode("ascii")
    return result


def doctor_json(clone, **kwargs):
    result = run_doctor(clone, "--json", **kwargs)
    return result, json.loads(result.stdout)


def global_json(tmp_path, stamp="3.0.0", mount_source=None, docker=None,
                env_extra=None, **kwargs):
    """A no-clone doctor run: staged bundled copy, faked central root.

    The container is mounted from the central runs directory unless
    mount_source overrides it. Returns (result, data, home).
    """
    script, home = make_global_install(tmp_path, stamp=stamp)
    responses = dict(docker or {})
    responses.setdefault(
        "inspect", mounts_response(mount_source or str(home / "runs"))["inspect"])
    env = dict(env_extra or {})
    env.setdefault("FOAMAGENT_HOME", str(home))
    result, data = doctor_json(tmp_path, script=script, docker=responses,
                               env_extra=env, **kwargs)
    return result, data, home


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


# ---------------------------------------------------------------------------
# Global install path (issue #51): the bundled doctor runs without a clone
# ---------------------------------------------------------------------------

def test_no_clone_doctor_skips_clone_only_checks_but_still_probes_docker(tmp_path):
    """Outside any Foam-Agent clone the doctor still delivers its verdict:
    clone-only checks skip with an explanatory note instead of failing, and
    the docker/endpoint probes run as always."""
    result, data, home = global_json(tmp_path)

    lfs = check_named(data, "git-lfs FAISS indices")
    assert lfs["ok"] is True
    assert "skip" in lfs["detail"].lower()  # a note, not a silent absence
    assert "clone" in lfs["detail"]  # says WHY it was skipped
    names = [c["name"] for c in data["checks"]]
    for expected in ("Docker daemon", "foamagent image", "foamagent-mcp container",
                     "MCP endpoint"):
        assert expected in names
    assert data["healthy"] is True
    assert result.returncode == 0


def test_no_clone_doctor_accepts_the_central_global_mount(tmp_path):
    result, data, home = global_json(tmp_path)

    check = check_named(data, "runs/ mount")
    assert check["ok"] is True, check["detail"]
    assert "central" in check["detail"]  # names WHICH accepted source matched


def test_clone_doctor_accepts_the_central_global_mount_too(tmp_path):
    """A clone user may deliberately anchor all projects on the central
    mount; that is a healthy setup, not a wrong-clone failure."""
    clone = make_clone(tmp_path)
    home = tmp_path / "foamhome"
    (home / "runs").mkdir(parents=True)

    result, data = doctor_json(
        clone,
        docker=mounts_response(str(home / "runs")),
        env_extra={"FOAMAGENT_HOME": str(home)},
    )

    check = check_named(data, "runs/ mount")
    assert check["ok"] is True, check["detail"]


def test_no_clone_doctor_still_fails_a_wrong_source_mount(tmp_path):
    """Global mode keeps #48's teeth: a container mounted from some clone's
    runs/ is NOT the central mount -- runs would land somewhere else."""
    elsewhere = str(tmp_path / "another-clone" / "runs")

    result, data, home = global_json(tmp_path, mount_source=elsewhere)

    check = check_named(data, "runs/ mount")
    assert check["ok"] is False
    assert check["warn"] is False  # still a hard gate
    assert "another-clone" in check["detail"]  # names the wrong source
    assert str(home / "runs") in check["detail"]  # and the expected central one
    assert "docker rm -f foamagent-mcp" in check["fix"]
    assert str(home / "runs") in check["fix"]  # the fix recipe mounts centrally
    assert data["healthy"] is False
    assert result.returncode == 1


def test_no_clone_lockstep_compares_the_installed_stamp_against_the_image(tmp_path):
    """In global mode the installed side of the lockstep is the version
    sync_agent_assets.py stamped into the bundled copy -- a mismatch with
    the image stamp is the same hard gate as in a clone (issue #48)."""
    result, data, home = global_json(
        tmp_path, stamp="3.1.0", docker={"exec": {"out": "3.0.0\n", "rc": 0}})

    check = check_named(data, "version lockstep")
    assert check["ok"] is False
    assert check["warn"] is False  # a hard gate on the global path too
    assert "3.1.0" in check["detail"] and "3.0.0" in check["detail"]
    assert "installed skills" in check["detail"]  # names the global-mode side
    # the fix is the matching update pair for THIS install path
    assert "npx skills add" in check["fix"]
    assert "git pull" not in check["fix"]  # there is no clone to pull
    assert "docker pull" in check["fix"]
    assert data["healthy"] is False
    assert result.returncode == 1


def test_no_clone_lockstep_match_is_green(tmp_path):
    result, data, home = global_json(
        tmp_path, stamp="3.0.0", docker={"exec": {"out": "3.0.0\n", "rc": 0}})

    check = check_named(data, "version lockstep")
    assert check["ok"] is True
    assert "3.0.0" in check["detail"]


def test_unstamped_hand_copied_doctor_warns_it_cannot_compare(tmp_path):
    """A doctor copy that never went through sync_agent_assets.py carries no
    stamp: the installed-side version is genuinely unknowable, which is a
    documented warn (with the reinstall fix), never a silent pass."""
    result, data, home = global_json(tmp_path, stamp=None)

    check = check_named(data, "version lockstep")
    assert check["ok"] is False
    assert check["warn"] is True
    assert "stamp" in check["detail"]
    assert "npx skills add" in check["fix"]  # reinstalling restores the stamp
    assert data["healthy"] is True  # warn never gates
    assert result.returncode == 0


def test_no_clone_release_check_notifies_from_the_installed_stamp(tmp_path):
    result, data, home = global_json(
        tmp_path, stamp="3.0.0", releases=[{"tag_name": "v3.2.0"}])

    check = check_named(data, "release check")
    assert check["ok"] is False
    assert check["warn"] is True  # notify-only, same as in a clone
    assert "v3.2.0" in check["detail"]
    assert "3.0.0" in check["detail"]
    assert data["healthy"] is True


def test_no_clone_onboarding_follows_the_central_ledger(tmp_path):
    """Global-mode first-run signal: no rows in the central ledger."""
    result, data, home = global_json(tmp_path)

    assert data["onboardingNeeded"] is True


def test_rows_in_the_central_ledger_clear_onboarding(tmp_path):
    """A prior run recorded by the real producer -- nested under a project
    key, as global-mode runs are -- clears the first-run signal."""
    script, home = make_global_install(tmp_path)
    sys.path.insert(0, str(REPO / "src"))
    import ledger  # noqa: E402

    case = home / "runs" / "my-project" / "cavity"
    case.mkdir(parents=True)
    ledger.track_planned(str(home / "runs"), str(case))

    result, data = doctor_json(
        tmp_path, script=script,
        docker=mounts_response(str(home / "runs")),
        env_extra={"FOAMAGENT_HOME": str(home)},
    )

    assert data["onboardingNeeded"] is False


def test_the_doctor_travels_with_the_foam_setup_skill():
    """The sync mechanism bundles a version-stamped doctor into foam-setup's
    references (issue #51): one canonical source (scripts/doctor.py), a
    generated copy with the standard GENERATED note -- the drift check
    (test_agent_assets_in_sync) keeps it regenerated."""
    version = re.search(
        r'^version\s*=\s*"([^"]+)"',
        (REPO / "pyproject.toml").read_text(encoding="utf-8"), re.MULTILINE,
    ).group(1)

    for target in (".claude/skills", ".opencode/skill", ".codex/skills", ".cursor/skills"):
        bundled = REPO / target / "foam-setup" / "references" / "doctor.py"
        assert bundled.is_file(), f"doctor not bundled into {target}/foam-setup"
        content = bundled.read_text(encoding="utf-8")
        assert "GENERATED by scripts/sync_agent_assets.py" in content
        assert f'INSTALLED_PRODUCT_VERSION = "{version}"' in content


def test_bundled_doctor_runs_from_an_arbitrary_directory(tmp_path):
    """The bundled copy is invoked from wherever the user's shell happens to
    be -- its verdict must not depend on the cwd."""
    elsewhere = tmp_path / "somewhere" / "else"
    elsewhere.mkdir(parents=True)

    result, data, home = global_json(tmp_path, cwd=str(elsewhere))

    assert data["healthy"] is True
    assert result.returncode == 0
