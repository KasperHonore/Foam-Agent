"""Unit tests for the solver-log convergence parser (issue #39).

The parser (src/convergence.py) reads a case's solver log and returns typed
convergence facts plus a verdict with evidence. Tests sit at the mechanics
seam: they parse committed fixture logs and assert on the typed output's
fields — never on parser internals. Expected values are read from the
fixtures by eye (an independent source of truth), never recomputed the way
the parser does. Key-free, stdlib-only, no OpenFOAM.

Fixture provenance (tests/fixtures/convergence/):
- converged/ is the REAL Foundation v10 lid-driven-cavity shakedown run
  (log.icoFoam, log.blockMesh, Allrun.out, system/controlDict), genuine
  sigFpe startup banner included.
- diverged/, incomplete/, fatal/, fpe/ are synthetic logs derived from it —
  each keeps the genuine banner, so the banner-class false positive
  (PR #37's shakedown regression) stays permanently caught.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import convergence  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "convergence"


# ---------------------------------------------------------------------------
# The real converged cavity run (genuine banner included)
# ---------------------------------------------------------------------------

def test_converged_cavity_run_gets_converged_verdict_despite_banner():
    # The log carries the genuine "sigFpe : Enabling floating point exception
    # trapping" banner — it must never read as divergence (PR #37 policy).
    analysis = convergence.parse_solver_log(str(FIXTURES / "converged"))

    assert analysis.solver == "icoFoam"
    assert analysis.completed is True
    assert analysis.verdict == "converged"
    assert analysis.evidence  # names what drove the verdict
    assert analysis.fatal_errors == []


def test_converged_cavity_run_time_progress():
    # Read from the fixture by eye: first step "Time = 0.005s", last
    # "Time = 0.5s"; system/controlDict says endTime 0.5.
    analysis = convergence.parse_solver_log(str(FIXTURES / "converged"))

    assert analysis.time.first_time == pytest.approx(0.005)
    assert analysis.time.latest_time == pytest.approx(0.5)
    assert analysis.time.end_time == pytest.approx(0.5)


def test_converged_cavity_run_per_field_residuals():
    # All expected values read from the fixture by eye. p is solved twice per
    # step (PISO corrector) — first/last/worst span every solve of the field.
    analysis = convergence.parse_solver_log(str(FIXTURES / "converged"))

    by_field = {r.field: r for r in analysis.residuals}
    assert list(by_field) == ["Ux", "Uy", "p"]  # first-appearance order

    ux = by_field["Ux"]
    assert ux.first_initial == pytest.approx(1.0)
    assert ux.last_initial == pytest.approx(2.3091e-07)
    assert ux.last_final == pytest.approx(2.3091e-07)
    assert ux.worst_initial == pytest.approx(1.0)
    assert ux.worst_initial_time == pytest.approx(0.005)  # the very first step

    uy = by_field["Uy"]
    assert uy.first_initial == pytest.approx(0.0)
    assert uy.last_final == pytest.approx(5.0684e-07)
    assert uy.worst_initial == pytest.approx(0.260828)
    assert uy.worst_initial_time == pytest.approx(0.01)  # the second step

    p = by_field["p"]
    assert p.first_initial == pytest.approx(1.0)
    assert p.last_initial == pytest.approx(9.59103e-07)  # last of the two solves
    assert p.last_final == pytest.approx(9.59103e-07)
    assert p.worst_initial == pytest.approx(1.0)
    assert p.worst_initial_time == pytest.approx(0.005)


def test_converged_cavity_run_courant_and_continuity():
    # By eye: the largest "max:" value in the log is 0.852134, first reached
    # at Time = 0.465s; the last Courant line reads "mean: 0.222158
    # max: 0.852134"; the last continuity line ends "cumulative = 2.56141e-19".
    analysis = convergence.parse_solver_log(str(FIXTURES / "converged"))

    assert analysis.courant is not None
    assert analysis.courant.max == pytest.approx(0.852134)
    assert analysis.courant.max_time == pytest.approx(0.465)
    assert analysis.courant.last_mean == pytest.approx(0.222158)
    assert analysis.courant.last_max == pytest.approx(0.852134)
    assert analysis.cumulative_continuity == pytest.approx(2.56141e-19)


# ---------------------------------------------------------------------------
# Log selection: controlDict application by default, explicit name overrides
# ---------------------------------------------------------------------------

def test_default_log_selection_follows_controldict_application():
    # The fixture also contains log.blockMesh — the controlDict application
    # entry (icoFoam) must decide which log is parsed.
    analysis = convergence.parse_solver_log(str(FIXTURES / "converged"))
    assert analysis.log_file == "log.icoFoam"


def test_explicit_log_file_overrides_default_selection():
    analysis = convergence.parse_solver_log(str(FIXTURES / "converged"),
                                            log_file="log.blockMesh")
    assert analysis.log_file == "log.blockMesh"
    assert analysis.solver == "blockMesh"  # the log's own Exec entry
    assert analysis.completed is True      # blockMesh reached its End marker
    assert analysis.residuals == []        # a mesh utility solves no fields


def test_missing_application_entry_without_log_file_raises(tmp_path):
    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "controlDict").write_text("FoamFile {}\n")
    with pytest.raises(ValueError, match="log_file"):
        convergence.parse_solver_log(str(tmp_path))


def test_missing_log_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="log.icoFoam"):
        convergence.parse_solver_log(str(tmp_path), log_file="log.icoFoam")


# ---------------------------------------------------------------------------
# Diverged: residual explosion / Courant blow-up, with named evidence
# ---------------------------------------------------------------------------

def test_diverged_run_names_field_time_and_value_in_evidence():
    # Synthetic fixture, by eye: at Time = 0.01s Ux explodes to 2.51004e+08
    # and the Courant line reads "mean: 45.8817 max: 1250.73". No End marker,
    # but the blow-up must outrank 'incomplete'.
    analysis = convergence.parse_solver_log(str(FIXTURES / "diverged"))

    assert analysis.verdict == "diverged"
    assert analysis.completed is False
    evidence = "\n".join(analysis.evidence)
    assert "Ux" in evidence and "2.51004e+08" in evidence and "0.01" in evidence
    assert "Courant" in evidence and "1250.73" in evidence
    assert analysis.courant.max == pytest.approx(1250.73)
    assert analysis.courant.max_time == pytest.approx(0.01)


def test_fpe_trap_firing_is_diverged_but_the_banner_alone_is_not():
    # The fixture carries BOTH the harmless startup banner and a genuine trap
    # trace ("Foam::sigFpe::sigHandler", "Floating point exception (core
    # dumped)"). Only the latter may drive the verdict — per-line exclusion.
    analysis = convergence.parse_solver_log(str(FIXTURES / "fpe"))

    assert analysis.verdict == "diverged"
    evidence = "\n".join(analysis.evidence)
    assert "floating point exception" in evidence.lower()
    assert "Enabling floating point exception trapping" not in evidence


# ---------------------------------------------------------------------------
# Incomplete: a partial/in-flight log parses cleanly to progress
# ---------------------------------------------------------------------------

def test_partial_log_is_incomplete_with_progress():
    # Synthetic fixture: the real log truncated after the Time = 0.1s step.
    # controlDict still targets endTime 0.5.
    analysis = convergence.parse_solver_log(str(FIXTURES / "incomplete"))

    assert analysis.verdict == "incomplete"
    assert analysis.completed is False
    assert analysis.time.first_time == pytest.approx(0.005)
    assert analysis.time.latest_time == pytest.approx(0.1)
    assert analysis.time.end_time == pytest.approx(0.5)
    evidence = "\n".join(analysis.evidence)
    assert "0.1" in evidence and "0.5" in evidence  # names the progress


# ---------------------------------------------------------------------------
# Error: FOAM FATAL errors are extracted and drive the verdict
# ---------------------------------------------------------------------------

def test_fatal_error_is_extracted_and_drives_error_verdict():
    analysis = convergence.parse_solver_log(str(FIXTURES / "fatal"))

    assert analysis.verdict == "error"
    assert len(analysis.fatal_errors) == 1
    assert "FOAM FATAL IO ERROR" in analysis.fatal_errors[0]
    assert "div(phi,U) is undefined" in analysis.fatal_errors[0]
    evidence = "\n".join(analysis.evidence)
    assert "FOAM FATAL" in evidence


# ---------------------------------------------------------------------------
# The converged threshold, seen from the other side
# ---------------------------------------------------------------------------

# The genuine banner every OpenFOAM log opens with — inline synthetic logs
# must carry it too, so banner-safety is exercised everywhere (PR #37).
SIGFPE_BANNER = "sigFpe : Enabling floating point exception trapping (FOAM_SIGFPE).\n"


def _write_case(tmp_path, log_body, end_time="0.5"):
    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "controlDict").write_text(
        f"application     icoFoam;\nendTime         {end_time};\n"
    )
    (tmp_path / "log.icoFoam").write_text(SIGFPE_BANNER + log_body)


def test_completed_run_with_final_residuals_over_threshold_is_not_converged(tmp_path):
    # 0.025 is far above the built-in converged threshold (1e-4): completing
    # is not enough, so the verdict stays short of converged and says why.
    _write_case(tmp_path, (
        "Time = 0.5s\n\n"
        "smoothSolver:  Solving for Ux, Initial residual = 0.31, "
        "Final residual = 0.025, No Iterations 1000\n"
        "End\n"
    ))

    analysis = convergence.parse_solver_log(str(tmp_path))

    assert analysis.completed is True
    assert analysis.verdict == "incomplete"
    evidence = "\n".join(analysis.evidence)
    assert "Ux" in evidence and "0.025" in evidence and "0.0001" in evidence


def test_nan_residual_reads_as_divergence(tmp_path):
    _write_case(tmp_path, (
        "Time = 0.005s\n\n"
        "smoothSolver:  Solving for Ux, Initial residual = nan, "
        "Final residual = nan, No Iterations 1000\n"
    ))

    analysis = convergence.parse_solver_log(str(tmp_path))

    assert analysis.verdict == "diverged"
    assert "Ux" in "\n".join(analysis.evidence)
