"""Unit tests for the force-coefficients parser and Key-result stamping (issue #55).

Three seams, all key-free (CI runs these with nothing but pytest):

- The PURE parser (src/forcecoeffs.py, parse_forcecoeffs_dat) over the
  committed fixture dat files, asserting on the typed output's fields —
  reference metadata, per-coefficient series stats, tail window, key result
  — with expected values computed INDEPENDENTLY from the fixture (column
  slices averaged by hand/one-off script), never via the parser's own
  arithmetic.
- Discovery/ambiguity/typed-error wiring (parse_force_coefficients) over
  postProcessing directory trees staged in tmp_path from the fixtures.
- Key-result stamping at the temp-runs-directory seam with a planted ledger
  row (prior art: test_ledger_note_unit.py), asserting on the ledger file
  as a reader would.

MCP registration for the tool lives in test_mcp_helpers.py (importorskip
pattern — fastmcp is not installed in CI).

Fixture provenance (tests/fixtures/forcecoeffs/):
- cavity/postProcessing/forceCoeffs1/0/forceCoeffs.dat is the REAL
  Foundation v10 forceCoeffs output from the live lid-driven-cavity
  shakedown run (101 samples, Time 0..0.5), byte-for-byte as harvested.
- variants/ are derived from it by slicing the original bytes on line
  boundaries: earlier.dat (genuine header + first 41 rows, Time 0..0.2 —
  an older time directory's content), truncated.dat (cut before the column
  header — the headerless shape), header_only.dat (full header, zero rows).
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import forcecoeffs  # noqa: E402
import mechanics  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "forcecoeffs"
REAL_DAT = FIXTURES / "cavity" / "postProcessing" / "forceCoeffs1" / "0" / "forceCoeffs.dat"


def _real_text() -> str:
    return REAL_DAT.read_text()


def _variant_text(name: str) -> str:
    return (FIXTURES / "variants" / name).read_text()


# ---------------------------------------------------------------------------
# The real cavity dat: reference metadata and series shape
# ---------------------------------------------------------------------------

def test_real_dat_reference_metadata_read_by_eye():
    # All values read from the fixture's header block by eye.
    analysis = forcecoeffs.parse_forcecoeffs_dat(_real_text())

    assert analysis.reference.mag_u_inf == pytest.approx(1.0)
    assert analysis.reference.l_ref == pytest.approx(0.1)
    assert analysis.reference.a_ref == pytest.approx(0.001)
    assert analysis.reference.lift_dir == pytest.approx([0.0, 1.0, 0.0])
    assert analysis.reference.drag_dir == pytest.approx([1.0, 0.0, 0.0])
    assert analysis.reference.pitch_axis == pytest.approx([0.0, 0.0, 1.0])
    assert analysis.reference.cofr == pytest.approx([0.05, 0.05, 0.005])


def test_real_dat_sample_count_and_time_span_read_by_eye():
    # The fixture holds 101 data rows, Time 0 .. 0.5 (steps of 0.005).
    analysis = forcecoeffs.parse_forcecoeffs_dat(_real_text())

    assert analysis.samples == 101
    assert analysis.start_time == pytest.approx(0.0)
    assert analysis.end_time == pytest.approx(0.5)
    # Coefficient columns in the fixture's own header order.
    assert [c.name for c in analysis.coefficients] == \
        ["Cm", "Cd", "Cl", "Cl(f)", "Cl(r)"]


# ---------------------------------------------------------------------------
# Tail window and per-coefficient statistics (independently computed)
# ---------------------------------------------------------------------------

def test_real_dat_tail_window_is_last_21_of_101_samples():
    # 101 samples: ceil(0.2 * 101) = 21, above the floor of 10 — so the
    # window is the last 21 rows, Time 0.4 .. 0.5 (read from the fixture).
    analysis = forcecoeffs.parse_forcecoeffs_dat(_real_text())

    assert analysis.window.samples == 21
    assert analysis.window.fraction == pytest.approx(0.2)
    assert analysis.window.min_samples == 10
    assert analysis.window.start_time == pytest.approx(0.4)
    assert analysis.window.end_time == pytest.approx(0.5)


def test_real_dat_cd_and_cl_statistics_computed_independently():
    # First/final values read from the fixture's first and last rows by eye;
    # tail statistics computed independently over the last 21 rows of the
    # Cd and Cl columns (one-off sum/21 outside the parser).
    analysis = forcecoeffs.parse_forcecoeffs_dat(_real_text())
    by_name = {c.name: c for c in analysis.coefficients}

    cd = by_name["Cd"]
    assert cd.first == pytest.approx(-8.0)
    assert cd.final == pytest.approx(-2.500952)
    assert cd.tail_mean == pytest.approx(-2.5009528571428571, rel=1e-12)
    assert cd.tail_min == pytest.approx(-2.500954)
    assert cd.tail_max == pytest.approx(-2.500952)

    cl = by_name["Cl"]
    assert cl.first == pytest.approx(0.0)
    assert cl.final == pytest.approx(0.1274838)
    assert cl.tail_mean == pytest.approx(0.1274839, rel=1e-9)
    assert cl.tail_min == pytest.approx(0.1274837)
    assert cl.tail_max == pytest.approx(0.1274840)


def test_real_dat_cm_and_lift_split_statistics_computed_independently():
    # Same independent computation for the moment and the fore/aft lift split.
    analysis = forcecoeffs.parse_forcecoeffs_dat(_real_text())
    by_name = {c.name: c for c in analysis.coefficients}

    cm = by_name["Cm"]
    assert cm.first == pytest.approx(4.0)
    assert cm.final == pytest.approx(1.765343)
    assert cm.tail_mean == pytest.approx(1.7653434761904762, rel=1e-12)
    assert cm.tail_min == pytest.approx(1.765343)
    assert cm.tail_max == pytest.approx(1.765344)

    clf = by_name["Cl(f)"]
    assert clf.first == pytest.approx(4.0)
    assert clf.final == pytest.approx(1.829085)
    assert clf.tail_mean == pytest.approx(1.8290854761904762, rel=1e-12)

    clr = by_name["Cl(r)"]
    assert clr.first == pytest.approx(-4.0)
    assert clr.final == pytest.approx(-1.701601)
    assert clr.tail_mean == pytest.approx(-1.7016015238095238, rel=1e-12)


def test_window_floor_applies_to_short_series():
    # earlier.dat holds the first 41 rows (Time 0 .. 0.2): ceil(0.2*41) = 9
    # is under the floor, so the window is 10 samples — Time 0.155 .. 0.2
    # read from the fixture by eye.
    analysis = forcecoeffs.parse_forcecoeffs_dat(_variant_text("earlier.dat"))

    assert analysis.samples == 41
    assert analysis.end_time == pytest.approx(0.2)
    assert analysis.window.samples == 10
    assert analysis.window.start_time == pytest.approx(0.155)
    assert analysis.window.end_time == pytest.approx(0.2)


def test_window_is_all_samples_when_fewer_than_the_floor():
    # Derived in-test from the genuine shape: header plus only the first
    # 4 data rows — the window must be all 4 samples, never padded.
    lines = _real_text().splitlines()
    text = "\n".join(lines[:9] + lines[9:13])
    analysis = forcecoeffs.parse_forcecoeffs_dat(text)

    assert analysis.samples == 4
    assert analysis.window.samples == 4
    assert analysis.window.start_time == pytest.approx(0.0)
    assert analysis.window.end_time == pytest.approx(0.015)


# ---------------------------------------------------------------------------
# Key result: the compact summary destined for the ledger cell
# ---------------------------------------------------------------------------

def test_real_dat_key_result_is_the_documented_compact_format():
    # Documented format: 'Cd=<tail mean:.4g> Cl=<tail mean:.4g> (tail mean)'.
    # Expected literal built from the independently computed tail means
    # (-2.50095285714..., 0.1274839) formatted to 4 significant digits.
    analysis = forcecoeffs.parse_forcecoeffs_dat(_real_text())

    assert analysis.key_result == "Cd=-2.501 Cl=0.1275 (tail mean)"


def test_key_result_pure_parse_is_never_marked_stamped():
    # Stamping is the orchestrator's side effect; the pure parse only
    # prepares the summary.
    analysis = forcecoeffs.parse_forcecoeffs_dat(_real_text())
    assert analysis.stamped is False


# ---------------------------------------------------------------------------
# Typed errors: headerless/truncated file, zero data rows — never statistics
# ---------------------------------------------------------------------------

def test_headerless_file_raises_pointing_at_the_recipe():
    # truncated.dat is the real file cut before the '# Time ...' column
    # header — the shape of a truncated or foreign file.
    with pytest.raises(ValueError, match="forces reference"):
        forcecoeffs.parse_forcecoeffs_dat(_variant_text("truncated.dat"))


def test_header_only_file_with_zero_rows_raises_pointing_at_the_recipe():
    # header_only.dat carries the full genuine header block but no data
    # rows — statistics over nothing are never computed.
    with pytest.raises(ValueError, match="zero data rows"):
        forcecoeffs.parse_forcecoeffs_dat(_variant_text("header_only.dat"))


def test_empty_text_raises():
    with pytest.raises(ValueError, match="column header"):
        forcecoeffs.parse_forcecoeffs_dat("")


def test_partial_final_row_is_skipped_not_guessed():
    # An in-flight write can leave a partial last line; the parser must skip
    # it (100 full samples remain) rather than fabricate a value.
    text = _real_text().rstrip("\n")
    cut = text.rfind("\t")  # chop the final row's last column
    analysis = forcecoeffs.parse_forcecoeffs_dat(text[:cut])

    assert analysis.samples == 100
    assert analysis.end_time == pytest.approx(0.495)


# ---------------------------------------------------------------------------
# Discovery over staged postProcessing trees: newest time dir wins,
# ambiguity is a typed error naming candidates
# ---------------------------------------------------------------------------

def _stage_dat(case_dir: Path, function: str, time: str, content: str) -> None:
    time_dir = case_dir / "postProcessing" / function / time
    time_dir.mkdir(parents=True, exist_ok=True)
    (time_dir / "forceCoeffs.dat").write_text(content)


def test_single_function_object_is_discovered(tmp_path):
    case_dir = tmp_path / "cavity"
    _stage_dat(case_dir, "forceCoeffs1", "0", _real_text())

    analysis = forcecoeffs.parse_force_coefficients(str(case_dir))

    assert analysis.function_name == "forceCoeffs1"
    assert analysis.dat_file == "postProcessing/forceCoeffs1/0/forceCoeffs.dat"
    assert analysis.samples == 101
    assert analysis.stamped is False  # no runs root in sight, nothing stamped


def test_newest_time_directory_wins(tmp_path):
    # Restart layout: 0/ holds the older series (Time 0..0.2), 0.25/ the
    # continuation ending at 0.5 — the newest time directory must be parsed.
    case_dir = tmp_path / "cavity"
    _stage_dat(case_dir, "forceCoeffs1", "0", _variant_text("earlier.dat"))
    _stage_dat(case_dir, "forceCoeffs1", "0.25", _real_text())

    analysis = forcecoeffs.parse_force_coefficients(str(case_dir))

    assert analysis.dat_file == "postProcessing/forceCoeffs1/0.25/forceCoeffs.dat"
    assert analysis.end_time == pytest.approx(0.5)


def test_several_function_objects_without_a_name_raise_naming_candidates(tmp_path):
    case_dir = tmp_path / "cavity"
    _stage_dat(case_dir, "forceCoeffs1", "0", _real_text())
    _stage_dat(case_dir, "airfoilCoeffs", "0", _variant_text("earlier.dat"))

    with pytest.raises(forcecoeffs.ForceCoefficientsError) as excinfo:
        forcecoeffs.parse_force_coefficients(str(case_dir))
    message = str(excinfo.value)
    assert "airfoilCoeffs" in message and "forceCoeffs1" in message
    assert "function_name" in message


def test_explicit_function_name_disambiguates(tmp_path):
    case_dir = tmp_path / "cavity"
    _stage_dat(case_dir, "forceCoeffs1", "0", _real_text())
    _stage_dat(case_dir, "airfoilCoeffs", "0", _variant_text("earlier.dat"))

    analysis = forcecoeffs.parse_force_coefficients(
        str(case_dir), function_name="airfoilCoeffs")

    assert analysis.function_name == "airfoilCoeffs"
    assert analysis.dat_file == "postProcessing/airfoilCoeffs/0/forceCoeffs.dat"
    assert analysis.end_time == pytest.approx(0.2)


def test_unknown_function_name_raises_naming_what_exists(tmp_path):
    case_dir = tmp_path / "cavity"
    _stage_dat(case_dir, "forceCoeffs1", "0", _real_text())

    with pytest.raises(forcecoeffs.ForceCoefficientsError, match="forceCoeffs1"):
        forcecoeffs.parse_force_coefficients(str(case_dir), function_name="nope")


def test_no_postprocessing_directory_raises_pointing_at_the_recipe(tmp_path):
    case_dir = tmp_path / "cavity"
    case_dir.mkdir()

    with pytest.raises(forcecoeffs.ForceCoefficientsError, match="forces reference"):
        forcecoeffs.parse_force_coefficients(str(case_dir))


def test_postprocessing_without_forcecoeffs_output_raises(tmp_path):
    # A postProcessing tree from some other function object (no
    # forceCoeffs.dat anywhere) is 'no output', not a candidate.
    case_dir = tmp_path / "cavity"
    probe = case_dir / "postProcessing" / "probes" / "0"
    probe.mkdir(parents=True)
    (probe / "p").write_text("0\t1.0\n")

    with pytest.raises(forcecoeffs.ForceCoefficientsError, match="forces reference"):
        forcecoeffs.parse_force_coefficients(str(case_dir))


def test_missing_case_dir_raises_typed_error(tmp_path):
    with pytest.raises(forcecoeffs.ForceCoefficientsError, match="does not exist"):
        forcecoeffs.parse_force_coefficients(str(tmp_path / "nope"))


def test_non_forcecoeffs_neighbours_do_not_make_ambiguity(tmp_path):
    # Only function dirs actually holding a forceCoeffs.dat are candidates;
    # a probes/ directory beside the real one must not force function_name.
    case_dir = tmp_path / "cavity"
    _stage_dat(case_dir, "forceCoeffs1", "0", _real_text())
    probe = case_dir / "postProcessing" / "probes" / "0"
    probe.mkdir(parents=True)
    (probe / "p").write_text("0\t1.0\n")

    analysis = forcecoeffs.parse_force_coefficients(str(case_dir))
    assert analysis.function_name == "forceCoeffs1"


def test_headerless_discovered_file_raises_the_typed_error(tmp_path):
    # The discovery path wraps the parser's ValueError into the module's
    # typed error, still pointing at the recipe.
    case_dir = tmp_path / "cavity"
    _stage_dat(case_dir, "forceCoeffs1", "0", _variant_text("truncated.dat"))

    with pytest.raises(forcecoeffs.ForceCoefficientsError, match="forces reference"):
        forcecoeffs.parse_force_coefficients(str(case_dir))


# ---------------------------------------------------------------------------
# Key-result stamping at the temp-runs seam (prior art:
# test_ledger_note_unit.py) — assertions read the ledger file as a reader would
# ---------------------------------------------------------------------------

COLUMNS = ["id", "case", "created", "solver", "mesh", "status", "result",
           "key_result", "notes"]


def _rows(runs_root: Path) -> list:
    """Parse the data rows out of a ledger file, as a reader would."""
    text = (runs_root / "ledger.md").read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        if line.startswith("| ID") or set(line) <= {"|", "-", " "}:
            continue  # header / separator
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(dict(zip(COLUMNS, cells)))
    return rows


def _row(runs_root: Path, case: str) -> dict:
    return next(r for r in _rows(runs_root) if r["case"] == case)


# The 4-row derivation's expected Key result, computed independently:
# Cd tail mean = (-8.0 - 4.021036 - 2.956467 - 2.729087)/4 = -4.4266475,
# Cl tail mean = (0.0 + 1.363988 + 0.0285347 + 0.3354565)/4 = 0.4319948.
FOUR_ROW_KEY = "Cd=-4.427 Cl=0.432 (tail mean)"
REAL_KEY = "Cd=-2.501 Cl=0.1275 (tail mean)"


def _four_row_text() -> str:
    lines = _real_text().splitlines()
    return "\n".join(lines[:9] + lines[9:13]) + "\n"


def test_parse_stamps_key_result_into_the_planted_row(tmp_path):
    case_dir = mechanics.resolve_case_dir("cavity", run_directory=str(tmp_path))
    _stage_dat(Path(case_dir), "forceCoeffs1", "0", _real_text())
    before = _row(tmp_path, "cavity")
    assert before["key_result"] == "-"  # the placeholder, never yet filled

    analysis = forcecoeffs.parse_force_coefficients(
        case_dir, run_directory=str(tmp_path))

    assert analysis.stamped is True
    assert analysis.key_result == REAL_KEY
    after = _row(tmp_path, "cavity")
    assert after["key_result"] == REAL_KEY
    # every other cell is untouched — this is a one-cell machine write
    assert {k: v for k, v in after.items() if k != "key_result"} == \
           {k: v for k, v in before.items() if k != "key_result"}


def test_restamping_is_idempotent(tmp_path):
    case_dir = mechanics.resolve_case_dir("cavity", run_directory=str(tmp_path))
    _stage_dat(Path(case_dir), "forceCoeffs1", "0", _real_text())

    forcecoeffs.parse_force_coefficients(case_dir, run_directory=str(tmp_path))
    first = (tmp_path / "ledger.md").read_text(encoding="utf-8")
    forcecoeffs.parse_force_coefficients(case_dir, run_directory=str(tmp_path))

    assert (tmp_path / "ledger.md").read_text(encoding="utf-8") == first
    assert len(_rows(tmp_path)) == 1  # re-parsing never grows the table


def test_restamp_overwrites_with_the_new_summary(tmp_path):
    # A longer run produces new numbers; re-parsing must overwrite the
    # machine-owned cell, not preserve the stale one.
    case_dir = mechanics.resolve_case_dir("cavity", run_directory=str(tmp_path))
    _stage_dat(Path(case_dir), "forceCoeffs1", "0", _four_row_text())
    forcecoeffs.parse_force_coefficients(case_dir, run_directory=str(tmp_path))
    assert _row(tmp_path, "cavity")["key_result"] == FOUR_ROW_KEY

    _stage_dat(Path(case_dir), "forceCoeffs1", "0", _real_text())
    forcecoeffs.parse_force_coefficients(case_dir, run_directory=str(tmp_path))

    assert _row(tmp_path, "cavity")["key_result"] == REAL_KEY


def test_rowless_case_returns_analysis_unstamped(tmp_path):
    # In-tree but never tracked: no row exists, stamping never adopts one.
    case_dir = tmp_path / "cavity"
    _stage_dat(case_dir, "forceCoeffs1", "0", _real_text())

    analysis = forcecoeffs.parse_force_coefficients(
        str(case_dir), run_directory=str(tmp_path))

    assert analysis.stamped is False
    assert analysis.key_result == REAL_KEY  # the analysis is still complete
    assert not (tmp_path / "ledger.md").exists()  # nothing was written


def test_out_of_tree_case_returns_analysis_unstamped(tmp_path):
    # A case outside the runs root is not the ledger's to track (same
    # convention as the lifecycle) — analysis returned, nothing stamped.
    runs_root = tmp_path / "runs"
    mechanics.resolve_case_dir("other", run_directory=str(runs_root))
    case_dir = tmp_path / "elsewhere" / "cavity"
    _stage_dat(case_dir, "forceCoeffs1", "0", _real_text())
    before = (runs_root / "ledger.md").read_text(encoding="utf-8")

    analysis = forcecoeffs.parse_force_coefficients(
        str(case_dir), run_directory=str(runs_root))

    assert analysis.stamped is False
    assert (runs_root / "ledger.md").read_text(encoding="utf-8") == before
