"""Unit tests for the shared FAISS builder skeleton (database/script/).

These run without langchain or any embeddings: they exercise only the pure
parse functions, the shared tokenizer, and the on-disk path layout. The point
is to lock the per-index behavior that used to live in four copy-pasted scripts,
so the de-duplication is provably behavior-preserving.
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "database" / "script"))

import mechanics  # noqa: E402
import text_utils  # noqa: E402
import _faiss_build  # noqa: E402
import faiss_allrun_scripts  # noqa: E402
import faiss_tutorials_structure  # noqa: E402
import faiss_tutorials_details  # noqa: E402
import faiss_command_help  # noqa: E402


_CASE_SEGMENT = """
<index>
case name: cavity
case domain: incompressible
case category: basic
case solver: icoFoam
</index>
<directory_structure>
<dir>0</dir>
<dir>system</dir>
</directory_structure>
<allrun_script>
blockMesh
icoFoam
</allrun_script>
<tutorials>
detailed tutorial text
</tutorials>
"""


# ---------------------------------------------------------------------------
# Single-source tokenizer: index-time and query-time must be the SAME function
# ---------------------------------------------------------------------------

def test_tokenize_is_single_source():
    assert mechanics.tokenize is text_utils.tokenize
    assert _faiss_build.tokenize is text_utils.tokenize


def test_tokenize_behavior():
    assert text_utils.tokenize("simpleFoam_pitzDaily") == "simple foam pitz daily"


# ---------------------------------------------------------------------------
# Per-index parse behavior (the quirks the four builders must each preserve)
# ---------------------------------------------------------------------------

def test_command_help_parse():
    segment = "<command>blockMesh</command><help_text>Usage: blockMesh [OPTIONS]</help_text>"
    page_content, meta = faiss_command_help.parse(segment)
    assert page_content == "block mesh"
    assert meta["command"] == "blockMesh"
    assert meta["help_text"] == "Usage: blockMesh [OPTIONS]"
    assert meta["full_content"] == segment.strip()


def test_allrun_parse_embeds_only_name_and_solver():
    page_content, meta = faiss_allrun_scripts.parse(_CASE_SEGMENT)
    assert meta["case_name"] == "cavity"
    assert meta["case_solver"] == "icoFoam"
    assert meta["allrun_script"] == "blockMesh\nicoFoam"
    # dir_structure metadata keeps the surrounding tags for this index
    assert meta["dir_structure"].startswith("<directory_structure>")
    # domain/category are intentionally dropped from the embedded text
    assert "incompressible" not in page_content
    assert "basic" not in page_content
    assert "cavity" in page_content and "ico foam" in page_content


def test_allrun_parse_skips_segment_without_index():
    assert faiss_allrun_scripts.parse("<allrun_script>x</allrun_script>") is None


def test_structure_parse_embeds_bare_index_no_tags():
    page_content, meta = faiss_tutorials_structure.parse(_CASE_SEGMENT)
    # structure embeds the bare <index> CONTENT, without the tags
    assert "directory structure" not in page_content
    assert "cavity" in page_content and "incompressible" in page_content
    # its dir_structure metadata is the inner content, without the tags
    assert not meta["dir_structure"].strip().startswith("<directory_structure>")
    assert "<dir>0</dir>" in meta["dir_structure"]


def test_details_parse_embeds_tags_and_keeps_tutorials():
    page_content, meta = faiss_tutorials_details.parse(_CASE_SEGMENT)
    # details embeds the tagged <index> + tagged <directory_structure>
    assert "index" in page_content and "directory structure" in page_content
    assert meta["tutorials"].strip() == "detailed tutorial text"
    # but its dir_structure metadata is the inner content only
    assert not meta["dir_structure"].strip().startswith("<directory_structure>")


# ---------------------------------------------------------------------------
# Path layout: the init_database bug regression
# ---------------------------------------------------------------------------

def test_persist_dir_includes_model_dir():
    got = _faiss_build.persist_dir("db", "Qwen/Qwen3-Embedding-0.6B", "openfoam_command_help")
    assert got == os.path.join("db", "faiss", "Qwen_Qwen3-Embedding-0.6B", "openfoam_command_help")
    # the model_dir segment (absent from the old init_database check) must be present
    assert "Qwen_Qwen3-Embedding-0.6B" in got


def test_index_specs_match_runtime_index_names():
    build_side = {spec.out_subdir for spec in _faiss_build.INDEX_SPECS}
    assert build_side == set(mechanics.FAISS_INDEX_NAMES)


# ---------------------------------------------------------------------------
# Embedding factory: one enumeration, honest error
# ---------------------------------------------------------------------------

def test_make_embeddings_rejects_unknown_provider():
    import pytest
    with pytest.raises(ValueError, match="Unknown provider"):
        _faiss_build.make_embeddings("bogus", "some-model")


def test_provider_choices_match_factory():
    assert set(_faiss_build.PROVIDER_CHOICES) == {"openai", "huggingface", "ollama"}
