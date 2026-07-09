"""Shared skeleton for the FAISS index builders.

The four ``faiss_*.py`` scripts used to be the same ~110-line program copied
four times (argparse, file read, segment regex, embedding-provider cascade,
``FAISS.from_documents`` + ``save_local``). Only three things actually differ
per index: the raw dump filename, the ``<begin>/<end>`` segment tags, and how
each segment maps to a ``(page_content, metadata)`` pair.

This module owns everything shared. Each builder now supplies only its
``parse_fn`` and calls :func:`build_cli`. ``init_database.py`` imports
:data:`INDEX_SPECS` and :func:`persist_dir` so the index set and the on-disk
path layout have exactly one definition.

``tokenize`` is imported from the runtime package (``src/text_utils.py``) so the
index-time and query-time tokenizers are literally the same function.

Heavy imports (langchain / embeddings) are deferred into the functions that use
them, so this module — and each builder's ``parse_fn`` — can be imported and
unit-tested without langchain installed.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import namedtuple
from pathlib import Path
from typing import Callable, Optional, Tuple

# Canonical tokenizer lives in the runtime package; share it so index-time and
# query-time normalization can never drift apart.
_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
from text_utils import tokenize  # noqa: E402

# A parse function turns one segment's inner text into (page_content, metadata),
# or returns None to skip the segment.
ParseResult = Optional[Tuple[str, dict]]
ParseFn = Callable[[str], ParseResult]

# One row per FAISS index — the single source of truth for the index set and
# the builder that produces it. Adding an index is one row + one builder file.
IndexSpec = namedtuple("IndexSpec", ["module", "out_subdir"])
INDEX_SPECS = [
    IndexSpec("faiss_command_help", "openfoam_command_help"),
    IndexSpec("faiss_allrun_scripts", "openfoam_allrun_scripts"),
    IndexSpec("faiss_tutorials_structure", "openfoam_tutorials_structure"),
    IndexSpec("faiss_tutorials_details", "openfoam_tutorials_details"),
]

# Embedding providers, as a map so the argparse choices and the factory share one
# enumeration. Values are lazy loaders (imports deferred to call time).
def _openai(model: str):
    from langchain_openai.embeddings import OpenAIEmbeddings
    return OpenAIEmbeddings(model=model)


def _huggingface(model: str):
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model)


def _ollama(model: str):
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=model)


_PROVIDERS = {
    "openai": _openai,
    "huggingface": _huggingface,
    "ollama": _ollama,
}
PROVIDER_CHOICES = list(_PROVIDERS)


def make_embeddings(provider: str, model: str):
    """Build an embeddings object for the given provider (single source of truth)."""
    try:
        loader = _PROVIDERS[provider]
    except KeyError:
        raise ValueError(f"Unknown provider: {provider}") from None
    return loader(model)


def extract_field(field_name: str, text: str) -> str:
    """Extract ``<field_name>: <value>`` from a block of text, or 'Unknown'."""
    match = re.search(fr"{field_name}:\s*(.*)", text)
    return match.group(1).strip() if match else "Unknown"


def _model_dir_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def persist_dir(database_path: str, model: str, out_subdir: str) -> str:
    """Return the on-disk FAISS index directory for an (model, index) pair.

    This is the ONE place the ``faiss/<model_dir>/<index>`` layout is defined,
    so builders (which write here) and init_database (which checks it) agree.
    """
    return os.path.join(database_path, "faiss", _model_dir_name(model), out_subdir)


def build_faiss_index(
    *,
    database_path: str,
    raw_filename: str,
    begin_tag: str,
    end_tag: str,
    parse_fn: ParseFn,
    out_subdir: str,
    provider: str,
    model: str,
) -> str:
    """Read a raw dump, parse it into Documents, embed, and persist a FAISS index.

    Returns the directory the index was saved to.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    raw_path = os.path.join(database_path, "raw", raw_filename)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"File not found: {raw_path}")
    with open(raw_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(fr"<{begin_tag}>(.*?)</{end_tag}>", re.DOTALL)
    matches = pattern.findall(content)
    if not matches:
        raise ValueError("No cases found in the input file. Please check the file content.")

    documents = []
    for match in matches:
        parsed = parse_fn(match)
        if parsed is None:
            continue
        page_content, metadata = parsed
        documents.append(Document(page_content=page_content, metadata=metadata))

    embeddings = make_embeddings(provider, model)
    vectordb = FAISS.from_documents(documents, embeddings)

    out_dir = persist_dir(database_path, model, out_subdir)
    vectordb.save_local(out_dir)
    print(f"{len(documents)} cases indexed successfully with metadata! Saved at: {out_dir}")
    return out_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process OpenFOAM case data and store embeddings in FAISS."
    )
    parser.add_argument(
        "--database_path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent),
        help="Path to the database directory (default: '../../')",
    )
    parser.add_argument(
        "--embedding_provider",
        type=str,
        default="openai",
        choices=PROVIDER_CHOICES,
        help="Embedding provider",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model name",
    )
    return parser.parse_args()


def build_cli(
    *,
    raw_filename: str,
    begin_tag: str,
    end_tag: str,
    parse_fn: ParseFn,
    out_subdir: str,
) -> None:
    """argparse entry point shared by every builder's ``__main__`` block."""
    args = _parse_args()
    print(f"Database path: {args.database_path}")
    print(f"Provider: {args.embedding_provider}, Model: {args.embedding_model}")
    build_faiss_index(
        database_path=args.database_path,
        raw_filename=raw_filename,
        begin_tag=begin_tag,
        end_tag=end_tag,
        parse_fn=parse_fn,
        out_subdir=out_subdir,
        provider=args.embedding_provider,
        model=args.embedding_model,
    )
