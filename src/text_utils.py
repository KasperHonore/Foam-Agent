"""Shared text utilities used at BOTH FAISS index-build time and query time.

``tokenize`` MUST be identical on both sides: the ``database/script`` builders
embed document text with it, and :func:`mechanics.retrieve_faiss` normalizes the
query with it before the similarity search. If the two ever diverged, retrieval
quality would silently degrade. Keeping it here, imported by both the runtime
(``src/mechanics.py``) and the offline builders
(``database/script/_faiss_build.py``), makes that impossible.
"""

from __future__ import annotations

import re


def tokenize(text: str) -> str:
    """Normalize text for embedding/query: split snake_case and camelCase, lowercase.

    e.g. ``"simpleFoam_pitzDaily"`` -> ``"simple foam pitz daily"``.
    """
    text = text.replace("_", " ")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    return text.lower()
