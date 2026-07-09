#!/usr/bin/env python
"""Build the openfoam_tutorials_structure FAISS index.

Shared build logic (argparse, embeddings, save) lives in _faiss_build.py.
This file only defines how one <case_begin>...</case_end> segment maps to a
(page_content, metadata) pair. Only the bare <index> content is embedded.
"""

import re

from _faiss_build import build_cli, extract_field, tokenize


def parse(segment: str):
    full_content = segment.strip()

    index_match = re.search(r"<index>(.*?)</index>", segment, re.DOTALL)
    index_content = index_match.group(1).strip()

    case_name = extract_field("case name", index_content)
    case_domain = extract_field("case domain", index_content)
    case_category = extract_field("case category", index_content)
    case_solver = extract_field("case solver", index_content)
    case_directory_structure = re.search(
        r"<directory_structure>([\s\S]*?)</directory_structure>", full_content
    ).group(1)

    return tokenize(index_content), {
        "full_content": full_content,
        "case_name": case_name,
        "case_domain": case_domain,
        "case_category": case_category,
        "case_solver": case_solver,
        "dir_structure": case_directory_structure,
    }


if __name__ == "__main__":
    build_cli(
        raw_filename="openfoam_tutorials_structure.txt",
        begin_tag="case_begin",
        end_tag="case_end",
        parse_fn=parse,
        out_subdir="openfoam_tutorials_structure",
    )
