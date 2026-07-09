#!/usr/bin/env python
"""Build the openfoam_allrun_scripts FAISS index.

Shared build logic (argparse, embeddings, save) lives in _faiss_build.py.
This file only defines how one <case_begin>...</case_end> segment maps to a
(page_content, metadata) pair. The embedded text deliberately keeps only the
case name + solver (allrun scripts are not sensitive to domain/category).
"""

import re

from _faiss_build import build_cli, extract_field, tokenize


def parse(segment: str):
    index_match = re.search(r"<index>(.*?)</index>", segment, re.DOTALL)
    if not index_match:
        return None
    index_content = index_match.group(0).strip()
    full_content = segment.strip()

    dir_match = re.search(r"<directory_structure>(.*?)</directory_structure>", segment, re.DOTALL)
    dir_structure = dir_match.group(0).strip() if dir_match else "Unknown"

    case_name = extract_field("case name", index_content)
    case_domain = extract_field("case domain", index_content)
    case_category = extract_field("case category", index_content)
    case_solver = extract_field("case solver", index_content)

    # allrun script content is not sensitive to case domain and category
    index_content = f"<index>\ncase name: {case_name}\ncase solver: {case_solver}\n</index>\n"

    script_match = re.search(r"<allrun_script>([\s\S]*?)</allrun_script>", full_content)
    case_allrun_script = script_match.group(1).strip() if script_match else "Unknown"

    return tokenize(index_content + dir_structure), {
        "full_content": full_content,
        "case_name": case_name,
        "case_domain": case_domain,
        "case_category": case_category,
        "case_solver": case_solver,
        "dir_structure": dir_structure,
        "allrun_script": case_allrun_script,
    }


if __name__ == "__main__":
    build_cli(
        raw_filename="openfoam_allrun_scripts.txt",
        begin_tag="case_begin",
        end_tag="case_end",
        parse_fn=parse,
        out_subdir="openfoam_allrun_scripts",
    )
