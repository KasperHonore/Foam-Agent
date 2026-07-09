#!/usr/bin/env python
"""Build the openfoam_command_help FAISS index.

Shared build logic (argparse, embeddings, save) lives in _faiss_build.py.
This file only defines how one <command_begin>...</command_end> segment maps
to a (page_content, metadata) pair.
"""

import re

from _faiss_build import build_cli, tokenize


def parse(segment: str):
    command = re.search(r"<command>(.*?)</command>", segment, re.DOTALL).group(1).strip()
    help_text = re.search(r"<help_text>(.*?)</help_text>", segment, re.DOTALL).group(1).strip()
    full_content = segment.strip()
    return tokenize(command), {
        "full_content": full_content,
        "command": command,
        "help_text": help_text,
    }


if __name__ == "__main__":
    build_cli(
        raw_filename="openfoam_command_help.txt",
        begin_tag="command_begin",
        end_tag="command_end",
        parse_fn=parse,
        out_subdir="openfoam_command_help",
    )
