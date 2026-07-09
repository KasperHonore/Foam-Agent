import os
import subprocess
import sys
import argparse
import shlex

# The FAISS index set and on-disk layout are defined once in the shared build
# module; import them so the skip-if-exists checks below match the real save
# location instead of re-encoding (and drifting from) the path scheme.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "database", "script"))
from _faiss_build import INDEX_SPECS, persist_dir  # noqa: E402

# Default to the key-free runtime default (see mechanics._embedding_settings):
# local HuggingFace embeddings, no API key required.
DEFAULT_EMBEDDING_PROVIDER = "huggingface"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize database for Foam-Agent project")
    parser.add_argument(
        '--openfoam_path',
        type=str,
        default=os.getenv("WM_PROJECT_DIR"),
        help="Path to OpenFOAM installation (WM_PROJECT_DIR)"
    )
    parser.add_argument(
        '--embedding_provider',
        type=str,
        default=DEFAULT_EMBEDDING_PROVIDER,
        help=f"Embedding provider for the FAISS indices (default: {DEFAULT_EMBEDDING_PROVIDER})"
    )
    parser.add_argument(
        '--embedding_model',
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model (default: {DEFAULT_EMBEDDING_MODEL})"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Force re-generate raw tutorial dumps + FAISS indices even if files exist"
    )
    return parser.parse_args()

def run_command(command_str):
    """
    Execute a command string using the current terminal's input/output,
    with the working directory set to the directory of the current file.

    Parameters:
        command_str (str): The command to execute, e.g. "python main.py --output_dir xxxx"
                           or "bash xxxxx.sh".
    """
    # Split the command string into a list of arguments
    args = shlex.split(command_str)
    # Set the working directory to the directory of the current file
    cwd = os.path.dirname(os.path.abspath(__file__))

    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            stdin=sys.stdin
        )
        print(f"Finished command: Return Code {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(e.returncode)

def main():
    args = parse_args()
    print(args)

    # Set environment variables
    WM_PROJECT_DIR = args.openfoam_path
    provider = args.embedding_provider
    model = args.embedding_model

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"script_dir: {script_dir}")

    database_dir = os.path.join(script_dir, "database")
    SCRIPTS = []

    # Preprocess the OpenFOAM tutorials (produces the raw/*.txt dumps)
    if args.force or not os.path.exists(os.path.join(database_dir, "raw", "openfoam_tutorials_details.txt")):
        SCRIPTS.append(f"python database/script/tutorial_parser.py --output_dir=./database/raw --wm_project_dir={WM_PROJECT_DIR}")

    # (Re)build FAISS indices. persist_dir() returns the exact directory each
    # builder writes to (faiss/<model_dir>/<index>), so the skip-if-exists check
    # matches the real save location for the chosen embedding model.
    for spec in INDEX_SPECS:
        if args.force or not os.path.exists(persist_dir(database_dir, model, spec.out_subdir)):
            SCRIPTS.append(
                f"python database/script/{spec.module}.py "
                f"--database_path=./database "
                f"--embedding_provider={provider} --embedding_model={model}"
            )

    if not SCRIPTS:
        print("All database files already exist. No initialization needed.")
        print("Tip: pass --force to rebuild.")
        return

    print("Starting database initialization...")
    for script in SCRIPTS:
        run_command(script)
    print("Database initialization completed successfully.")

if __name__ == "__main__":
    ## python init_database.py --openfoam_path $WM_PROJECT_DIR
    main()
