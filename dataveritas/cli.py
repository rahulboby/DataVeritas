"""Command-line entry point for the DataVeritas Streamlit dashboard."""

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch the DataVeritas dashboard.",
        epilog="Streamlit options can be passed through, for example: dataveritas serve --server.port 8502",
    )
    parser.add_argument("command", nargs="?", choices=("serve",), default="serve")
    _, streamlit_args = parser.parse_known_args()

    dashboard = Path(__file__).resolve().parent.parent / "dashboard.py"
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(dashboard), *streamlit_args]
    )


if __name__ == "__main__":
    raise SystemExit(main())
