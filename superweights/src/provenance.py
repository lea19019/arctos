"""Provenance helpers shared by the detector and the ablation runner.

`docs/research_standards.md` §20.3: a result that cannot be traced back to
the exact code that produced it is not a result. Both runners embed the
values below in their output JSON.
"""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def git_sha():
    """Current commit, with '-dirty' appended if the tree has edits.

    Returns None outside a git checkout rather than raising: a missing sha
    should not lose a finished run's results.
    """
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--porcelain"],
            text=True, stderr=subprocess.DEVNULL).strip()
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
