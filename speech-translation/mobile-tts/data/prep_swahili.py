"""
prep_swahili.py — Filter Swahili TTS metadata to speaker 7 and write absolute-path CSVs.

Usage (from project root):
    python3 data/prep_swahili.py

Outputs:
    data/spk7_train.csv
    data/spk7_test.csv
    data/spk7_eval.csv
"""

import os
import sys
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

METADATA_ROOT = "/home/vacl2/groups/grp_mtlab/nobackup/autodelete/african_tts/swahili"
AUDIO_ROOT    = "/home/vacl2/groups/grp_mtlab/nobackup/autodelete/african_tts/swahili/swahili_cleaned"

# Directory containing this script — used to build output paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SPLITS = {
    "train": {
        "in":  os.path.join(METADATA_ROOT, "metadata_train_clean.csv"),
        "out": os.path.join(SCRIPT_DIR, "spk7_train.csv"),
    },
    "test": {
        "in":  os.path.join(METADATA_ROOT, "metadata_test_clean.csv"),
        "out": os.path.join(SCRIPT_DIR, "spk7_test.csv"),
    },
    "eval": {
        "in":  os.path.join(METADATA_ROOT, "metadata_eval_clean.csv"),
        "out": os.path.join(SCRIPT_DIR, "spk7_eval.csv"),
    },
}

TARGET_SPEAKER = "7"
SECS_PER_CLIP  = 4  # assumed average clip duration for estimation


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_split(name: str, in_path: str, out_path: str) -> dict:
    """Filter one split to speaker 7, validate files, write output CSV."""

    if not os.path.isfile(in_path):
        print(f"[ERROR] Input CSV not found: {in_path}")
        sys.exit(1)

    df = pd.read_csv(in_path, sep="|", dtype=str)

    # Sanity-check expected columns
    required = {"audio_file", "text", "speaker_name"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        print(f"[ERROR] {in_path} is missing columns: {missing_cols}")
        sys.exit(1)

    total_before = len(df)

    # Filter to target speaker
    df = df[df["speaker_name"].str.strip() == TARGET_SPEAKER].copy()
    filtered_count = len(df)

    # Rewrite audio_file to absolute paths
    df["audio_file"] = df["audio_file"].apply(
        lambda fn: os.path.join(AUDIO_ROOT, fn.strip())
    )

    # Validate existence
    missing_mask = ~df["audio_file"].apply(os.path.isfile)
    missing_files = df[missing_mask]["audio_file"].tolist()
    for path in missing_files:
        print(f"  [WARN] Missing audio file: {path}")

    df = df[~missing_mask].copy()
    kept_count = len(df)

    # Write output (pipe-delimited, with header)
    df.to_csv(out_path, sep="|", index=False)

    return {
        "total_in":   total_before,
        "spk7":       filtered_count,
        "missing":    len(missing_files),
        "kept":       kept_count,
    }


def main():
    print(f"Metadata root : {METADATA_ROOT}")
    print(f"Audio root    : {AUDIO_ROOT}")
    print(f"Target speaker: {TARGET_SPEAKER}")
    print()

    grand_kept    = 0
    grand_missing = 0

    for split_name, paths in SPLITS.items():
        print(f"--- {split_name.upper()} ---")
        stats = process_split(split_name, paths["in"], paths["out"])

        duration_h = stats["kept"] * SECS_PER_CLIP / 3600
        grand_kept    += stats["kept"]
        grand_missing += stats["missing"]

        print(f"  Rows in CSV       : {stats['total_in']:>7,}")
        print(f"  Speaker-7 rows    : {stats['spk7']:>7,}")
        print(f"  Missing files     : {stats['missing']:>7,}")
        print(f"  Written to output : {stats['kept']:>7,}")
        print(f"  Est. duration     : {duration_h:.1f} h  ({stats['kept'] * SECS_PER_CLIP:,} s @ {SECS_PER_CLIP}s/clip)")
        print(f"  Output            : {paths['out']}")
        print()

    total_h = grand_kept * SECS_PER_CLIP / 3600
    print("=== TOTAL ===")
    print(f"  Clips written : {grand_kept:,}")
    print(f"  Missing files : {grand_missing:,}")
    print(f"  Est. duration : {total_h:.1f} h")


if __name__ == "__main__":
    main()
