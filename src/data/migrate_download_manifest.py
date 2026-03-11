"""
Migrate downloaded audio filenames to track_id-based names.

Reads a manifest CSV (for example sampled_spotify_tracks_with_paths.csv), renames
downloaded files to <track_id>.<ext>, and writes an updated CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def migrate_manifest_filenames(
    manifest_csv: str | Path,
    *,
    out_csv: str | Path | None = None,
    track_id_col: str = "track_id",
    filepath_col: str = "download_filepath",
    immutable_key_col: str = "immutable_key",
    hash_col: str = "file_sha256",
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Rename files referenced in manifest CSV to track_id-based filenames.

    Output/updated columns:
      - download_filepath
      - immutable_key
      - file_sha256
      - migration_status
    """
    df = pd.read_csv(manifest_csv).copy()

    statuses: list[str] = []
    new_paths: list[str] = []
    immutable_keys: list[str] = []
    hashes: list[str] = []

    for _, row in df.iterrows():
        track_id = str(row.get(track_id_col, "")).strip()
        src_str = str(row.get(filepath_col, "")).strip()

        if not track_id:
            statuses.append("missing_track_id")
            new_paths.append(src_str)
            immutable_keys.append("")
            hashes.append("")
            continue
        if not src_str or src_str.lower() == "nan":
            statuses.append("missing_filepath")
            new_paths.append("")
            immutable_keys.append(track_id)
            hashes.append("")
            continue

        src = Path(src_str)
        if not src.exists():
            statuses.append("source_not_found")
            new_paths.append(src_str)
            immutable_keys.append(track_id)
            hashes.append("")
            continue

        dest = src.parent / f"{track_id}{src.suffix.lower()}"
        if src.resolve() == dest.resolve():
            statuses.append("already_track_id")
            new_paths.append(str(dest.resolve()))
            immutable_keys.append(track_id)
            hashes.append(sha256_file(dest))
            continue

        if dest.exists():
            statuses.append("conflict_target_exists")
            new_paths.append(src_str)
            immutable_keys.append(track_id)
            hashes.append("")
            continue

        if dry_run:
            statuses.append("would_rename")
            new_paths.append(str(dest.resolve()))
            immutable_keys.append(track_id)
            hashes.append("")
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        statuses.append("renamed")
        new_paths.append(str(dest.resolve()))
        immutable_keys.append(track_id)
        hashes.append(sha256_file(dest))

    df[filepath_col] = new_paths
    df[immutable_key_col] = immutable_keys
    df[hash_col] = hashes
    df["migration_status"] = statuses

    output_path = Path(out_csv) if out_csv is not None else Path(manifest_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rename downloaded files to <track_id>.<ext> using a manifest CSV"
    )
    parser.add_argument("--manifest_csv", required=True, help="Input manifest CSV path")
    parser.add_argument(
        "--out_csv",
        default=None,
        help="Output CSV path (default: overwrite manifest_csv)",
    )
    parser.add_argument("--track_id_col", default="track_id")
    parser.add_argument("--filepath_col", default="download_filepath")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    out_df = migrate_manifest_filenames(
        args.manifest_csv,
        out_csv=args.out_csv,
        track_id_col=args.track_id_col,
        filepath_col=args.filepath_col,
        dry_run=args.dry_run,
    )
    print(out_df["migration_status"].value_counts(dropna=False).to_string())
