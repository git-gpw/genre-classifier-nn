"""
Prepare GTZAN-like folder datasets (e.g. "Data 2/genres_original") for
spectrogram CNN training.

Outputs:
- metadata CSV with one row per track (track_id, genre, filepath, source)
- train/val/test CSV splits at track level (stratified by genre)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


VALID_AUDIO_EXTS = {".wav", ".mp3", ".au", ".flac", ".ogg", ".m4a", ".aac"}


def _is_audio_file(path: Path) -> bool:
    # macOS sidecar files "._foo.wav" should be ignored.
    return path.is_file() and path.suffix.lower() in VALID_AUDIO_EXTS and not path.name.startswith("._")


def build_metadata_csv(
    data_root: str,
    out_csv: str,
    source_label: str = "data2",
) -> pd.DataFrame:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")

    rows: list[dict] = []
    for genre_dir in sorted([d for d in root.iterdir() if d.is_dir()], key=lambda p: p.name.lower()):
        genre = genre_dir.name.strip().lower()
        audio_files = sorted([f for f in genre_dir.iterdir() if _is_audio_file(f)], key=lambda p: p.name.lower())

        for f in audio_files:
            stem = f.stem
            # Stable unique track id. Includes source + genre to avoid collisions.
            track_id = f"{source_label}__{genre}__{stem}"
            rows.append(
                {
                    "track_id": track_id,
                    "artist": source_label,
                    "title": stem,
                    "album": source_label,
                    "genre": genre,
                    "filepath": str(f.resolve()),
                    "source": source_label,
                }
            )

    if not rows:
        raise RuntimeError(f"No audio files found under {root}")

    df = pd.DataFrame(rows).drop_duplicates(subset=["track_id"]).reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def split_metadata(
    metadata_csv: str,
    out_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(metadata_csv)
    if "genre" not in df.columns or "track_id" not in df.columns:
        raise RuntimeError("metadata_csv must contain at least: track_id, genre")

    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0,1)")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0,1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("test ratio must be > 0")

    tracks = df["track_id"].astype(str).to_numpy()
    labels = df["genre"].astype(str).to_numpy()

    train_tracks, temp_tracks, train_labels, temp_labels = train_test_split(
        tracks,
        labels,
        test_size=(1.0 - train_ratio),
        stratify=labels,
        random_state=seed,
    )

    temp_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_tracks, test_tracks = train_test_split(
        temp_tracks,
        test_size=temp_test_ratio,
        stratify=temp_labels,
        random_state=seed,
    )

    train_set = set(train_tracks.tolist())
    val_set = set(val_tracks.tolist())
    test_set = set(test_tracks.tolist())

    train_df = df[df["track_id"].astype(str).isin(train_set)].reset_index(drop=True)
    val_df = df[df["track_id"].astype(str).isin(val_set)].reset_index(drop=True)
    test_df = df[df["track_id"].astype(str).isin(test_set)].reset_index(drop=True)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_path / "train.csv", index=False)
    val_df.to_csv(out_path / "val.csv", index=False)
    test_df.to_csv(out_path / "test.csv", index=False)

    return train_df, val_df, test_df


def _summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    total = len(all_df)

    def split_info(df: pd.DataFrame) -> dict:
        return {
            "rows": int(len(df)),
            "pct": float(len(df) / total) if total else 0.0,
            "genres": {k: int(v) for k, v in df["genre"].value_counts().to_dict().items()},
        }

    return {
        "rows_total": int(total),
        "genres_total": int(all_df["genre"].nunique()),
        "train": split_info(train_df),
        "val": split_info(val_df),
        "test": split_info(test_df),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare GTZAN/Data2 metadata + track-level splits for spectrogram CNN"
    )
    parser.add_argument(
        "--data_root",
        default="Data 2/genres_original",
        help="Folder with layout data_root/<genre>/*",
    )
    parser.add_argument(
        "--out_metadata_csv",
        default="data/processed/data2_metadata.csv",
    )
    parser.add_argument(
        "--out_splits_dir",
        default="data/splits/data2_spectrogram",
    )
    parser.add_argument("--source_label", default="data2")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report_json",
        default="data/processed/data2_prepare_report.json",
    )
    args = parser.parse_args()

    df = build_metadata_csv(
        data_root=args.data_root,
        out_csv=args.out_metadata_csv,
        source_label=args.source_label,
    )
    train_df, val_df, test_df = split_metadata(
        metadata_csv=args.out_metadata_csv,
        out_dir=args.out_splits_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    report = _summary(train_df, val_df, test_df)
    report.update(
        {
            "data_root": str(Path(args.data_root).resolve()),
            "out_metadata_csv": str(Path(args.out_metadata_csv).resolve()),
            "out_splits_dir": str(Path(args.out_splits_dir).resolve()),
            "source_label": args.source_label,
        }
    )

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
