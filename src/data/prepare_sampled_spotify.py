"""
Prepare sampled Spotify metadata for modeling.

This script targets the schema used in sampled_spotify_tracks.csv and produces:
- a cleaned metadata CSV with canonical columns (artist/title/genre),
- optional artist-grouped train/val/test splits via src.data.make_splits.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from src.data.make_splits import make_splits


def _primary_artist(artists: str) -> str:
    if pd.isna(artists):
        return ""
    value = str(artists).strip()
    if not value:
        return ""
    # Source CSV uses ";" for collaborations. Keep first credit for group splits.
    return re.split(r"\s*;\s*", value, maxsplit=1)[0].strip()


def prepare_sampled_spotify(
    in_csv: str,
    out_csv: str,
    split_dir: str | None = None,
    min_duration_ms: int = 30_000,
    max_duration_ms: int = 900_000,
    min_tempo: float = 1.0,
    max_tempo: float = 260.0,
    drop_conflicting_track_ids: bool = True,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    report_json: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(in_csv)
    rows_in = len(df)

    # Drop index-like artifacts from CSV exports.
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Canonical columns expected by the existing pipeline.
    df["artist"] = df["artists"].apply(_primary_artist)
    df["title"] = df["track_name"]
    df["album"] = df["album_name"]
    df["genre"] = df["track_genre"].astype(str).str.strip().str.lower()

    # Coerce numeric fields used downstream.
    numeric_cols = [
        "popularity",
        "duration_ms",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "time_signature",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows missing required identifiers/labels.
    before_required = len(df)
    df = df.dropna(subset=["track_id", "artist", "title", "genre"])
    rows_dropped_required = before_required - len(df)

    # Remove cross-genre label conflicts for the same track_id.
    conflicting_track_ids = (
        df.groupby("track_id")["genre"].nunique().loc[lambda s: s > 1].index.tolist()
    )
    if drop_conflicting_track_ids and conflicting_track_ids:
        df = df[~df["track_id"].isin(conflicting_track_ids)]

    # Keep one row per track_id for remaining duplicates.
    before_dedupe = len(df)
    df = (
        df.sort_values(["track_id", "popularity"], ascending=[True, False])
        .drop_duplicates(subset=["track_id"], keep="first")
        .reset_index(drop=True)
    )
    rows_dropped_dedupe = before_dedupe - len(df)

    # Filter obvious metadata anomalies.
    before_filters = len(df)
    df = df[
        (df["duration_ms"] >= min_duration_ms)
        & (df["duration_ms"] <= max_duration_ms)
        & (df["tempo"] >= min_tempo)
        & (df["tempo"] <= max_tempo)
    ].reset_index(drop=True)
    rows_dropped_anomaly_filters = before_filters - len(df)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary = {
        "rows_in": int(rows_in),
        "rows_out": int(len(df)),
        "rows_dropped_required": int(rows_dropped_required),
        "conflicting_track_ids_removed": int(len(conflicting_track_ids))
        if drop_conflicting_track_ids
        else 0,
        "rows_dropped_dedupe": int(rows_dropped_dedupe),
        "rows_dropped_anomaly_filters": int(rows_dropped_anomaly_filters),
        "genres_out": int(df["genre"].nunique()),
        "class_counts": {k: int(v) for k, v in df["genre"].value_counts().to_dict().items()},
    }

    if split_dir:
        make_splits(
            metadata_csv=out_csv,
            out_dir=split_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

    if report_json:
        Path(report_json).parent.mkdir(parents=True, exist_ok=True)
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved cleaned metadata to: {out_csv}")
    if split_dir:
        print(f"Saved splits to: {split_dir}")
    if report_json:
        print(f"Saved report to: {report_json}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean sampled_spotify_tracks.csv and optionally create grouped splits."
    )
    parser.add_argument("--in_csv", default="sampled_spotify_tracks.csv")
    parser.add_argument("--out_csv", default="data/processed/sampled_spotify_metadata.csv")
    parser.add_argument("--split_dir", default="data/splits/sampled_spotify")
    parser.add_argument("--min_duration_ms", type=int, default=30_000)
    parser.add_argument("--max_duration_ms", type=int, default=900_000)
    parser.add_argument("--min_tempo", type=float, default=1.0)
    parser.add_argument("--max_tempo", type=float, default=260.0)
    parser.add_argument(
        "--keep_conflicting_track_ids",
        action="store_true",
        help="Keep track_ids that appear with more than one genre label",
    )
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_json", default="data/processed/sampled_spotify_report.json")
    args = parser.parse_args()

    prepare_sampled_spotify(
        in_csv=args.in_csv,
        out_csv=args.out_csv,
        split_dir=args.split_dir,
        min_duration_ms=args.min_duration_ms,
        max_duration_ms=args.max_duration_ms,
        min_tempo=args.min_tempo,
        max_tempo=args.max_tempo,
        drop_conflicting_track_ids=not args.keep_conflicting_track_ids,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        report_json=args.report_json,
    )
