"""
Split the dataset into train / val / test sets (default 70 / 15 / 15).

Strategy:
- Grouped by artist: every track by an artist goes to exactly one split,
  preventing the model from learning artist -> genre shortcuts (leakage).
- Stratified by genre: artist assignment is done per-genre so that each
  split has a proportional genre distribution.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def make_splits(
    metadata_csv: str,
    out_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Write train.csv / val.csv / test.csv to out_dir and return the three DataFrames.

    Artists are assigned to splits as whole units to prevent leakage.
    Genre stratification is applied at the artist level using each artist's
    most common genre.
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(metadata_csv)

    # Map each artist to their dominant genre
    artist_genre: dict[str, str] = {}
    for artist, group in df.groupby("artist"):
        artist_genre[artist] = group["genre"].value_counts().idxmax()

    # Bucket artists by genre for stratified assignment
    genre_artists: dict[str, list] = defaultdict(list)
    for artist, genre in artist_genre.items():
        genre_artists[genre].append(artist)

    train_set, val_set, test_set = set(), set(), set()
    for genre, artists in genre_artists.items():
        artists = list(artists)
        rng.shuffle(artists)
        n = len(artists)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        train_set.update(artists[:n_train])
        val_set.update(artists[n_train : n_train + n_val])
        test_set.update(artists[n_train + n_val :])

    def assign_split(artist: str) -> str:
        if artist in val_set:
            return "val"
        if artist in test_set:
            return "test"
        return "train"

    df["split"] = df["artist"].apply(assign_split)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    splits = {}
    for name in ("train", "val", "test"):
        split_df = df[df["split"] == name].drop(columns=["split"])
        split_df.to_csv(out_path / f"{name}.csv", index=False)
        splits[name] = split_df

    # Summary
    total = len(df)
    for name, split_df in splits.items():
        pct = 100 * len(split_df) / total
        print(
            f"{name:>5}: {len(split_df):>5} tracks ({pct:.1f}%)  |"
            f"  {split_df['artist'].nunique()} artists  |"
            f"  {split_df['genre'].nunique()} genres"
        )

    return splits["train"], splits["val"], splits["test"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create artist-grouped train/val/test splits")
    parser.add_argument("--metadata_csv", default="data/processed/metadata.csv")
    parser.add_argument("--out_dir", default="data/splits")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    make_splits(
        args.metadata_csv,
        args.out_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
