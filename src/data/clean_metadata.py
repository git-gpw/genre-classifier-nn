"""
Clean and normalize the raw metadata CSV.

Steps:
- Normalize genre labels (e.g. "hip hop", "HipHop" -> "hip-hop")
- Clean text fields (lowercase, strip punctuation) for embedding
- Normalize and unify numeric fields
- Drop rare genres and duplicate tracks
- Assign unique track IDs if missing
"""

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# Canonical mapping for variant genre spellings
GENRE_ALIASES: dict[str, str] = {
    "hip hop": "hip-hop",
    "hiphop": "hip-hop",
    "hip_hop": "hip-hop",
    "r&b": "rnb",
    "r & b": "rnb",
    "rhythm and blues": "rnb",
    "rhythm & blues": "rnb",
    "electro": "electronic",
    "electronica": "electronic",
    "edm": "electronic",
    "dance": "electronic",
    "pop rock": "pop",
    "indie rock": "indie",
    "indie pop": "indie",
    "alternative rock": "alternative",
    "alt rock": "alternative",
    "classic rock": "rock",
    "folk rock": "folk",
    "heavy metal": "metal",
    "death metal": "metal",
    "black metal": "metal",
}


def normalize_genre(genre: str) -> str:
    genre = str(genre).lower().strip()
    genre = unicodedata.normalize("NFKD", genre)
    genre = re.sub(r"[^\w\s\-&]", "", genre)
    genre = re.sub(r"\s+", " ", genre).strip()
    return GENRE_ALIASES.get(genre, genre)


def clean_text(text: str) -> str:
    """Lowercase and strip punctuation; used for text fields before embedding."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_metadata(
    in_csv: str,
    out_csv: str,
    min_genre_count: int = 50,
    max_genres: int | None = None,
    drop_genres: list[str] | None = None,
    max_samples_per_genre: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    df = pd.read_csv(in_csv)
    print(f"Loaded {len(df)} rows from {in_csv}")

    # Resolve genre column across different source schemas
    if "discogs_genres" in df.columns:
        df["genre"] = df["discogs_genres"].str.split("|").str[0]
    elif "genre" not in df.columns and "genre_query" in df.columns:
        df["genre"] = df["genre_query"]

    # Normalize genre labels
    df["genre"] = df["genre"].apply(normalize_genre)

    # Drop missing genres
    before = len(df)
    df = df[df["genre"].notna() & (df["genre"].str.len() > 0)]
    print(f"Dropped {before - len(df)} rows with missing genre")

    # Drop genres with too few samples to train reliably
    counts = df["genre"].value_counts()
    valid_genres = counts[counts >= min_genre_count].index
    before = len(df)
    df = df[df["genre"].isin(valid_genres)]
    print(f"Dropped {before - len(df)} rows in genres with <{min_genre_count} samples")

    if drop_genres:
        drop_norm = {normalize_genre(g) for g in drop_genres if str(g).strip()}
        before = len(df)
        df = df[~df["genre"].isin(drop_norm)]
        print(f"Dropped {before - len(df)} rows from explicitly dropped genres: {sorted(drop_norm)}")

    if max_genres is not None and max_genres > 0:
        top_genres = df["genre"].value_counts().head(max_genres).index
        before = len(df)
        df = df[df["genre"].isin(top_genres)]
        print(f"Dropped {before - len(df)} rows outside top-{max_genres} genres")

    # Clean text fields for embedding
    for col in ["title", "artist", "album"]:
        if col in df.columns:
            df[f"{col}_clean"] = df[col].apply(clean_text)

    # Normalize numeric fields
    numeric_cols = [
        "duration_ms", "duration_s", "tempo", "loudness",
        "danceability", "energy", "valence", "popularity",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Unify duration to seconds
    if "duration_ms" in df.columns and "duration_s" not in df.columns:
        df["duration_s"] = df["duration_ms"] / 1000.0

    # Drop duplicate tracks (same artist + title)
    before = len(df)
    df = df.drop_duplicates(subset=["artist", "title"])
    print(f"Dropped {before - len(df)} duplicate tracks")

    if max_samples_per_genre is not None and max_samples_per_genre > 0:
        parts = []
        rng = np.random.default_rng(seed)
        before = len(df)
        for _, group in df.groupby("genre", sort=False):
            if len(group) > max_samples_per_genre:
                keep_idx = rng.choice(group.index.values, size=max_samples_per_genre, replace=False)
                group = group.loc[keep_idx]
            parts.append(group)
        df = pd.concat(parts, ignore_index=True)
        print(
            "Applied per-genre cap: "
            f"max_samples_per_genre={max_samples_per_genre} "
            f"(dropped={before - len(df)}, rows now={len(df)})"
        )

    # Assign stable unique IDs if not already present
    if "track_id" not in df.columns:
        df["track_id"] = [f"T{i:06d}" for i in range(len(df))]

    print(f"Remaining genres ({df['genre'].nunique()}): {sorted(df['genre'].unique())}")
    print("Class counts:", df["genre"].value_counts().to_dict())

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} tracks ({df['genre'].nunique()} genres) -> {out_csv}")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean and normalize metadata CSV")
    parser.add_argument("--in_csv", default="data/raw/metadata/discogs_enriched.csv")
    parser.add_argument("--out_csv", default="data/processed/metadata.csv")
    parser.add_argument(
        "--min_genre_count",
        type=int,
        default=50,
        help="Drop genres with fewer than this many tracks",
    )
    parser.add_argument(
        "--max_genres",
        type=int,
        default=0,
        help="Keep only the top-N most frequent genres after filtering (0 disables)",
    )
    parser.add_argument(
        "--drop_genres",
        default="",
        help="Comma-separated list of genres to drop after normalization",
    )
    parser.add_argument(
        "--max_samples_per_genre",
        type=int,
        default=0,
        help="Optional cap per genre to reduce class imbalance (0 disables)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    drop_genres = [g.strip() for g in args.drop_genres.split(",") if g.strip()]
    clean_metadata(
        in_csv=args.in_csv,
        out_csv=args.out_csv,
        min_genre_count=args.min_genre_count,
        max_genres=args.max_genres if args.max_genres > 0 else None,
        drop_genres=drop_genres,
        max_samples_per_genre=(
            args.max_samples_per_genre if args.max_samples_per_genre > 0 else None
        ),
        seed=args.seed,
    )
