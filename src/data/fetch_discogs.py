"""
Enrich track metadata with genres and styles from the Discogs API.

Discogs has higher genre specificity than Spotify (more nuanced subgenres).
Rate limit: ~60 requests/min — default delay of 1.0s respects this.

Requires environment variable:
    DISCOGS_USER_TOKEN
"""

import os
import time

import discogs_client
import pandas as pd
from pathlib import Path


def get_client() -> discogs_client.Client:
    token = os.environ.get("DISCOGS_USER_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing Discogs token. Set DISCOGS_USER_TOKEN in your environment."
        )
    return discogs_client.Client("MusicGenreClassifier/1.0", user_token=token)


def search_release(d: discogs_client.Client, artist: str, title: str) -> dict | None:
    """Search Discogs for a release and return genre/style/label info."""
    try:
        results = d.search(f"{artist} {title}", type="release")
        if not results.count:
            return None
        release = results[0]
        return {
            "discogs_genres": "|".join(release.genres or []),
            "discogs_styles": "|".join(release.styles or []),
            "discogs_year": getattr(release, "year", None),
            "discogs_label": release.labels[0].name if release.labels else None,
        }
    except Exception:
        return None


def enrich_with_discogs(in_csv: str, out_csv: str, delay: float = 1.0) -> pd.DataFrame:
    """
    Load a metadata CSV and add Discogs genre/style columns.

    Iterates row-by-row and writes results to out_csv when done.
    Respects Discogs rate limit via `delay` (seconds between requests).
    """
    df = pd.read_csv(in_csv)
    d = get_client()

    genres, styles, years, labels = [], [], [], []
    for i, (_, row) in enumerate(df.iterrows()):
        result = search_release(d, row["artist"], row["title"])
        if result:
            genres.append(result["discogs_genres"])
            styles.append(result["discogs_styles"])
            years.append(result["discogs_year"])
            labels.append(result["discogs_label"])
        else:
            genres.append(None)
            styles.append(None)
            years.append(None)
            labels.append(None)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(df)} tracks...")
        time.sleep(delay)

    df["discogs_genres"] = genres
    df["discogs_styles"] = styles
    df["discogs_year"] = years
    df["discogs_label"] = labels

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    matched = df["discogs_genres"].notna().sum()
    print(f"Enriched {matched}/{len(df)} tracks with Discogs data -> {out_csv}")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enrich metadata with Discogs genres")
    parser.add_argument("--in_csv", default="data/raw/metadata/spotify_raw.csv")
    parser.add_argument("--out_csv", default="data/raw/metadata/discogs_enriched.csv")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between API calls")
    args = parser.parse_args()

    enrich_with_discogs(args.in_csv, args.out_csv, delay=args.delay)
