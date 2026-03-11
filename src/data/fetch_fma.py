"""
Download tracks and metadata from the Free Music Archive (FMA).

FMA provides pre-built CSV metadata and direct MP3 downloads.
Dataset info: https://github.com/mdeff/fma

Subsets:
    small  — 8,000 tracks,  ~7.2 GB
    medium — 25,000 tracks, ~22 GB
"""

import zipfile
from pathlib import Path

import pandas as pd
import requests

FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
FMA_AUDIO_URLS = {
    "small": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
    "medium": "https://os.unil.cloud.switch.ch/fma/fma_medium.zip",
}


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Stream-download a file with progress reporting. Skips if already present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Already exists, skipping: {dest}")
        return
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print(f"\n  Saved to {dest}")


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    print(f"Extracting {zip_path.name} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)


def load_fma_metadata(raw_dir: Path) -> pd.DataFrame:
    """
    Parse FMA's tracks.csv into a normalized DataFrame.

    FMA CSVs use multi-level column headers; this flattens to single-level.
    Audio paths follow FMA's subdirectory convention: <first3digits>/<trackid>.mp3
    """
    tracks_path = raw_dir / "fma_metadata" / "tracks.csv"
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])

    df = pd.DataFrame()
    df["track_id"] = tracks.index.astype(str).str.zfill(6)
    df["title"] = tracks["track", "title"]
    df["artist"] = tracks["artist", "name"]
    df["album"] = tracks["album", "title"]
    df["duration_s"] = tracks["track", "duration"]
    df["genre"] = tracks["track", "genre_top"]
    df["genres_all"] = tracks["track", "genres_all"]
    df["license"] = tracks["track", "license"]
    df["fma_split"] = tracks["set", "split"]    # FMA's own train/val/test labels
    df["fma_subset"] = tracks["set", "subset"]  # small / medium / large / full

    # Construct expected audio path using FMA directory layout
    df["audio_path"] = df["track_id"].apply(
        lambda tid: str(raw_dir / f"fma_small" / tid[:3] / f"{tid}.mp3")
    )

    return df.dropna(subset=["genre"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download FMA dataset")
    parser.add_argument(
        "--subset",
        choices=["small", "medium"],
        default="small",
        help="Audio subset to download (small=8k tracks ~7GB, medium=25k tracks ~22GB)",
    )
    parser.add_argument("--out_dir", default="data/raw", help="Root directory for downloads")
    parser.add_argument(
        "--metadata_only",
        action="store_true",
        help="Download only metadata CSVs, skip audio",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # Metadata
    meta_zip = out_dir / "fma_metadata.zip"
    download_file(FMA_METADATA_URL, meta_zip)
    extract_zip(meta_zip, out_dir)

    # Audio
    if not args.metadata_only:
        audio_zip = out_dir / f"fma_{args.subset}.zip"
        download_file(FMA_AUDIO_URLS[args.subset], audio_zip)
        extract_zip(audio_zip, out_dir)

    # Parse and save normalized metadata CSV
    df = load_fma_metadata(out_dir)
    out_csv = out_dir / "metadata" / "fma_metadata.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} FMA tracks to {out_csv}")
    print(f"Genre distribution:\n{df['genre'].value_counts().to_string()}")
