"""
Batch-convert audio files to a standardized MP3 format using ffmpeg.

Standardization:
- Bitrate: 128 / 192 / 320 kbps (configurable)
- Sample rate: 22050 Hz (sufficient for genre classification)
- Channels: stereo (default) or mono
- ID3v2.3 tags preserved

Requires ffmpeg to be installed and on PATH.
"""

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

VALID_BITRATES = {128, 192, 320}


def convert_to_mp3(
    in_path: Path,
    out_path: Path,
    bitrate: int = 192,
    mono: bool = False,
    sample_rate: int = 22050,
) -> bool:
    """Convert a single audio file to MP3 via ffmpeg. Returns True on success."""
    if bitrate not in VALID_BITRATES:
        raise ValueError(f"bitrate must be one of {VALID_BITRATES}, got {bitrate}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return True  # already converted

    cmd = [
        "ffmpeg",
        "-i", str(in_path),
        "-vn",                       # strip video/art streams
        "-ar", str(sample_rate),
        "-ac", "1" if mono else "2",
        "-b:a", f"{bitrate}k",
        "-id3v2_version", "3",       # broad player compatibility
        "-loglevel", "error",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def batch_convert(
    metadata_csv: str,
    in_audio_dir: str,
    out_audio_dir: str,
    bitrate: int = 192,
    mono: bool = False,
    workers: int = 4,
) -> pd.DataFrame:
    """
    Convert all audio files referenced in a metadata CSV.

    Expects audio files at <in_audio_dir>/<track_id>.mp3 or, for FMA-style
    layout, <in_audio_dir>/<track_id[:3]>/<track_id>.mp3.

    Returns a DataFrame with per-track conversion status.
    """
    df = pd.read_csv(metadata_csv)
    in_dir = Path(in_audio_dir)
    out_dir = Path(out_audio_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    futures: dict = {}
    missing = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _, row in df.iterrows():
            tid = str(row["track_id"])
            # Support both flat and FMA subdirectory layouts
            in_path = in_dir / tid[:3] / f"{tid}.mp3"
            if not in_path.exists():
                in_path = in_dir / f"{tid}.mp3"
            if not in_path.exists():
                missing.append(tid)
                continue
            out_path = out_dir / f"{tid}.mp3"
            future = executor.submit(convert_to_mp3, in_path, out_path, bitrate, mono)
            futures[future] = tid

        results = [{"track_id": tid, "status": "missing_source"} for tid in missing]
        for future in as_completed(futures):
            tid = futures[future]
            success = future.result()
            results.append({"track_id": tid, "status": "ok" if success else "failed"})
            if not success:
                print(f"Conversion failed: {tid}")

    status_df = pd.DataFrame(results)
    ok = (status_df["status"] == "ok").sum()
    print(f"Converted {ok}/{len(df)} tracks  |  missing={len(missing)}  |  failed={(status_df['status']=='failed').sum()}")
    return status_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch-convert audio to standardized MP3")
    parser.add_argument("--metadata_csv", default="data/processed/metadata.csv")
    parser.add_argument("--in_dir", default="data/raw/audio")
    parser.add_argument("--out_dir", default="data/processed/audio")
    parser.add_argument("--bitrate", type=int, default=192, choices=[128, 192, 320])
    parser.add_argument("--mono", action="store_true", help="Convert to mono (default: stereo)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel conversion workers")
    args = parser.parse_args()

    batch_convert(
        args.metadata_csv,
        args.in_dir,
        args.out_dir,
        bitrate=args.bitrate,
        mono=args.mono,
        workers=args.workers,
    )
