"""
Prepare an additional folder-organized audio dataset for audio-model training.

Expected layout:
  extra_audio/
    <genre_a>/*.mp3
    <genre_b>/*.wav
    ...

Outputs:
- extra metadata CSV (track_id, artist, title, genre, source, ...)
- extra-only artist-grouped splits
- optional combined audio splits (base + extra) with source labels
- optional combined audio metadata for embedding extraction
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from src.data.convert_audio import convert_to_mp3
from src.data.make_splits import make_splits

SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma"}


def _is_hidden_path(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts) or path.name.startswith("._")


def _parse_artist_title(filename_stem: str) -> tuple[str, str]:
    stem = filename_stem.strip()
    if " - " in stem:
        artist, title = stem.split(" - ", 1)
        return artist.strip() or "unknown", title.strip() or stem
    return "unknown", stem or "unknown"


def _stable_extra_track_id(genre: str, relative_path: Path) -> str:
    digest = hashlib.sha1(f"{genre}|{relative_path.as_posix()}".encode("utf-8")).hexdigest()
    return f"extra_{digest[:16]}"


def scan_extra_audio(extra_audio_dir: str) -> pd.DataFrame:
    root = Path(extra_audio_dir)
    if not root.exists():
        raise FileNotFoundError(f"Extra audio directory does not exist: {root}")

    rows: list[dict] = []
    for genre_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if _is_hidden_path(genre_dir):
            continue
        genre = genre_dir.name.strip().lower()
        if not genre:
            continue

        for path in sorted(genre_dir.rglob("*")):
            if not path.is_file():
                continue
            if _is_hidden_path(path):
                continue
            if path.suffix.lower() not in SUPPORTED_AUDIO_EXTS:
                continue

            rel = path.relative_to(root)
            artist, title = _parse_artist_title(path.stem)
            rows.append(
                {
                    "track_id": _stable_extra_track_id(genre, rel),
                    "artist": artist,
                    "title": title,
                    "album": "",
                    "genre": genre,
                    "source": "extra",
                    "source_relpath": rel.as_posix(),
                    "source_filepath": str(path.resolve()),
                }
            )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError(f"No supported audio files found under: {root}")

    # Protect against rare hash collisions.
    if df["track_id"].duplicated().any():
        duplicates = df[df["track_id"].duplicated(keep=False)].copy()
        raise RuntimeError(
            "Generated duplicate extra track_ids; please rename colliding files:\n"
            f"{duplicates[['track_id', 'source_relpath']].to_string(index=False)}"
        )
    return df


def _convert_extra_audio(
    df: pd.DataFrame,
    processed_audio_dir: str,
    bitrate: int,
    mono: bool,
    sample_rate: int,
    workers: int,
) -> pd.DataFrame:
    out_dir = Path(processed_audio_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    futures = {}
    results = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for _, row in df.iterrows():
            tid = str(row["track_id"])
            in_path = Path(str(row["source_filepath"]))
            out_path = out_dir / f"{tid}.mp3"
            fut = executor.submit(
                convert_to_mp3,
                in_path,
                out_path,
                bitrate,
                mono,
                sample_rate,
            )
            futures[fut] = tid

        for fut in as_completed(futures):
            tid = futures[fut]
            try:
                ok = bool(fut.result())
            except Exception:
                ok = False
            results.append({"track_id": tid, "status": "ok" if ok else "failed"})

    status_df = pd.DataFrame(results)
    if len(status_df) == 0:
        raise RuntimeError("No conversion jobs were scheduled for extra audio.")
    return status_df


def _load_if_exists(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def _ensure_source_column(df: pd.DataFrame, default_label: str) -> pd.DataFrame:
    out = df.copy()
    if len(out) == 0:
        return out
    if "source" not in out.columns:
        out["source"] = default_label
    else:
        out["source"] = out["source"].fillna(default_label).astype(str)
    return out


def _combine_audio_splits(
    base_splits_dir: str,
    extra_splits_dir: str,
    out_audio_splits_dir: str,
    include_extra_val: bool = False,
    include_extra_test: bool = False,
) -> dict:
    base_dir = Path(base_splits_dir)
    extra_dir = Path(extra_splits_dir)
    out_dir = Path(out_audio_splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        base_df = _ensure_source_column(
            _load_if_exists(base_dir / f"{split}.csv"),
            default_label="base",
        )
        extra_df = _ensure_source_column(
            _load_if_exists(extra_dir / f"{split}.csv"),
            default_label="extra",
        )
        extra_df["source"] = "extra"

        include_extra = split == "train" or (
            split == "val" and include_extra_val
        ) or (split == "test" and include_extra_test)

        parts = [base_df]
        if include_extra and len(extra_df) > 0:
            parts.append(extra_df)
        combined = pd.concat(parts, ignore_index=True, sort=False)
        if len(combined) > 0 and {"track_id", "genre"}.issubset(combined.columns):
            combined = combined.drop_duplicates(subset=["track_id", "genre"], keep="first")
        combined.to_csv(out_dir / f"{split}.csv", index=False)

        source_counts = (
            combined["source"].value_counts().to_dict() if "source" in combined.columns else {}
        )
        summary[split] = {
            "rows": int(len(combined)),
            "source_counts": {str(k): int(v) for k, v in source_counts.items()},
        }
    return summary


def _combine_audio_metadata(
    base_metadata_csv: str,
    extra_metadata_csv: str,
    out_audio_metadata_csv: str,
) -> dict:
    base = _ensure_source_column(pd.read_csv(base_metadata_csv), default_label="base")
    base["source"] = "base"
    extra = _ensure_source_column(pd.read_csv(extra_metadata_csv), default_label="extra")
    extra["source"] = "extra"

    combined = pd.concat([base, extra], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["track_id"], keep="first")

    out_path = Path(out_audio_metadata_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    return {
        "rows": int(len(combined)),
        "source_counts": {str(k): int(v) for k, v in combined["source"].value_counts().to_dict().items()},
    }


def prepare_extra_audio_dataset(
    extra_audio_dir: str,
    processed_audio_dir: str,
    out_metadata_csv: str,
    out_extra_splits_dir: str,
    *,
    base_metadata_csv: str | None = None,
    base_splits_dir: str | None = None,
    out_audio_splits_dir: str | None = None,
    out_audio_metadata_csv: str | None = None,
    include_extra_val: bool = False,
    include_extra_test: bool = False,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    bitrate: int = 192,
    mono: bool = False,
    sample_rate: int = 22050,
    workers: int = 4,
    report_json: str | None = None,
) -> dict:
    scanned = scan_extra_audio(extra_audio_dir)

    # Keep only genres already present in base metadata if provided.
    dropped_genres = {}
    if base_metadata_csv:
        base_df = pd.read_csv(base_metadata_csv)
        if "genre" in base_df.columns:
            allowed = set(base_df["genre"].dropna().astype(str).str.strip().str.lower())
            before = len(scanned)
            scanned = scanned[scanned["genre"].isin(allowed)].reset_index(drop=True)
            after = len(scanned)
            if before != after:
                dropped = before - after
                dropped_genres = {"dropped_rows": int(dropped), "allowed_genres": sorted(allowed)}

    if len(scanned) == 0:
        raise RuntimeError("No extra tracks left after filtering.")

    convert_status = _convert_extra_audio(
        scanned,
        processed_audio_dir=processed_audio_dir,
        bitrate=bitrate,
        mono=mono,
        sample_rate=sample_rate,
        workers=workers,
    )
    ok_ids = set(convert_status.loc[convert_status["status"] == "ok", "track_id"].astype(str))
    usable = scanned[scanned["track_id"].isin(ok_ids)].copy().reset_index(drop=True)

    out_meta = Path(out_metadata_csv)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    usable.to_csv(out_meta, index=False)

    make_splits(
        metadata_csv=str(out_meta),
        out_dir=out_extra_splits_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    summary: dict = {
        "extra_scan_rows": int(len(scanned)),
        "extra_usable_rows": int(len(usable)),
        "extra_genres": {str(k): int(v) for k, v in usable["genre"].value_counts().to_dict().items()},
        "conversion_status": {str(k): int(v) for k, v in convert_status["status"].value_counts().to_dict().items()},
        "extra_metadata_csv": str(out_meta),
        "extra_splits_dir": str(Path(out_extra_splits_dir).resolve()),
        "dropped_genres_filter": dropped_genres,
    }

    if base_splits_dir and out_audio_splits_dir:
        summary["audio_combined_splits"] = _combine_audio_splits(
            base_splits_dir=base_splits_dir,
            extra_splits_dir=out_extra_splits_dir,
            out_audio_splits_dir=out_audio_splits_dir,
            include_extra_val=include_extra_val,
            include_extra_test=include_extra_test,
        )

    if base_metadata_csv and out_audio_metadata_csv:
        summary["audio_combined_metadata"] = _combine_audio_metadata(
            base_metadata_csv=base_metadata_csv,
            extra_metadata_csv=str(out_meta),
            out_audio_metadata_csv=out_audio_metadata_csv,
        )
        summary["audio_metadata_csv"] = str(Path(out_audio_metadata_csv).resolve())

    if report_json:
        rp = Path(report_json)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        summary["report_json"] = str(rp.resolve())

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare extra folder-based audio dataset")
    parser.add_argument("--extra_audio_dir", default="extra_audio")
    parser.add_argument("--processed_audio_dir", default="data/processed/audio")
    parser.add_argument("--out_metadata_csv", default="data/processed/extra_audio_metadata.csv")
    parser.add_argument("--out_extra_splits_dir", default="data/splits/extra_audio")
    parser.add_argument("--base_metadata_csv", default="data/processed/metadata.csv")
    parser.add_argument("--base_splits_dir", default="data/splits")
    parser.add_argument("--out_audio_splits_dir", default="data/splits/audio_augmented")
    parser.add_argument(
        "--out_audio_metadata_csv",
        default="data/processed/metadata_audio_augmented.csv",
    )
    parser.add_argument("--include_extra_val", action="store_true")
    parser.add_argument("--include_extra_test", action="store_true")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bitrate", type=int, default=192, choices=[128, 192, 320])
    parser.add_argument("--mono", action="store_true")
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--report_json",
        default="data/processed/extra_audio_prepare_report.json",
    )
    args = parser.parse_args()

    prepare_extra_audio_dataset(
        extra_audio_dir=args.extra_audio_dir,
        processed_audio_dir=args.processed_audio_dir,
        out_metadata_csv=args.out_metadata_csv,
        out_extra_splits_dir=args.out_extra_splits_dir,
        base_metadata_csv=args.base_metadata_csv,
        base_splits_dir=args.base_splits_dir,
        out_audio_splits_dir=args.out_audio_splits_dir,
        out_audio_metadata_csv=args.out_audio_metadata_csv,
        include_extra_val=args.include_extra_val,
        include_extra_test=args.include_extra_test,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        bitrate=args.bitrate,
        mono=args.mono,
        sample_rate=args.sample_rate,
        workers=args.workers,
        report_json=args.report_json,
    )
