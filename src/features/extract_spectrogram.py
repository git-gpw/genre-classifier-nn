"""
Extract fixed-size mel spectrogram segments from split CSVs and save NPZ caches.

Expected split CSV columns:
- track_id
- genre
- either filepath (absolute/relative source file) OR audio_dir + track_id resolution

Outputs for each split (train/val/test):
- X          : float16 or float32 array [n_segments, n_mels, n_frames, 1]
- y          : int32 labels [n_segments]
- track_ids  : object array [n_segments]
- genres     : object array [n_segments]
- class_names: object array [n_classes]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly, stft
from tqdm import tqdm


def resolve_audio_path(row: pd.Series, audio_dir: Path | None) -> Path | None:
    # Priority 1: explicit filepath in split CSV
    if "filepath" in row and not pd.isna(row["filepath"]):
        p = Path(str(row["filepath"]))
        if p.exists() and not p.name.startswith("._"):
            return p

    # Priority 2: lookup by track_id in processed audio directory
    if audio_dir is None:
        return None

    tid = str(row.get("track_id", "")).strip()
    if not tid:
        return None

    cands = [
        audio_dir / f"{tid}.mp3",
        audio_dir / f"{tid}.wav",
        audio_dir / tid[:3] / f"{tid}.mp3",
        audio_dir / tid[:3] / f"{tid}.wav",
    ]
    for p in cands:
        if p.exists():
            return p
    return None


def extract_segments(
    audio_path: Path,
    sampling_key: str | None = None,
    sr: int = 22050,
    segment_sec: float = 3.0,
    overlap: float = 0.5,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    max_segments_per_track: int = 0,
    sampling: str = "start",
    seed: int = 42,
) -> list[np.ndarray]:
    y, sr_ = _load_audio(audio_path, target_sr=sr)
    if y.size == 0:
        return []

    segment_samples = int(segment_sec * sr_)
    if segment_samples <= 0:
        raise ValueError("segment_sec must be > 0")

    hop_samples = int(segment_samples * (1.0 - overlap))
    hop_samples = max(1, hop_samples)

    max_start = len(y) - segment_samples
    if max_start < 0:
        return []
    n_positions = (max_start // hop_samples) + 1

    positions = np.arange(n_positions, dtype=np.int32) * hop_samples
    if max_segments_per_track > 0 and len(positions) > max_segments_per_track:
        if sampling == "start":
            keep = np.arange(max_segments_per_track, dtype=np.int32)
        elif sampling == "uniform":
            keep = np.linspace(0, len(positions) - 1, num=max_segments_per_track, dtype=np.int32)
            keep = np.unique(keep)
        elif sampling == "random":
            key = sampling_key if sampling_key is not None else str(audio_path)
            digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
            key_seed = int.from_bytes(digest, byteorder="little", signed=False)
            rng = np.random.default_rng((int(seed) + key_seed) % (2**32))
            keep = np.sort(rng.choice(len(positions), size=max_segments_per_track, replace=False))
        else:
            raise ValueError(f"Unsupported sampling='{sampling}'")
        positions = positions[keep]

    out: list[np.ndarray] = []
    for start in positions.tolist():
        chunk = y[start : start + segment_samples]
        mel = _mel_spectrogram_power(
            chunk,
            sr=sr_,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        mel_db = _power_to_db(mel)
        mel_min = float(np.min(mel_db))
        mel_max = float(np.max(mel_db))
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-6)
        out.append(mel_norm.astype(np.float32))

    return out


def _load_audio(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    sr = int(sr)

    if y.ndim == 2:
        y = y.mean(axis=1, dtype=np.float32)
    if y.size == 0:
        return np.zeros(target_sr, dtype=np.float32), target_sr

    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        y = resample_poly(y, up, down).astype(np.float32, copy=False)
        sr = target_sr
    return y, sr


def _stft_power(y: np.ndarray, sr: int, n_fft: int, hop_length: int) -> np.ndarray:
    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size))
    noverlap = max(0, n_fft - hop_length)
    _, _, zxx = stft(
        y,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
    )
    mag = np.abs(zxx).astype(np.float32, copy=False)
    return np.square(mag, dtype=np.float32)


def _hz_to_mel(freq_hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq_hz) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    n_freq = n_fft // 2 + 1
    fmax = float(fmax) if fmax is not None else sr / 2.0
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_points = _mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_freq - 1)

    fb = np.zeros((n_mels, n_freq), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center <= left:
            center = min(left + 1, n_freq - 1)
        if right <= center:
            right = min(center + 1, n_freq)
        left_span = max(center - left, 1)
        right_span = max(right - center, 1)
        for j in range(left, center):
            fb[i - 1, j] = (j - left) / left_span
        for j in range(center, right):
            fb[i - 1, j] = (right - j) / right_span
    return fb


def _mel_spectrogram_power(
    y: np.ndarray,
    sr: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    power = _stft_power(y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    mel_fb = _mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel = mel_fb @ power
    return np.asarray(mel, dtype=np.float32)


def _power_to_db(power: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, 1e-10))


def _split_df(split_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(split_csv)
    needed = {"track_id", "genre"}
    missing = sorted(list(needed - set(df.columns)))
    if missing:
        raise RuntimeError(f"{split_csv} missing required columns: {missing}")
    return df.drop_duplicates(subset=["track_id"]).reset_index(drop=True)


def _class_names_from_splits(split_dfs: dict[str, pd.DataFrame]) -> list[str]:
    genres = sorted(
        set(
            g.strip().lower()
            for df in split_dfs.values()
            for g in df["genre"].astype(str).tolist()
            if str(g).strip()
        )
    )
    if not genres:
        raise RuntimeError("No genres found in split CSVs")
    return genres


def build_npz_for_split(
    split_df: pd.DataFrame,
    split_name: str,
    out_npz: Path,
    class_names: list[str],
    *,
    audio_dir: Path | None,
    sr: int,
    segment_sec: float,
    overlap: float,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    max_segments_per_track: int,
    sampling: str,
    seed: int,
    out_dtype: str,
) -> dict:
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    tid_rows: list[str] = []
    genre_rows: list[str] = []
    missing_audio = 0
    skipped_genre = 0
    tracks_used = 0

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"spectrogram:{split_name}"):
        genre = str(row["genre"]).strip().lower()
        if genre not in class_to_idx:
            skipped_genre += 1
            continue

        path = resolve_audio_path(row, audio_dir)
        if path is None:
            missing_audio += 1
            continue

        try:
            segs = extract_segments(
                path,
                sampling_key=str(row.get("track_id", path.stem)),
                sr=sr,
                segment_sec=segment_sec,
                overlap=overlap,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                max_segments_per_track=max_segments_per_track,
                sampling=sampling,
                seed=seed,
            )
        except Exception:
            missing_audio += 1
            continue

        if not segs:
            missing_audio += 1
            continue

        tracks_used += 1
        tid = str(row["track_id"])
        y_idx = class_to_idx[genre]
        for seg in segs:
            X_rows.append(seg)
            y_rows.append(y_idx)
            tid_rows.append(tid)
            genre_rows.append(genre)

    if not X_rows:
        raise RuntimeError(f"No segments extracted for split={split_name}")

    X = np.stack(X_rows, axis=0)[..., np.newaxis]  # N x M x T x 1
    if out_dtype == "float16":
        X = X.astype(np.float16, copy=False)
    else:
        X = X.astype(np.float32, copy=False)

    y = np.asarray(y_rows, dtype=np.int32)
    track_ids = np.asarray(tid_rows, dtype=object)
    genres = np.asarray(genre_rows, dtype=object)
    class_names_arr = np.asarray(class_names, dtype=object)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        X=X,
        y=y,
        track_ids=track_ids,
        genres=genres,
        class_names=class_names_arr,
    )

    return {
        "split": split_name,
        "rows_tracks_input": int(len(split_df)),
        "rows_tracks_used": int(tracks_used),
        "rows_segments": int(len(X)),
        "missing_or_failed_audio": int(missing_audio),
        "skipped_unknown_genre": int(skipped_genre),
        "shape_X": list(X.shape),
        "dtype_X": str(X.dtype),
        "out_npz": str(out_npz.resolve()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract spectrogram NPZ caches from split CSVs")
    parser.add_argument("--splits_dir", required=True, help="Directory containing train.csv/val.csv/test.csv")
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for spectrogram_train.npz / spectrogram_val.npz / spectrogram_test.npz",
    )
    parser.add_argument(
        "--audio_dir",
        default="",
        help="Optional processed audio directory for track_id lookup when split CSV lacks filepath",
    )
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--segment_sec", type=float, default=3.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--max_segments_per_track", type=int, default=0)
    parser.add_argument(
        "--sampling",
        choices=["start", "uniform", "random"],
        default="start",
        help="How to choose windows when max_segments_per_track caps available windows",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Stored spectrogram dtype",
    )
    parser.add_argument("--report_json", default="")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    split_csvs = {
        "train": splits_dir / "train.csv",
        "val": splits_dir / "val.csv",
        "test": splits_dir / "test.csv",
    }
    missing = [name for name, p in split_csvs.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing split CSVs in {splits_dir}: {missing}")

    split_dfs = {name: _split_df(path) for name, path in split_csvs.items()}
    class_names = _class_names_from_splits(split_dfs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = Path(args.audio_dir) if args.audio_dir else None

    report: dict[str, object] = {
        "class_names": class_names,
        "config": {
            "sr": args.sr,
            "segment_sec": args.segment_sec,
            "overlap": args.overlap,
            "n_mels": args.n_mels,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "max_segments_per_track": args.max_segments_per_track,
            "sampling": args.sampling,
            "seed": args.seed,
            "out_dtype": args.out_dtype,
            "audio_dir": str(audio_dir.resolve()) if audio_dir is not None else None,
            "splits_dir": str(splits_dir.resolve()),
        },
        "splits": {},
    }

    for split_name in ("train", "val", "test"):
        out_npz = out_dir / f"spectrogram_{split_name}.npz"
        rep = build_npz_for_split(
            split_dfs[split_name],
            split_name,
            out_npz,
            class_names,
            audio_dir=audio_dir,
            sr=args.sr,
            segment_sec=args.segment_sec,
            overlap=args.overlap,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            max_segments_per_track=args.max_segments_per_track,
            sampling=args.sampling,
            seed=args.seed,
            out_dtype=args.out_dtype,
        )
        report["splits"][split_name] = rep

    report_json = Path(args.report_json) if args.report_json else (out_dir / "spectrogram_extract_report.json")
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
