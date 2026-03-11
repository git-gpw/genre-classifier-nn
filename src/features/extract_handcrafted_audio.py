"""
Extract handcrafted audio features from track audio files.

Feature groups:
- MFCC (per-coefficient mean/std)
- Chroma STFT (per-bin mean/std)
- Spectral centroid (mean/std)
- Spectral rolloff (mean/std)
- Zero-crossing rate (mean/std)

Intended use:
1) extract to a CSV keyed by track_id
2) merge columns into metadata/split CSVs
3) train metadata model with these extra columns
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.fft import dct
from scipy.signal import resample_poly, stft
from tqdm import tqdm


def resolve_audio_path(audio_dir: Path, track_id: str) -> Path | None:
    path = audio_dir / f"{track_id}.mp3"
    if path.exists():
        return path
    path = audio_dir / track_id[:3] / f"{track_id}.mp3"
    if path.exists():
        return path
    return None


def _safe_mean_std(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return 0.0, 0.0
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 0.0
    return float(np.mean(x)), float(np.std(x))


def _add_vector_stats(row: dict, name: str, values: np.ndarray) -> None:
    mean, std = _safe_mean_std(values)
    row[f"{name}_mean"] = mean
    row[f"{name}_std"] = std


def _add_matrix_stats(row: dict, name: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape={values.shape}")
    for i in range(values.shape[0]):
        mean, std = _safe_mean_std(values[i])
        row[f"{name}_{i + 1:02d}_mean"] = mean
        row[f"{name}_{i + 1:02d}_std"] = std


def _zero_crossing_rate_numpy(
    y: np.ndarray,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.size < frame_length:
        y = np.pad(y, (0, frame_length - y.size))

    starts = np.arange(0, y.size - frame_length + 1, hop_length, dtype=np.int64)
    if starts.size == 0:
        starts = np.array([0], dtype=np.int64)

    zcr = np.empty(starts.size, dtype=np.float32)
    denom = float(max(frame_length - 1, 1))
    for i, start in enumerate(starts):
        frame = y[start : start + frame_length]
        signs = frame >= 0
        crossings = np.count_nonzero(signs[1:] != signs[:-1])
        zcr[i] = crossings / denom
    return zcr


def _hz_to_mel(freq_hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq_hz) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _load_audio(
    audio_path: Path,
    sample_rate: int,
    max_duration_sec: float,
) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(audio_path), always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    sr = int(sr)

    if y.ndim == 2:
        y = y.mean(axis=1, dtype=np.float32)

    if max_duration_sec > 0:
        max_samples = int(max_duration_sec * sr)
        if max_samples > 0:
            y = y[:max_samples]

    if y.size == 0:
        y = np.zeros(sample_rate, dtype=np.float32)
        return y, sample_rate

    if sr != sample_rate:
        gcd = math.gcd(sr, sample_rate)
        up = sample_rate // gcd
        down = sr // gcd
        y = resample_poly(y, up, down).astype(np.float32, copy=False)
        sr = sample_rate

    if y.size == 0:
        y = np.zeros(sample_rate, dtype=np.float32)
    return y, sr


def _stft_magnitude(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size))
    noverlap = max(0, n_fft - hop_length)
    freqs, _, zxx = stft(
        y,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
    )
    magnitude = np.abs(zxx).astype(np.float32, copy=False)
    power = np.square(magnitude, dtype=np.float32)
    return freqs.astype(np.float32, copy=False), magnitude, power


def _mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
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


def _mfcc_from_power(
    power: np.ndarray,
    sr: int,
    n_fft: int,
    n_mfcc: int,
) -> np.ndarray:
    n_mels = max(40, n_mfcc * 2)
    mel_fb = _mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_spec = mel_fb @ power
    log_mel = np.log(np.maximum(mel_spec, 1e-10))
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc]
    return mfcc.astype(np.float32, copy=False)


def _chroma_from_magnitude(magnitude: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    chroma = np.zeros((12, magnitude.shape[1]), dtype=np.float32)
    valid = freqs > 0.0
    if not np.any(valid):
        return chroma

    freq_valid = freqs[valid]
    bin_indices = np.where(valid)[0]
    midi = np.round(69.0 + 12.0 * np.log2(freq_valid / 440.0)).astype(np.int64)
    pitch_class = np.mod(midi, 12)

    for bin_idx, cls in zip(bin_indices, pitch_class):
        chroma[cls] += magnitude[bin_idx]

    frame_energy = chroma.sum(axis=0, keepdims=True)
    chroma = chroma / np.maximum(frame_energy, 1e-12)
    return chroma


def _spectral_centroid(magnitude: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    denom = magnitude.sum(axis=0)
    numer = (freqs[:, None] * magnitude).sum(axis=0)
    return numer / np.maximum(denom, 1e-12)


def _spectral_rolloff(power: np.ndarray, freqs: np.ndarray, roll_percent: float = 0.85) -> np.ndarray:
    cumulative = np.cumsum(power, axis=0)
    total = power.sum(axis=0)
    threshold = roll_percent * total
    idx = np.argmax(cumulative >= threshold[None, :], axis=0)
    idx = np.clip(idx, 0, len(freqs) - 1)
    return freqs[idx]


def extract_track_features(
    audio_path: Path,
    sample_rate: int = 22050,
    max_duration_sec: float = 120.0,
    n_mfcc: int = 20,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> dict:
    y, sr = _load_audio(
        audio_path=audio_path,
        sample_rate=sample_rate,
        max_duration_sec=max_duration_sec,
    )
    freqs, magnitude, power = _stft_magnitude(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)

    row: dict[str, float | int] = {}
    row["audio_seconds_used"] = float(y.size / sr)

    mfcc = _mfcc_from_power(power=power, sr=sr, n_fft=n_fft, n_mfcc=n_mfcc)
    chroma = _chroma_from_magnitude(magnitude=magnitude, freqs=freqs)
    spectral_centroid = _spectral_centroid(magnitude=magnitude, freqs=freqs)
    spectral_rolloff = _spectral_rolloff(power=power, freqs=freqs, roll_percent=0.85)
    zero_crossing_rate = _zero_crossing_rate_numpy(
        y=y,
        frame_length=n_fft,
        hop_length=hop_length,
    )

    _add_matrix_stats(row, "mfcc", mfcc)
    _add_matrix_stats(row, "chroma", chroma)
    _add_vector_stats(row, "spectral_centroid", spectral_centroid)
    _add_vector_stats(row, "spectral_rolloff", spectral_rolloff)
    _add_vector_stats(row, "zero_crossing_rate", zero_crossing_rate)
    return row


def extract_handcrafted_features(
    metadata_csv: str,
    audio_dir: str,
    out_csv: str,
    sample_rate: int = 22050,
    max_duration_sec: float = 120.0,
    n_mfcc: int = 20,
    n_fft: int = 2048,
    hop_length: int = 512,
    resume: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    if "track_id" not in df.columns:
        raise ValueError(f"{metadata_csv} is missing required column 'track_id'")

    track_ids = df["track_id"].astype(str).dropna().drop_duplicates().tolist()
    audio_root = Path(audio_dir)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_df = pd.DataFrame()
    done_ids: set[str] = set()
    if resume and out_path.exists():
        existing_df = pd.read_csv(out_path)
        if "track_id" in existing_df.columns:
            done_ids = set(existing_df["track_id"].astype(str))
            print(f"Resuming handcrafted extraction: {len(done_ids)} track_ids already saved")

    rows = []
    missing_audio = 0
    failed = 0
    skipped_done = 0

    for tid in tqdm(track_ids, desc="Extracting handcrafted"):
        if tid in done_ids:
            skipped_done += 1
            continue

        path = resolve_audio_path(audio_root, tid)
        if path is None:
            missing_audio += 1
            continue

        try:
            feats = extract_track_features(
                path,
                sample_rate=sample_rate,
                max_duration_sec=max_duration_sec,
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length,
            )
            feats["track_id"] = tid
            rows.append(feats)
        except Exception as exc:
            failed += 1
            print(f"  [skip] {path.name}: {exc}")

    new_df = pd.DataFrame(rows)
    merged_df = pd.concat([existing_df, new_df], ignore_index=True)
    if "track_id" in merged_df.columns:
        merged_df["track_id"] = merged_df["track_id"].astype(str)
        merged_df = merged_df.drop_duplicates(subset=["track_id"], keep="last")

    merged_df.to_csv(out_path, index=False)
    print(
        "Saved handcrafted features -> "
        f"{out_path} (rows={len(merged_df)}, new={len(new_df)}, "
        f"missing_audio={missing_audio}, failed={failed}, skipped_done={skipped_done})"
    )
    return merged_df


def merge_handcrafted_into_metadata(
    metadata_csv: str,
    features_csv: str,
    out_csv: str | None = None,
) -> pd.DataFrame:
    meta_path = Path(metadata_csv)
    out_path = Path(out_csv) if out_csv else meta_path

    meta = pd.read_csv(meta_path)
    try:
        feats = pd.read_csv(features_csv)
    except pd.errors.EmptyDataError:
        feats = pd.DataFrame(columns=["track_id"])
    if "track_id" not in meta.columns or "track_id" not in feats.columns:
        raise ValueError("Both CSV files must include 'track_id'")

    meta["track_id"] = meta["track_id"].astype(str)
    feats["track_id"] = feats["track_id"].astype(str)
    feats = feats.drop_duplicates(subset=["track_id"], keep="last")

    feature_cols = [c for c in feats.columns if c != "track_id"]
    existing_feature_cols = [c for c in feature_cols if c in meta.columns]
    if existing_feature_cols:
        meta = meta.drop(columns=existing_feature_cols)

    merged = meta.merge(feats, on="track_id", how="left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    match_count = int(merged[feature_cols[0]].notna().sum()) if feature_cols else 0
    print(
        f"Merged handcrafted features into {out_path} "
        f"(rows={len(merged)}, matched={match_count})"
    )
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract handcrafted audio features")
    parser.add_argument(
        "--metadata_csv",
        default="data/processed/metadata.csv",
        help="CSV with track_id column",
    )
    parser.add_argument(
        "--audio_dir",
        default="data/processed/audio",
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--out_csv",
        default="data/processed/features/handcrafted_audio_features.csv",
        help="Output CSV path for handcrafted features",
    )
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument(
        "--max_duration_sec",
        type=float,
        default=120.0,
        help="Max seconds loaded per track (<=0 uses full audio)",
    )
    parser.add_argument("--n_mfcc", type=int, default=20)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable resume mode and recompute all track features",
    )
    args = parser.parse_args()

    extract_handcrafted_features(
        metadata_csv=args.metadata_csv,
        audio_dir=args.audio_dir,
        out_csv=args.out_csv,
        sample_rate=args.sample_rate,
        max_duration_sec=args.max_duration_sec,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        resume=not args.no_resume,
    )
