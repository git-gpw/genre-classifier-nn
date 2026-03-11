"""
Extract track-level audio embeddings using LAION-CLAP.

Default behavior extracts one 512-dim embedding per track using CLAP's
internal truncation/pooling. Optional window mode can extract multiple
embeddings per track (for data augmentation), stored as repeated track_ids.

On first run the pretrained checkpoint (~300 MB) is downloaded automatically.

Output: data/processed/embeddings/audio_embeddings.npz
  - 'track_ids'  : string array, shape (N,) (track_id may repeat for windows)
  - 'sample_ids' : string array, shape (N,) unique embedding sample ids
  - 'embeddings' : float32 array, shape (N, 512)

Supports incremental runs: if output exists, already saved sample_ids are skipped.

Usage:
  python -m src.features.extract_audio [options]
"""

import argparse
import hashlib
from pathlib import Path

import laion_clap
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_model() -> laion_clap.CLAP_Module:
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    return model


def resolve_audio_path(audio_dir: Path, track_id: str) -> Path | None:
    path = audio_dir / f"{track_id}.mp3"
    if path.exists():
        return path
    path = audio_dir / track_id[:3] / f"{track_id}.mp3"
    if path.exists():
        return path
    return None


def target_sample_ids(track_id: str, windows_per_track: int, window_sec: float) -> list[str]:
    if windows_per_track <= 1 and window_sec <= 0:
        # Keep legacy sample id format for compatibility with existing files.
        return [track_id]
    return [f"{track_id}__w{i:03d}" for i in range(windows_per_track)]


def track_seed(base_seed: int, track_id: str) -> int:
    digest = hashlib.sha1(f"{base_seed}:{track_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def build_random_windows(
    waveform: np.ndarray,
    sr: int,
    window_sec: float,
    num_windows: int,
    seed: int,
) -> list[np.ndarray]:
    window_len = max(1, int(round(window_sec * sr)))
    if waveform.ndim != 1:
        waveform = np.asarray(waveform).reshape(-1)
    waveform = waveform.astype(np.float32, copy=False)

    if len(waveform) == 0:
        return [np.zeros(window_len, dtype=np.float32) for _ in range(num_windows)]

    if len(waveform) <= window_len:
        padded = np.pad(waveform, (0, window_len - len(waveform)))
        return [padded.astype(np.float32, copy=False) for _ in range(num_windows)]

    max_start = len(waveform) - window_len
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max_start + 1, size=num_windows)
    return [waveform[s : s + window_len] for s in starts]


def load_existing_embeddings(
    out_path: Path,
) -> tuple[list[str], list[str], list[np.ndarray]]:
    track_ids: list[str] = []
    sample_ids: list[str] = []
    embs: list[np.ndarray] = []
    if out_path.exists():
        data = np.load(out_path, allow_pickle=True)
        track_ids = [str(x) for x in data["track_ids"]]
        embs = [x.astype(np.float32) for x in data["embeddings"]]
        if "sample_ids" in data:
            sample_ids = [str(x) for x in data["sample_ids"]]
        else:
            # Legacy files had exactly one embedding per track.
            sample_ids = track_ids.copy()
        print(f"Resuming: {len(sample_ids)} embeddings already saved")
    return track_ids, sample_ids, embs


def extract_embeddings(
    metadata_csv: str,
    audio_dir: str,
    out_file: str,
    batch_size: int = 32,
    window_sec: float = 0.0,
    windows_per_track: int = 1,
    window_seed: int = 42,
) -> None:
    """
    Extract CLAP embeddings for all tracks listed in a metadata CSV.

    Audio files are looked up at <audio_dir>/<track_id>.mp3 and, as a
    fallback, at the FMA subdirectory layout <audio_dir>/<track_id[:3]>/<track_id>.mp3.

    If window_sec > 0, each track is split into windows_per_track random windows
    and one embedding is extracted per window.
    If window_sec <= 0 and windows_per_track > 1, multiple random CLAP truncations
    are extracted from the full file.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if windows_per_track <= 0:
        raise ValueError("windows_per_track must be > 0")
    if window_sec < 0:
        raise ValueError("window_sec must be >= 0")

    df = pd.read_csv(metadata_csv)
    audio_dir = Path(audio_dir)
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load previously computed embeddings for incremental runs
    existing_track_ids, existing_sample_ids, existing_embs = load_existing_embeddings(
        out_path
    )
    already_done = set(existing_sample_ids)

    pending_full: list[tuple[str, str, Path]] = []
    pending_repeat_full: list[tuple[str, list[str], Path]] = []
    pending_windowed: list[tuple[str, list[str], Path]] = []
    missing_tracks = 0

    for _, row in df.iterrows():
        tid = str(row["track_id"])
        wanted_sample_ids = target_sample_ids(tid, windows_per_track, window_sec)
        missing_sample_ids = [sid for sid in wanted_sample_ids if sid not in already_done]
        if not missing_sample_ids:
            continue

        path = resolve_audio_path(audio_dir, tid)
        if path is None:
            missing_tracks += 1
            continue

        if window_sec > 0:
            pending_windowed.append((tid, missing_sample_ids, path))
        elif len(missing_sample_ids) == 1 and missing_sample_ids[0] == tid:
            pending_full.append((tid, missing_sample_ids[0], path))
        else:
            pending_repeat_full.append((tid, missing_sample_ids, path))

    total_pending = len(pending_full) + len(pending_repeat_full) + len(pending_windowed)
    if total_pending == 0:
        print(f"Nothing to do (missing_audio={missing_tracks})")
        return

    print(f"Loading CLAP model...")
    model = load_model()

    new_track_ids: list[str] = []
    new_sample_ids: list[str] = []
    new_embs: list[np.ndarray] = []

    if pending_full:
        for i in tqdm(range(0, len(pending_full), batch_size), desc="Extracting full-track"):
            batch = pending_full[i : i + batch_size]
            batch_tids = [tid for tid, _, _ in batch]
            batch_sids = [sid for _, sid, _ in batch]
            batch_paths = [str(path) for _, _, path in batch]

            try:
                embs = model.get_audio_embedding_from_filelist(
                    x=batch_paths, use_tensor=False
                )
                for tid, sid, emb in zip(batch_tids, batch_sids, embs):
                    new_track_ids.append(tid)
                    new_sample_ids.append(sid)
                    new_embs.append(emb.astype(np.float32))
            except Exception:
                # Fall back to per-file extraction so one bad file doesn't lose the batch.
                for tid, sid, path in batch:
                    try:
                        emb = model.get_audio_embedding_from_filelist(
                            x=[str(path)], use_tensor=False
                        )[0]
                        new_track_ids.append(tid)
                        new_sample_ids.append(sid)
                        new_embs.append(emb.astype(np.float32))
                    except Exception as e2:
                        print(f"  [skip] {path.name}: {e2}")

    if pending_repeat_full:
        for tid, sample_ids, path in tqdm(
            pending_repeat_full, desc="Extracting repeated full-track"
        ):
            try:
                embs = model.get_audio_embedding_from_filelist(
                    x=[str(path)] * len(sample_ids), use_tensor=False
                )
                for sid, emb in zip(sample_ids, embs):
                    new_track_ids.append(tid)
                    new_sample_ids.append(sid)
                    new_embs.append(emb.astype(np.float32))
            except Exception as e:
                print(f"  [skip] {path.name}: {e}")

    if pending_windowed:
        for tid, sample_ids, path in tqdm(pending_windowed, desc="Extracting windowed"):
            try:
                waveform, sr = librosa.load(str(path), sr=48000, mono=True)
                windows = build_random_windows(
                    waveform=waveform,
                    sr=sr,
                    window_sec=window_sec,
                    num_windows=len(sample_ids),
                    seed=track_seed(window_seed, tid),
                )
                embs = model.get_audio_embedding_from_data(x=windows, use_tensor=False)
                for sid, emb in zip(sample_ids, embs):
                    new_track_ids.append(tid)
                    new_sample_ids.append(sid)
                    new_embs.append(emb.astype(np.float32))
            except Exception as e:
                print(f"  [skip] {path.name}: {e}")

    all_track_ids = existing_track_ids + new_track_ids
    all_sample_ids = existing_sample_ids + new_sample_ids
    all_embs = existing_embs + new_embs

    np.savez_compressed(
        out_path,
        track_ids=np.array(all_track_ids),
        sample_ids=np.array(all_sample_ids),
        embeddings=np.array(all_embs, dtype=np.float32),
    )
    print(
        f"Saved {len(all_track_ids)} embeddings -> {out_path}  "
        f"(new={len(new_sample_ids)}, missing_audio={missing_tracks})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CLAP audio embeddings")
    parser.add_argument(
        "--metadata_csv",
        default="data/processed/metadata.csv",
        help="CSV with track_id column (output of clean_metadata.py)",
    )
    parser.add_argument(
        "--audio_dir",
        default="data/processed/audio",
        help="Directory containing converted MP3 files (output of convert_audio.py)",
    )
    parser.add_argument(
        "--out_file",
        default="data/processed/embeddings/audio_embeddings.npz",
        help="Output .npz path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of audio files to embed per CLAP forward pass",
    )
    parser.add_argument(
        "--window_sec",
        type=float,
        default=0.0,
        help=(
            "Window length in seconds. "
            "If >0, each track is split into random windows before embedding."
        ),
    )
    parser.add_argument(
        "--windows_per_track",
        type=int,
        default=1,
        help=(
            "Embeddings per track. If window_sec>0 this is windows per track; "
            "otherwise it is repeated random CLAP truncations of the full file."
        ),
    )
    parser.add_argument(
        "--window_seed",
        type=int,
        default=42,
        help="Random seed for window sampling (used when window_sec > 0)",
    )
    args = parser.parse_args()

    extract_embeddings(
        args.metadata_csv,
        args.audio_dir,
        args.out_file,
        batch_size=args.batch_size,
        window_sec=args.window_sec,
        windows_per_track=args.windows_per_track,
        window_seed=args.window_seed,
    )
