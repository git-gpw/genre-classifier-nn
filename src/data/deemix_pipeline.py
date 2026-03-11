"""
CSV-driven deemix downloader utilities.

This module reads tracks from a CSV, downloads each track via `deemix`,
and writes an output CSV containing the detected local file path.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import time
import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd

AUDIO_EXTENSIONS = {".mp3", ".flac", ".m4a", ".wav"}


def _tokens(text: str) -> set[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    return {t for t in cleaned.split() if t}


def build_search_query(
    row: pd.Series,
    *,
    artist_col: str = "artists",
    title_col: str = "track_name",
) -> str:
    """Build a deemix search query from one CSV row."""
    title = str(row.get(title_col, "")).strip()
    artists_raw = str(row.get(artist_col, "")).strip()
    primary_artist = artists_raw.split(";")[0].strip()
    if primary_artist and title:
        return f"{primary_artist} - {title}"
    return title or primary_artist


def build_spotify_track_url(row: pd.Series, *, track_id_col: str = "track_id") -> str:
    """Build a Spotify track URL from the CSV track_id column."""
    track_id = str(row.get(track_id_col, "")).strip()
    if not track_id:
        return ""
    return f"https://open.spotify.com/track/{track_id}"


def build_deemix_command(
    query: str,
    *,
    output_dir: str | Path,
    deemix_binary: str | Iterable[str] = "deemix",
    output_flag: str = "--path",
    extra_args: Iterable[str] | None = None,
) -> list[str]:
    """
    Build the deemix CLI command for one query.

    Note: if your deemix install uses a different output flag, pass it via
    `output_flag` (for example `-p`).
    """
    if isinstance(deemix_binary, str):
        base_cmd = shlex.split(deemix_binary)
    else:
        base_cmd = list(deemix_binary)
    cmd = [*base_cmd, output_flag, str(output_dir), query]
    if extra_args:
        cmd.extend(list(extra_args))
    return cmd


def find_downloaded_file(
    output_dir: str | Path,
    *,
    track_name: str,
    artists: str,
) -> str:
    """
    Best-effort file matcher after download.

    Returns absolute path string when a likely match is found, otherwise "".
    """
    output_root = Path(output_dir)
    if not output_root.exists():
        return ""

    title_tokens = _tokens(track_name)
    artist_tokens = _tokens(artists.split(";")[0] if artists else "")
    required = {next(iter(title_tokens), ""), next(iter(artist_tokens), "")} - {""}

    candidates: list[Path] = []
    for path in output_root.rglob("*"):
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        haystack = _tokens(path.stem)
        if required.issubset(haystack):
            candidates.append(path)

    if not candidates:
        return ""
    best = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(best.resolve())


def _pick_staged_audio_file(staging_dir: Path) -> Path | None:
    """Return the most recent audio file in one staging directory."""
    candidates: list[Path] = []
    for path in staging_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _wait_for_staged_audio_file(
    staging_dir: Path,
    *,
    wait_sec: float = 5.0,
    poll_sec: float = 0.25,
) -> Path | None:
    """Poll staging directory briefly for delayed file visibility."""
    deadline = time.time() + wait_sec
    while time.time() < deadline:
        found = _pick_staged_audio_file(staging_dir)
        if found is not None:
            return found
        time.sleep(poll_sec)
    return _pick_staged_audio_file(staging_dir)


def _unique_destination_path(final_dir: Path, filename: str) -> Path:
    """Avoid accidental overwrite if two rows resolve to the same filename."""
    dest = final_dir / filename
    if not dest.exists():
        return dest
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    n = 2
    while True:
        candidate = final_dir / f"{stem} ({n}){suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def _find_existing_by_track_id(output_dir: Path, track_id: str) -> str:
    """Return existing downloaded file path for an exact track_id basename."""
    if not track_id:
        return ""
    for ext in AUDIO_EXTENSIONS:
        candidate = output_dir / f"{track_id}{ext}"
        if candidate.exists():
            return str(candidate.resolve())
    return ""


def _sha256_file(path: Path) -> str:
    """Compute SHA256 for immutable file identity tracking."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_tracks_from_csv(
    csv_in: str | Path,
    *,
    csv_out: str | Path,
    output_dir: str | Path,
    artist_col: str = "artists",
    title_col: str = "track_name",
    track_id_col: str = "track_id",
    target: str = "spotify_url",
    deemix_binary: str | Iterable[str] = "deemix",
    output_flag: str = "--path",
    extra_args: Iterable[str] | None = None,
    timeout_sec: int = 90,
    limit: int | None = None,
    dry_run: bool = False,
    verbose: bool = True,
    save_every: int = 25,
    dedupe_on: str = "track_id",
    post_download_wait_sec: float = 5.0,
    check_existing: bool = True,
) -> pd.DataFrame:
    """
    Download tracks listed in a CSV and write output CSV with file paths.

    Adds/overwrites output columns:
      - deemix_query
      - deemix_command
      - deemix_status
      - deemix_returncode
      - download_filepath
      - duplicate_of_row
      - immutable_key
      - file_sha256
    """
    input_df = pd.read_csv(csv_in).copy()
    if limit is not None:
        input_df = input_df.head(limit).copy()

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    total = len(input_df)
    results: list[dict[str, object]] = []
    staging_root = output_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    seen_keys: dict[str, int] = {}
    for i, (_, row) in enumerate(input_df.iterrows(), start=1):
        if dedupe_on == "track_id":
            key = str(row.get(track_id_col, "")).strip().lower()
        elif dedupe_on == "artist_title":
            key = build_search_query(
                row, artist_col=artist_col, title_col=title_col
            ).strip().lower()
        else:
            raise ValueError("dedupe_on must be 'track_id' or 'artist_title'")

        duplicate_of = ""
        if key:
            if key in seen_keys:
                duplicate_of = str(seen_keys[key])
            else:
                seen_keys[key] = i

        if target == "spotify_url":
            query = build_spotify_track_url(row, track_id_col=track_id_col)
        elif target == "search":
            query = build_search_query(row, artist_col=artist_col, title_col=title_col)
        else:
            raise ValueError("target must be 'spotify_url' or 'search'")
        row_track_id = str(row.get(track_id_col, "")).strip() or f"row{i}"
        staging_dir = staging_root / f"{i:06d}_{row_track_id}"
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_deemix_command(
            query,
            output_dir=staging_dir,
            deemix_binary=deemix_binary,
            output_flag=output_flag,
            extra_args=extra_args,
        )

        status = "dry_run"
        returncode = None
        existing_filepath = ""
        if check_existing:
            existing_filepath = _find_existing_by_track_id(output_root, row_track_id)
        if not query:
            status = "skipped_empty_query"
        elif existing_filepath:
            status = "already_downloaded"
        elif duplicate_of:
            status = "skipped_duplicate"
        elif not dry_run:
            try:
                completed = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                )
                returncode = completed.returncode
                status = "downloaded" if completed.returncode == 0 else "deemix_error"
            except subprocess.TimeoutExpired:
                status = "timeout"

        filepath = existing_filepath
        file_sha256 = ""
        if status == "downloaded":
            staged_file = _wait_for_staged_audio_file(
                staging_dir, wait_sec=post_download_wait_sec
            )
            if staged_file is None:
                status = "downloaded_no_file"
            else:
                if row_track_id.startswith("row"):
                    dest = _unique_destination_path(output_root, staged_file.name)
                else:
                    dest = output_root / f"{row_track_id}{staged_file.suffix.lower()}"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(staged_file), str(dest))
                filepath = str(dest.resolve())
                file_sha256 = _sha256_file(dest)
        if status == "already_downloaded" and filepath:
            file_sha256 = _sha256_file(Path(filepath))
        if status == "dry_run":
            filepath = ""
        shutil.rmtree(staging_dir, ignore_errors=True)

        result = row.to_dict()
        result["deemix_query"] = query
        result["deemix_command"] = " ".join(cmd)
        result["deemix_status"] = status
        result["deemix_returncode"] = returncode
        result["download_filepath"] = filepath
        result["duplicate_of_row"] = duplicate_of
        result["immutable_key"] = row_track_id if row_track_id else ""
        result["file_sha256"] = file_sha256
        results.append(result)

        if verbose:
            print(f"[{i}/{total}] {status}: {query}")
        if save_every > 0 and (i % save_every == 0 or i == total):
            pd.DataFrame(results).to_csv(csv_out, index=False)

    out_df = pd.DataFrame(results)
    out_df.to_csv(csv_out, index=False)
    return out_df
