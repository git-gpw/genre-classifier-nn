"""
Fetch track metadata from the Spotify API.

Requires environment variables:
    SPOTIFY_CLIENT_ID
    SPOTIFY_CLIENT_SECRET
"""

import os
import time
from urllib.parse import urlparse

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


def get_client() -> spotipy.Spotify:
    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing Spotify credentials. Set SPOTIFY_CLIENT_ID and "
            "SPOTIFY_CLIENT_SECRET in your environment."
        )
    auth = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
    )
    return spotipy.Spotify(auth_manager=auth)


def fetch_track_metadata(track_ids: list[str], batch_size: int = 50) -> pd.DataFrame:
    """Fetch basic metadata for a list of Spotify track IDs (max 50 per call)."""
    sp = get_client()
    records = []
    for i in range(0, len(track_ids), batch_size):
        batch = track_ids[i : i + batch_size]
        tracks = sp.tracks(batch)["tracks"]
        for t in tracks:
            if t is None:
                continue
            records.append({
                "track_id": t["id"],
                "title": t["name"],
                "artist": t["artists"][0]["name"],
                "album": t["album"]["name"],
                "release_year": t["album"]["release_date"][:4],
                "duration_ms": t["duration_ms"],
                "popularity": t["popularity"],
            })
        time.sleep(0.1)
    return pd.DataFrame(records)


def parse_playlist_id(url: str) -> str:
    """
    Extract the playlist ID from a Spotify playlist URL.

    Accepts both full URLs and bare IDs:
        https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M
        https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M?si=abc123
        37i9dQZF1DXcBWIGoYBM5M
    """
    parsed = urlparse(url)
    if parsed.scheme:  # it's a URL
        parts = parsed.path.strip("/").split("/")
        if len(parts) < 2 or parts[-2] != "playlist":
            raise ValueError(f"Not a valid Spotify playlist URL: {url}")
        return parts[-1]
    return url  # assume bare ID was passed


def fetch_playlist_tracks(url: str) -> pd.DataFrame:
    """
    Fetch all tracks from a Spotify playlist URL.

    Paginates through the full playlist (Spotify returns max 100 tracks per call).
    Records the playlist name (from the API) in the `playlist` column.
    """
    sp = get_client()
    playlist_id = parse_playlist_id(url)

    # Fetch playlist name for labelling
    playlist_name = sp.playlist(playlist_id, fields="name")["name"]

    records = []
    offset = 0
    per_page = 100
    while True:
        page = sp.playlist_tracks(
            playlist_id,
            fields="items(track(id,name,artists,album,duration_ms,popularity)),next",
            limit=per_page,
            offset=offset,
        )
        items = page["items"]
        if not items:
            break
        for item in items:
            t = item.get("track")
            # Skip local files and null entries
            if not t or not t.get("id"):
                continue
            records.append({
                "track_id": t["id"],
                "title": t["name"],
                "artist": t["artists"][0]["name"],
                "album": t["album"]["name"],
                "release_year": t["album"]["release_date"][:4],
                "duration_ms": t["duration_ms"],
                "popularity": t["popularity"],
                "playlist": playlist_name,
            })
        offset += per_page
        time.sleep(0.1)
        if not page.get("next"):
            break

    print(f"  '{playlist_name}': {len(records)} tracks")
    return pd.DataFrame(records)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Spotify metadata from playlists")
    parser.add_argument(
        "playlists",
        nargs="+",
        help="One or more Spotify playlist URLs (e.g. https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M)",
    )
    parser.add_argument("--out", default="data/raw/metadata/spotify_raw.csv")
    args = parser.parse_args()

    dfs = []
    for url in args.playlists:
        print(f"Fetching playlist: {url}")
        df = fetch_playlist_tracks(url)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        print("No tracks fetched — check playlist names and API credentials.")
    else:
        combined = pd.concat(dfs, ignore_index=True).drop_duplicates("track_id")
        combined.to_csv(args.out, index=False)
        print(f"Saved {len(combined)} tracks to {args.out}")
