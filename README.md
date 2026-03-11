# Music Genre Classifier

End-to-end pipeline for genre classification using:
- metadata features (Spotify/FMA numeric features),
- audio embeddings (LAION-CLAP),
- late fusion of metadata + audio probabilities.

The project supports both script-driven runs and notebook-driven runs.

## Setup

```bash
uv pip install -r requirements.txt
```

Optional dependency for spectrogram CNN branch:
- TensorFlow (`tensorflow`, `tensorflow-macos`, or equivalent for your platform)

System dependency:
- `ffmpeg` (required by `src.data.convert_audio`)

Environment variables:

| Variable | Used by |
|---|---|
| `SPOTIFY_CLIENT_ID` | `src.data.fetch_spotify` |
| `SPOTIFY_CLIENT_SECRET` | `src.data.fetch_spotify` |
| `DISCOGS_USER_TOKEN` | `src.data.fetch_discogs` |

## Pipeline Overview

```text
fetch_spotify / fetch_fma -> fetch_discogs -> clean_metadata -> make_splits
-> convert_audio -> extract_audio -> train (metadata/audio/fusion) -> evaluate
```

## Script Reference

### `src.data.fetch_spotify`
Fetch track metadata and Spotify audio features from one or more playlist URLs.

```bash
python -m src.data.fetch_spotify <playlist_url> [<playlist_url> ...] [--out PATH]
```

| Argument | Default |
|---|---|
| `playlists` (positional) | required |
| `--out` | `data/raw/metadata/spotify_raw.csv` |

### `src.data.fetch_discogs`
Enrich metadata with Discogs genre/style/year/label.

```bash
python -m src.data.fetch_discogs [--in_csv PATH] [--out_csv PATH] [--delay SECONDS]
```

| Argument | Default |
|---|---|
| `--in_csv` | `data/raw/metadata/spotify_raw.csv` |
| `--out_csv` | `data/raw/metadata/discogs_enriched.csv` |
| `--delay` | `1.0` |

### `src.data.fetch_fma`
Download FMA metadata and optionally audio files.

```bash
python -m src.data.fetch_fma [--subset {small,medium}] [--out_dir PATH] [--metadata_only]
```

| Argument | Default |
|---|---|
| `--subset` | `small` |
| `--out_dir` | `data/raw` |
| `--metadata_only` | `False` |

### `src.data.clean_metadata`
Normalize labels/features and control noisy/imbalanced classes.

```bash
python -m src.data.clean_metadata \
  [--in_csv PATH] [--out_csv PATH] [--min_genre_count N] \
  [--max_genres N] [--drop_genres CSV] [--max_samples_per_genre N] [--seed N]
```

| Argument | Default | Notes |
|---|---|---|
| `--in_csv` | `data/raw/metadata/discogs_enriched.csv` | input metadata |
| `--out_csv` | `data/processed/metadata.csv` | cleaned output |
| `--min_genre_count` | `50` | drop rare genres |
| `--max_genres` | `0` | keep top-N genres (`0` disables) |
| `--drop_genres` | `""` | comma-separated genres to remove |
| `--max_samples_per_genre` | `0` | cap per-class rows (`0` disables) |
| `--seed` | `42` | sampling seed |

### `src.data.make_splits`
Create artist-grouped train/val/test splits (prevents artist leakage).

```bash
python -m src.data.make_splits \
  [--metadata_csv PATH] [--out_dir PATH] [--train_ratio F] [--val_ratio F] [--seed N]
```

| Argument | Default |
|---|---|
| `--metadata_csv` | `data/processed/metadata.csv` |
| `--out_dir` | `data/splits` |
| `--train_ratio` | `0.70` |
| `--val_ratio` | `0.15` |
| `--seed` | `42` |

### `src.data.convert_audio`
Convert source audio to standardized MP3 for embedding extraction.

```bash
python -m src.data.convert_audio \
  [--metadata_csv PATH] [--in_dir PATH] [--out_dir PATH] \
  [--bitrate {128,192,320}] [--mono] [--workers N]
```

| Argument | Default |
|---|---|
| `--metadata_csv` | `data/processed/metadata.csv` |
| `--in_dir` | `data/raw/audio` |
| `--out_dir` | `data/processed/audio` |
| `--bitrate` | `192` |
| `--mono` | `False` |
| `--workers` | `4` |

### `src.data.prepare_extra_audio`
Prepare a folder-organized extra audio dataset and merge it into audio-only training splits.

Expected folder layout:
- `extra_audio/<genre>/*` (mp3/wav/flac/m4a/aac/ogg/opus/wma)

```bash
python -m src.data.prepare_extra_audio \
  [--extra_audio_dir PATH] [--processed_audio_dir PATH] \
  [--out_metadata_csv PATH] [--out_extra_splits_dir PATH] \
  [--base_metadata_csv PATH] [--base_splits_dir PATH] \
  [--out_audio_splits_dir PATH] [--out_audio_metadata_csv PATH] \
  [--include_extra_val] [--include_extra_test] \
  [--train_ratio F] [--val_ratio F] [--seed N] \
  [--bitrate {128,192,320}] [--mono] [--sample_rate N] [--workers N]
```

Outputs:
- `data/processed/extra_audio_metadata.csv`
- `data/splits/extra_audio/{train,val,test}.csv`
- `data/splits/audio_augmented/{train,val,test}.csv` (base + extra, with `source`)
- `data/processed/metadata_audio_augmented.csv`
- `data/processed/extra_audio_prepare_report.json`

### `src.data.prepare_data2_spectrogram`
Prepare GTZAN-style folder data (for example `Data 2/genres_original`) into
metadata + track-level stratified splits for spectrogram CNN training.

```bash
python -m src.data.prepare_data2_spectrogram \
  [--data_root PATH] [--out_metadata_csv PATH] [--out_splits_dir PATH] \
  [--train_ratio F] [--val_ratio F] [--seed N] [--report_json PATH]
```

Notes:
- Ignores macOS sidecar files like `._*.wav`.
- Output splits are `train.csv`, `val.csv`, `test.csv`.

### `src.features.extract_audio`
Extract CLAP embeddings. Supports single embedding per track or multiple windowed embeddings.

```bash
python -m src.features.extract_audio \
  [--metadata_csv PATH] [--audio_dir PATH] [--out_file PATH] [--batch_size N] \
  [--window_sec S] [--windows_per_track K] [--window_seed N]
```

| Argument | Default | Notes |
|---|---|---|
| `--metadata_csv` | `data/processed/metadata.csv` | must include `track_id` |
| `--audio_dir` | `data/processed/audio` | converted MP3 directory |
| `--out_file` | `data/processed/embeddings/audio_embeddings.npz` | output NPZ |
| `--batch_size` | `32` | CLAP batch size |
| `--window_sec` | `0.0` | `>0` enables random windows |
| `--windows_per_track` | `1` | embeddings per track |
| `--window_seed` | `42` | deterministic sampling |

Output NPZ keys:
- `track_ids`
- `sample_ids`
- `embeddings`

Notes:
- `track_id` may repeat when using windows.
- extraction is incremental by `sample_id` (already-saved samples are skipped).

### `src.features.extract_spectrogram`
Extract fixed-length mel spectrogram segments from split CSVs and save NPZ caches.

```bash
python -m src.features.extract_spectrogram \
  --splits_dir data/splits/data2_spectrogram \
  --out_dir data/processed/spectrograms/data2 \
  [--audio_dir PATH] [--segment_sec S] [--overlap F] [--n_mels N] \
  [--n_fft N] [--hop_length N] [--max_segments_per_track N] \
  [--out_dtype {float16,float32}] [--report_json PATH]
```

Output files:
- `spectrogram_train.npz`
- `spectrogram_val.npz`
- `spectrogram_test.npz`

Each NPZ contains `X`, `y`, `track_ids`, `genres`, `class_names`.

### `src.models.train`
Train metadata, audio, fusion, or audio-fusion model.

```bash
python -m src.models.train --modality {metadata,audio,fusion,audio_fusion} [options]
```

Core arguments:

| Argument | Default |
|---|---|
| `--train_csv` | `data/splits/train.csv` |
| `--val_csv` | `data/splits/val.csv` |
| `--embeddings` | `data/processed/embeddings/audio_embeddings.npz` |
| `--classifier` | `lightgbm` |
| `--out_dir` | `models` |

Classifier options:
- `--classifier`: `lightgbm`, `logreg`, `mlp`
- MLP options: `--mlp_hidden_layers`, `--mlp_lr`, `--mlp_alpha`, `--mlp_batch_size`, `--mlp_max_iter`, `--mlp_patience`, `--seed`

Audio preprocessing options (audio modality):
- `--audio_l2_norm` / `--no_audio_l2_norm`
- `--audio_standardize {auto,on,off}`
- `--audio_pca_components` (`0` disables; `(0,1)` keeps variance ratio; integer `>=1` keeps fixed components)
- Source-aware train resampling:
- `--audio_source_col` (default: `source`)
- `--audio_source_default_label` (default: `base`)
- `--audio_source_extra_label` (default: `extra`)
- `--audio_resample_extra_ratio` (example: `1.0` for extra~=base)
- `--audio_resample_seed`

Fusion options:
- `--fusion_meta_classifier`, `--fusion_audio_classifier`
- `--fusion_weight_meta` (fixed weight in `[0,1]`)
- `--fusion_weight_search_steps` (weight grid search size when no fixed weight is provided)

Audio-fusion options (`--modality audio_fusion`):
- `--audio_fusion_classifier_a` (default `logreg`)
- `--audio_fusion_classifier_b` (default `mlp`)
- `--audio_fusion_weight_a` (fixed weight in `[0,1]` for classifier A)
- `--audio_fusion_weight_search_steps` (weight grid search size when no fixed weight is provided)

Saved artifacts include:
- `label_encoder.joblib`
- `metadata_*.joblib` + metadata preprocessing artifacts
- `audio_*.joblib`
- `audio_preprocessor_*.joblib`
- `fusion_weights_meta-<meta>_audio-<audio>.joblib`

### `src.models.evaluate`
Evaluate metadata/audio/fusion on test split.

```bash
python -m src.models.evaluate --modality {metadata,audio,fusion,audio_fusion} [options]
```

| Argument | Default |
|---|---|
| `--test_csv` | `data/splits/test.csv` |
| `--embeddings` | `data/processed/embeddings/audio_embeddings.npz` |
| `--classifier` | `lightgbm` |
| `--fusion_meta_classifier` | `None` |
| `--fusion_audio_classifier` | `None` |
| `--fusion_weight_meta` | `None` |
| `--audio_fusion_classifier_a` | `logreg` |
| `--audio_fusion_classifier_b` | `mlp` |
| `--audio_fusion_weight_a` | `None` |
| `--model_dir` | `models` |

Evaluation behavior:
- Audio/fusion use track-level prediction (mean pooling over window probabilities).
- Fusion uses saved tuned weights if available; `--fusion_weight_meta` overrides.

### `src.models.train_spectrogram_cnn`
Train/fine-tune a spectrogram CNN from NPZ splits.

```bash
python -m src.models.train_spectrogram_cnn \
  --train_npz data/processed/spectrograms/main/spectrogram_train.npz \
  --val_npz data/processed/spectrograms/main/spectrogram_val.npz \
  [--test_npz data/processed/spectrograms/main/spectrogram_test.npz] \
  --out_dir models/full_run/cnn_main \
  [--pretrained_weights PATH] [--freeze_backbone_epochs N] \
  [--epochs N] [--batch_size N] [--learning_rate F] [--no_augment]
```

Saved artifacts include:
- `best_model.keras`, `final_model.keras`, `model.weights.h5`
- `history.json`, `report.json`, `class_names.joblib`

### `src.models.evaluate_spectrogram_cnn`
Evaluate a trained spectrogram CNN and export track-level probabilities for fusion.

```bash
python -m src.models.evaluate_spectrogram_cnn \
  --model_path models/full_run/cnn_main/final_model.keras \
  --split_npz data/processed/spectrograms/main/spectrogram_test.npz \
  --out_dir models/full_run/cnn_main/eval --tag test
```

Outputs include:
- `track_probs_<tag>.csv` with `prob_<genre>` columns
- segment/track confusion matrices (count + normalized)
- `report_<tag>.json`

## Recommended Commands (Windowed Audio + MLP + Weighted Fusion)

```bash
python -m src.data.clean_metadata \
  --in_csv data/raw/metadata/discogs_enriched.csv \
  --out_csv data/processed/metadata.csv \
  --min_genre_count 120 \
  --max_genres 10 \
  --max_samples_per_genre 800

python -m src.data.make_splits --metadata_csv data/processed/metadata.csv --out_dir data/splits --seed 42

python -m src.data.convert_audio --metadata_csv data/processed/metadata.csv --in_dir data/raw/deemix_downloads --out_dir data/processed/audio --workers 8

python -m src.features.extract_audio \
  --metadata_csv data/processed/metadata_audio_augmented.csv \
  --audio_dir data/processed/audio \
  --out_file data/processed/embeddings/audio_embeddings_windows.npz \
  --window_sec 8 --windows_per_track 8 --window_seed 42

python -m src.models.train --modality metadata --classifier lightgbm --out_dir models/full_run

python -m src.models.train --modality audio --classifier mlp \
  --train_csv data/splits/audio_augmented/train.csv \
  --val_csv data/splits/audio_augmented/val.csv \
  --embeddings data/processed/embeddings/audio_embeddings_windows.npz \
  --audio_source_col source --audio_source_extra_label extra \
  --audio_source_default_label base --audio_resample_extra_ratio 1.0 \
  --audio_pca_components 0.95 \
  --audio_l2_norm --audio_standardize auto \
  --mlp_hidden_layers 1024,512,256 --mlp_lr 5e-4 --mlp_alpha 3e-4 \
  --mlp_batch_size 256 --mlp_max_iter 500 --mlp_patience 25 \
  --out_dir models/full_run

python -m src.models.train --modality fusion \
  --fusion_meta_classifier lightgbm --fusion_audio_classifier mlp \
  --fusion_weight_search_steps 41 \
  --embeddings data/processed/embeddings/audio_embeddings_windows.npz \
  --out_dir models/full_run

python -m src.models.evaluate --modality fusion \
  --fusion_meta_classifier lightgbm --fusion_audio_classifier mlp \
  --embeddings data/processed/embeddings/audio_embeddings_windows.npz \
  --model_dir models/full_run

# Optional: audio-only ensemble (logreg + mlp)
python -m src.models.train --modality audio --classifier logreg \
  --embeddings data/processed/embeddings/audio_embeddings_windows.npz \
  --out_dir models/full_run

python -m src.models.train --modality audio_fusion \
  --audio_fusion_classifier_a logreg --audio_fusion_classifier_b mlp \
  --audio_fusion_weight_search_steps 41 \
  --embeddings data/processed/embeddings/audio_embeddings_windows.npz \
  --out_dir models/full_run

python -m src.models.evaluate --modality audio_fusion \
  --audio_fusion_classifier_a logreg --audio_fusion_classifier_b mlp \
  --embeddings data/processed/embeddings/audio_embeddings_windows.npz \
  --model_dir models/full_run
```

## Notebook Workflow

Use `post_download_pipeline_full.ipynb` for a guided full run.
Use `post_download_pipeline_extra_audio_only.ipynb` to train/evaluate an audio-only model on `extra_audio/` only.

Key notebook flags:
- `RUN_TUNING_SWEEP`: enable/disable window+MLP sweep
- `WINDOW_SEC_CANDIDATES`, `WINDOWS_PER_TRACK_CANDIDATES`, `MLP_SWEEP`: sweep search space
- `MIN_GENRE_COUNT`, `MAX_GENRES`, `DROP_GENRES`, `MAX_SAMPLES_PER_GENRE`: class quality controls

When sweep is enabled, the notebook selects the best validation configuration and copies the best trial artifacts into `models/full_run`.
