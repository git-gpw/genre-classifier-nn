"""
Train music genre classifiers.

Four modalities, each producing saved artifacts in --out_dir:

  metadata  Numeric metadata features (danceability, energy, tempo, etc.)
            -> LightGBM, logistic regression, or MLP.

  audio     CLAP embeddings -> LightGBM, logistic regression, or MLP.
            Supports both single-embedding-per-track and multi-window embeddings
            (same track_id repeated in embeddings.npz).
            Optional audio preprocessing is supported (L2 normalization,
            standardization, PCA), and the fitted preprocessor is saved.

  fusion    Late fusion of metadata/audio probabilities.
            Metadata/audio classifiers can be mixed (e.g.
            metadata=lightgbm + audio=mlp). Fusion weight can be fixed or
            tuned on validation by searching weight_meta in [0,1].

  audio_fusion  Late fusion of two audio classifiers (e.g. logreg + mlp)
                using track-level probabilities. Weight can be fixed or
                tuned on validation.

Usage:
  python -m src.models.train --modality metadata [options]
  python -m src.models.train --modality audio    [options]
  python -m src.models.train --modality fusion   [options]
  python -m src.models.train --modality audio_fusion [options]
"""

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

CLASSIFIER_CHOICES = ["lightgbm", "logreg", "mlp"]
AUDIO_STANDARDIZE_CHOICES = ["auto", "on", "off"]

# Numeric features the metadata model can use. Columns absent from the CSV
# (e.g. Spotify features not present in FMA data) are silently dropped.
METADATA_FEATURE_COLS = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
    "duration_s",
    "popularity",
    "release_year",
]

HANDCRAFTED_METADATA_PREFIXES = (
    "mfcc_",
    "chroma_",
    "spectral_centroid_",
    "spectral_rolloff_",
    "zero_crossing_rate_",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_split(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "duration_ms" in df.columns and "duration_s" not in df.columns:
        df["duration_s"] = df["duration_ms"] / 1000.0
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    return df.reset_index(drop=True)


def get_metadata_X(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Return metadata feature matrix in a stable column order.

    Missing columns are created as NaN via reindex so train/val/test transforms
    stay shape-compatible even if one split lacks a feature column.
    """
    X = df.reindex(columns=feature_cols).copy()
    return X.apply(pd.to_numeric, errors="coerce")


def discover_handcrafted_metadata_cols(df: pd.DataFrame) -> list[str]:
    return sorted(
        c
        for c in df.columns
        if any(c.startswith(prefix) for prefix in HANDCRAFTED_METADATA_PREFIXES)
    )


def metadata_feature_cols_for_training(df: pd.DataFrame) -> list[str]:
    base_cols = [c for c in METADATA_FEATURE_COLS if c in df.columns]
    handcrafted_cols = discover_handcrafted_metadata_cols(df)
    # Keep base columns first for readability/stable artefacts.
    return base_cols + handcrafted_cols


def parse_hidden_layers(spec: str) -> tuple[int, ...]:
    vals = [s.strip() for s in spec.split(",") if s.strip()]
    if not vals:
        raise ValueError("--mlp_hidden_layers must contain at least one integer")
    layers = tuple(int(v) for v in vals)
    if any(v <= 0 for v in layers):
        raise ValueError("--mlp_hidden_layers values must be positive integers")
    return layers


def fit_classifier(
    clf_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args,
):
    """Train and return a LightGBM, logistic regression, or MLP classifier."""
    if clf_type == "lightgbm":
        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            class_weight="balanced",
            verbose=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
    elif clf_type == "logreg":
        model = LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="multinomial",
        )
        model.fit(X_train, y_train)
    else:
        hidden_layers = parse_hidden_layers(args.mlp_hidden_layers)
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            alpha=args.mlp_alpha,
            batch_size=args.mlp_batch_size,
            learning_rate_init=args.mlp_lr,
            max_iter=args.mlp_max_iter,
            early_stopping=True,
            n_iter_no_change=args.mlp_patience,
            random_state=args.seed,
        )
        model.fit(X_train, y_train)
    return model


def report_f1(model, X_val: np.ndarray, y_val: np.ndarray, label: str) -> None:
    score = f1_score(y_val, model.predict(X_val), average="macro")
    print(f"  {label} macro F1: {score:.4f}")


def assert_model_matches_label_encoder(model, le: LabelEncoder, model_name: str) -> None:
    """
    Fail fast when a model artifact is not compatible with the current label encoder.
    """
    if not hasattr(model, "classes_"):
        return
    classes = np.asarray(model.classes_)
    expected = np.arange(len(le.classes_))
    if classes.shape != expected.shape or not np.array_equal(classes, expected):
        raise RuntimeError(
            f"{model_name} classes ({classes.tolist()}) do not match "
            f"label encoder classes (0..{len(le.classes_) - 1}). "
            "Use model/encoder artefacts from the same run directory."
        )


def _load_embeddings(npz_path: str) -> tuple[dict[str, list[int]], np.ndarray]:
    """
    Load embeddings.npz and return (track_id -> row_indices, embeddings array).

    If the NPZ has one row per track, each list will contain one index.
    If it has windowed embeddings, a track_id may map to multiple rows.
    """
    data = np.load(npz_path, allow_pickle=True)
    id_to_indices: dict[str, list[int]] = {}
    for i, tid in enumerate(data["track_ids"]):
        key = str(tid)
        id_to_indices.setdefault(key, []).append(i)
    return id_to_indices, data["embeddings"]


def _emb_matrix(
    df: pd.DataFrame,
    id_to_indices: dict[str, list[int]],
    embeddings: np.ndarray,
    known_genres: set[str],
    le: LabelEncoder,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) arrays from embedding lookup.

    For windowed NPZ files, each track contributes one row per window.
    """
    rows, labels = [], []
    for _, row in df.iterrows():
        tid = str(row["track_id"])
        genre = row.get("genre")
        if pd.isna(genre) or genre not in known_genres or tid not in id_to_indices:
            continue
        label = le.transform([genre])[0]
        for idx in id_to_indices[tid]:
            rows.append(embeddings[idx])
            labels.append(label)
    return np.array(rows, dtype=np.float32), np.array(labels, dtype=np.int32)


def _emb_matrix_with_source(
    df: pd.DataFrame,
    id_to_indices: dict[str, list[int]],
    embeddings: np.ndarray,
    known_genres: set[str],
    le: LabelEncoder,
    source_col: str = "source",
    default_source_label: str = "base",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X, y, source) arrays from embedding lookup.

    Source labels are repeated per embedding row (window) and are used for
    source-aware train resampling (e.g. base vs extra dataset ratio).
    """
    rows, labels, sources = [], [], []
    has_source_col = source_col in df.columns

    for _, row in df.iterrows():
        tid = str(row["track_id"])
        genre = row.get("genre")
        if pd.isna(genre) or genre not in known_genres or tid not in id_to_indices:
            continue

        src = default_source_label
        if has_source_col:
            raw_src = row.get(source_col)
            if not pd.isna(raw_src):
                src = str(raw_src)

        label = le.transform([genre])[0]
        for idx in id_to_indices[tid]:
            rows.append(embeddings[idx])
            labels.append(label)
            sources.append(src)

    return (
        np.array(rows, dtype=np.float32),
        np.array(labels, dtype=np.int32),
        np.array(sources, dtype=object),
    )


def _build_track_level_audio_eval_df(
    df: pd.DataFrame,
    known_genres: set[str],
    available_ids: set[str],
) -> pd.DataFrame:
    mask = df["genre"].isin(known_genres) & df["track_id"].astype(str).isin(available_ids)
    return df[mask].reset_index(drop=True)


def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def _resolve_audio_standardize(audio_standardize: str, classifier: str) -> bool:
    if audio_standardize == "on":
        return True
    if audio_standardize == "off":
        return False
    # auto
    return classifier in {"logreg", "mlp"}


def _resolve_audio_pca_components(value: float) -> int | float | None:
    v = float(value)
    if v <= 0:
        return None
    if v < 1:
        return v
    if not v.is_integer():
        raise ValueError(
            "--audio_pca_components must be <=0 (disabled), in (0,1), or a positive integer"
        )
    return int(v)


def _fit_audio_preprocessor(
    X_train: np.ndarray,
    args,
    classifier: str,
) -> tuple[dict, np.ndarray]:
    pca_components = _resolve_audio_pca_components(args.audio_pca_components)
    preprocessor = {
        "l2_norm": bool(args.audio_l2_norm),
        "standardize": _resolve_audio_standardize(args.audio_standardize, classifier),
        "scaler": None,
        "pca_components": pca_components,
        "pca_output_dim": None,
        "pca_explained_variance": None,
        "pca": None,
    }

    X_proc = np.asarray(X_train, dtype=np.float32)
    if preprocessor["l2_norm"]:
        X_proc = _l2_normalize_rows(X_proc)

    if preprocessor["standardize"]:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X_proc)
        preprocessor["scaler"] = scaler

    if pca_components is not None:
        pca = PCA(n_components=pca_components, svd_solver="auto", random_state=args.seed)
        X_proc = pca.fit_transform(X_proc)
        preprocessor["pca"] = pca
        preprocessor["pca_output_dim"] = int(X_proc.shape[1])
        preprocessor["pca_explained_variance"] = float(np.sum(pca.explained_variance_ratio_))

    return preprocessor, np.asarray(X_proc, dtype=np.float32)


def _apply_audio_preprocessor(X: np.ndarray, preprocessor: dict | None) -> np.ndarray:
    if preprocessor is None:
        return np.asarray(X, dtype=np.float32)

    X_proc = np.asarray(X, dtype=np.float32)
    if preprocessor.get("l2_norm", False):
        X_proc = _l2_normalize_rows(X_proc)

    scaler = preprocessor.get("scaler")
    if scaler is not None:
        X_proc = scaler.transform(X_proc)

    pca = preprocessor.get("pca")
    if pca is not None:
        X_proc = pca.transform(X_proc)

    return np.asarray(X_proc, dtype=np.float32)


def _audio_preprocessor_path(out_dir: Path, classifier: str) -> Path:
    return out_dir / f"audio_preprocessor_{classifier}.joblib"


def _load_audio_preprocessor(out_dir: Path, classifier: str) -> dict:
    path = _audio_preprocessor_path(out_dir, classifier)
    if path.exists():
        return joblib.load(path)
    return {
        "l2_norm": False,
        "standardize": False,
        "scaler": None,
        "pca_components": None,
        "pca_output_dim": None,
        "pca_explained_variance": None,
        "pca": None,
    }


def _resample_audio_train_by_source_ratio(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_sources: np.ndarray,
    *,
    base_label: str,
    extra_label: str,
    extra_to_base_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample train rows so extra/base ~= extra_to_base_ratio.

    Only rows with source == extra_label are down/up-sampled.
    Base rows are kept intact.
    """
    if extra_to_base_ratio < 0:
        raise ValueError("--audio_resample_extra_ratio must be >= 0")

    src = np.asarray(train_sources, dtype=object)
    base_idx = np.where(src == base_label)[0]
    extra_idx = np.where(src == extra_label)[0]
    other_idx = np.where((src != base_label) & (src != extra_label))[0]

    if len(base_idx) == 0:
        print(
            "  source resampling skipped: no base rows found "
            f"(source={base_label!r})"
        )
        return X_train, y_train, src
    if len(extra_idx) == 0:
        print(
            "  source resampling skipped: no extra rows found "
            f"(source={extra_label!r})"
        )
        return X_train, y_train, src

    target_extra = int(round(extra_to_base_ratio * len(base_idx)))
    rng = np.random.default_rng(seed)

    if target_extra == len(extra_idx):
        sampled_extra_idx = extra_idx
    elif target_extra <= 0:
        sampled_extra_idx = np.array([], dtype=np.int64)
    elif target_extra < len(extra_idx):
        sampled_extra_idx = rng.choice(extra_idx, size=target_extra, replace=False)
    else:
        sampled_extra_idx = rng.choice(extra_idx, size=target_extra, replace=True)

    keep_idx = np.concatenate([base_idx, sampled_extra_idx, other_idx])
    rng.shuffle(keep_idx)

    print(
        "  source resampling: "
        f"base={len(base_idx)}, extra_before={len(extra_idx)}, "
        f"extra_after={len(sampled_extra_idx)}, "
        f"target_ratio={extra_to_base_ratio:.3f}"
    )
    if len(other_idx) > 0:
        print(f"  source resampling: kept other source rows={len(other_idx)}")

    return X_train[keep_idx], y_train[keep_idx], src[keep_idx]


def _audio_probs_for_track_ids(
    track_ids: pd.Series,
    id_to_indices: dict[str, list[int]],
    embeddings: np.ndarray,
    audio_model,
    audio_preprocessor: dict | None = None,
) -> np.ndarray:
    """
    Return per-track probabilities.

    For each track, run predict_proba on all window embeddings and mean-pool
    probabilities to one vector.
    """
    probs = []
    for tid in track_ids.astype(str):
        idxs = id_to_indices[tid]
        X_track = _apply_audio_preprocessor(embeddings[idxs], audio_preprocessor)
        track_probs = audio_model.predict_proba(X_track).mean(axis=0)
        probs.append(track_probs)
    return np.array(probs, dtype=np.float32)


def _fuse_probabilities(
    prob_meta: np.ndarray,
    prob_audio: np.ndarray,
    weight_meta: float,
) -> np.ndarray:
    return weight_meta * prob_meta + (1.0 - weight_meta) * prob_audio


def _search_best_fusion_weight(
    y_true: np.ndarray,
    prob_meta: np.ndarray,
    prob_audio: np.ndarray,
    steps: int,
) -> tuple[float, float]:
    if steps <= 1:
        prob_fused = _fuse_probabilities(prob_meta, prob_audio, 0.5)
        return 0.5, f1_score(y_true, prob_fused.argmax(axis=1), average="macro")

    best_w = 0.5
    best_f1 = -1.0
    for w in np.linspace(0.0, 1.0, steps):
        prob_fused = _fuse_probabilities(prob_meta, prob_audio, float(w))
        f1 = f1_score(y_true, prob_fused.argmax(axis=1), average="macro")
        if f1 > best_f1 + 1e-12:
            best_f1 = f1
            best_w = float(w)

    return best_w, best_f1


def _fusion_weights_path(out_dir: Path, meta_clf: str, audio_clf: str) -> Path:
    return out_dir / f"fusion_weights_meta-{meta_clf}_audio-{audio_clf}.joblib"


def _audio_fusion_weights_path(out_dir: Path, audio_a_clf: str, audio_b_clf: str) -> Path:
    return out_dir / f"audio_fusion_weights_audioA-{audio_a_clf}_audioB-{audio_b_clf}.joblib"


# ---------------------------------------------------------------------------
# Metadata modality
# ---------------------------------------------------------------------------


def train_metadata(args, out_dir: Path) -> None:
    train_df = load_split(args.train_csv)
    val_df = load_split(args.val_csv)

    # Fit label encoder on train genres; save for reuse by audio/fusion/evaluate
    le = LabelEncoder()
    le.fit(train_df["genre"])
    joblib.dump(le, out_dir / "label_encoder.joblib")
    print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")

    # Build feature matrices (base metadata + optional handcrafted audio stats).
    feature_cols = metadata_feature_cols_for_training(train_df)
    if not feature_cols:
        raise RuntimeError(
            "No metadata feature columns were found in train split. "
            "Expected Spotify-style numeric metadata and/or handcrafted columns."
        )
    X_train = get_metadata_X(train_df, feature_cols)
    train_medians = X_train.median().fillna(0.0)

    X_val = get_metadata_X(val_df, feature_cols)

    # Fit scaler on train; impute NaNs with train medians to avoid leakage
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.fillna(train_medians).fillna(0.0))
    X_val_s = scaler.transform(X_val.fillna(train_medians).fillna(0.0))

    joblib.dump(scaler, out_dir / "metadata_scaler.joblib")
    joblib.dump(feature_cols, out_dir / "metadata_feature_cols.joblib")
    joblib.dump(train_medians, out_dir / "metadata_train_medians.joblib")

    # Filter to tracks with a known genre
    train_mask = train_df["genre"].isin(le.classes_)
    val_mask = val_df["genre"].isin(le.classes_)
    y_train = le.transform(train_df.loc[train_mask, "genre"])
    y_val = le.transform(val_df.loc[val_mask, "genre"])
    X_train_s = X_train_s[train_mask]
    X_val_s = X_val_s[val_mask]

    n_handcrafted = sum(
        any(col.startswith(prefix) for prefix in HANDCRAFTED_METADATA_PREFIXES)
        for col in feature_cols
    )
    print(
        f"Metadata model: {len(feature_cols)} features "
        f"({n_handcrafted} handcrafted) | "
        f"{len(le.classes_)} classes | "
        f"{len(y_train)} train / {len(y_val)} val tracks"
    )

    model = fit_classifier(args.classifier, X_train_s, y_train, X_val_s, y_val, args)
    report_f1(model, X_val_s, y_val, label="val")

    out_path = out_dir / f"metadata_{args.classifier}.joblib"
    joblib.dump(model, out_path)
    print(f"Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Audio modality
# ---------------------------------------------------------------------------


def train_audio(args, out_dir: Path) -> None:
    train_df = load_split(args.train_csv)
    val_df = load_split(args.val_csv)

    # Always fit label encoder from current train split to avoid stale artefacts
    # from previous runs silently filtering classes.
    le_path = out_dir / "label_encoder.joblib"
    le = LabelEncoder()
    le.fit(train_df["genre"])
    joblib.dump(le, le_path)

    id_to_indices, embeddings = _load_embeddings(args.embeddings)
    known = set(le.classes_)

    X_train, y_train, train_sources = _emb_matrix_with_source(
        train_df,
        id_to_indices,
        embeddings,
        known,
        le,
        source_col=args.audio_source_col,
        default_source_label=args.audio_source_default_label,
    )
    X_val, y_val = _emb_matrix(val_df, id_to_indices, embeddings, known, le)
    if len(y_train) == 0:
        raise RuntimeError("No training rows matched embeddings and known genres.")
    if len(y_val) == 0:
        raise RuntimeError("No validation rows matched embeddings and known genres.")

    if args.audio_resample_extra_ratio is not None:
        X_train, y_train, train_sources = _resample_audio_train_by_source_ratio(
            X_train,
            y_train,
            train_sources,
            base_label=args.audio_source_default_label,
            extra_label=args.audio_source_extra_label,
            extra_to_base_ratio=float(args.audio_resample_extra_ratio),
            seed=args.audio_resample_seed,
        )

    source_counts = pd.Series(train_sources).value_counts().to_dict()

    audio_preprocessor, X_train_p = _fit_audio_preprocessor(X_train, args, args.classifier)
    X_val_p = _apply_audio_preprocessor(X_val, audio_preprocessor)

    print(
        f"Audio model: {X_train.shape[1]}-dim embeddings | "
        f"{len(le.classes_)} classes | "
        f"{len(y_train)} train / {len(y_val)} val embedding-rows"
    )
    pca_model = audio_preprocessor.get("pca")
    if pca_model is None:
        pca_desc = "off"
    else:
        pca_desc = (
            f"on({audio_preprocessor.get('pca_output_dim')}d, "
            f"var={audio_preprocessor.get('pca_explained_variance', 0.0):.3f})"
        )
    print(
        "  audio preprocessing: "
        f"l2_norm={audio_preprocessor['l2_norm']} | "
        f"standardize={audio_preprocessor['standardize']} | "
        f"pca={pca_desc}"
    )
    print(f"  train sources (embedding-rows): {source_counts}")

    model = fit_classifier(args.classifier, X_train_p, y_train, X_val_p, y_val, args)
    report_f1(model, X_val_p, y_val, label="val (embedding-level)")

    # Report track-level score (mean pooled over windows)
    val_track_df = _build_track_level_audio_eval_df(val_df, known, set(id_to_indices))
    if len(val_track_df) > 0:
        prob_val_track = _audio_probs_for_track_ids(
            val_track_df["track_id"],
            id_to_indices,
            embeddings,
            model,
            audio_preprocessor=audio_preprocessor,
        )
        y_val_track = le.transform(val_track_df["genre"])
        f1_track = f1_score(y_val_track, prob_val_track.argmax(axis=1), average="macro")
        print(f"  val (track-level) macro F1: {f1_track:.4f}")

    out_path = out_dir / f"audio_{args.classifier}.joblib"
    joblib.dump(model, out_path)
    print(f"Saved -> {out_path}")

    preproc_path = _audio_preprocessor_path(out_dir, args.classifier)
    joblib.dump(audio_preprocessor, preproc_path)
    print(f"Saved -> {preproc_path}")


# ---------------------------------------------------------------------------
# Audio-only fusion: combine probabilities from two saved audio models
# ---------------------------------------------------------------------------


def train_audio_fusion(args, out_dir: Path) -> None:
    val_df = load_split(args.val_csv)

    audio_a_clf = args.audio_fusion_classifier_a
    audio_b_clf = args.audio_fusion_classifier_b

    le: LabelEncoder = joblib.load(out_dir / "label_encoder.joblib")
    model_a = joblib.load(out_dir / f"audio_{audio_a_clf}.joblib")
    model_b = joblib.load(out_dir / f"audio_{audio_b_clf}.joblib")
    preproc_a = _load_audio_preprocessor(out_dir, audio_a_clf)
    preproc_b = _load_audio_preprocessor(out_dir, audio_b_clf)
    assert_model_matches_label_encoder(model_a, le, "audio model A")
    assert_model_matches_label_encoder(model_b, le, "audio model B")

    id_to_indices, embeddings = _load_embeddings(args.embeddings)
    known = set(le.classes_)

    df = _build_track_level_audio_eval_df(val_df, known, set(id_to_indices))
    if len(df) == 0:
        raise RuntimeError("No validation tracks have audio embeddings and known genres.")

    y = le.transform(df["genre"])
    prob_a = _audio_probs_for_track_ids(
        df["track_id"],
        id_to_indices,
        embeddings,
        model_a,
        audio_preprocessor=preproc_a,
    )
    prob_b = _audio_probs_for_track_ids(
        df["track_id"],
        id_to_indices,
        embeddings,
        model_b,
        audio_preprocessor=preproc_b,
    )

    f1_a = f1_score(y, prob_a.argmax(axis=1), average="macro")
    f1_b = f1_score(y, prob_b.argmax(axis=1), average="macro")
    f1_equal = f1_score(
        y,
        _fuse_probabilities(prob_a, prob_b, 0.5).argmax(axis=1),
        average="macro",
    )

    if args.audio_fusion_weight_a is not None:
        weight_a = float(args.audio_fusion_weight_a)
        f1_fused = f1_score(
            y,
            _fuse_probabilities(prob_a, prob_b, weight_a).argmax(axis=1),
            average="macro",
        )
        strategy = "fixed"
    else:
        weight_a, f1_fused = _search_best_fusion_weight(
            y,
            prob_a,
            prob_b,
            steps=args.audio_fusion_weight_search_steps,
        )
        strategy = f"searched_{args.audio_fusion_weight_search_steps}_steps"

    weight_b = 1.0 - weight_a

    print(
        f"Audio Fusion ({audio_a_clf}+{audio_b_clf}) on {len(y)} val tracks:"
    )
    print(f"  {audio_a_clf:8s}: {f1_a:.4f}")
    print(f"  {audio_b_clf:8s}: {f1_b:.4f}")
    print(f"  equal(0.5/0.5): {f1_equal:.4f}")
    print(
        f"  fusion(weighted, {audio_a_clf}={weight_a:.3f}, "
        f"{audio_b_clf}={weight_b:.3f}): {f1_fused:.4f}"
    )

    weights_payload = {
        "audio_classifier_a": audio_a_clf,
        "audio_classifier_b": audio_b_clf,
        "weight_a": float(weight_a),
        "weight_b": float(weight_b),
        "strategy": strategy,
        "val_macro_f1": float(f1_fused),
    }
    weights_path = _audio_fusion_weights_path(out_dir, audio_a_clf, audio_b_clf)
    joblib.dump(weights_payload, weights_path)
    print(f"Saved audio-fusion weights -> {weights_path}")


# ---------------------------------------------------------------------------
# Fusion: combine probabilities from saved audio + metadata models
# ---------------------------------------------------------------------------


def train_fusion(args, out_dir: Path) -> None:
    val_df = load_split(args.val_csv)

    meta_clf = args.fusion_meta_classifier or args.classifier
    audio_clf = args.fusion_audio_classifier or args.classifier

    le: LabelEncoder = joblib.load(out_dir / "label_encoder.joblib")
    scaler: StandardScaler = joblib.load(out_dir / "metadata_scaler.joblib")
    feature_cols: list[str] = joblib.load(out_dir / "metadata_feature_cols.joblib")
    train_medians: pd.Series = joblib.load(out_dir / "metadata_train_medians.joblib")
    meta_model = joblib.load(out_dir / f"metadata_{meta_clf}.joblib")
    audio_model = joblib.load(out_dir / f"audio_{audio_clf}.joblib")
    audio_preprocessor = _load_audio_preprocessor(out_dir, audio_clf)
    assert_model_matches_label_encoder(meta_model, le, "metadata model")
    assert_model_matches_label_encoder(audio_model, le, "audio model")

    id_to_indices, embeddings = _load_embeddings(args.embeddings)
    known = set(le.classes_)

    # Filter val set to tracks with both modalities and a known genre
    df = _build_track_level_audio_eval_df(val_df, known, set(id_to_indices))
    if len(df) == 0:
        raise RuntimeError("No validation tracks have both metadata and audio embeddings.")

    X_meta_raw = get_metadata_X(df, feature_cols)
    X_meta = scaler.transform(X_meta_raw.fillna(train_medians).fillna(0.0))
    y = le.transform(df["genre"])

    prob_meta = meta_model.predict_proba(X_meta)
    prob_audio = _audio_probs_for_track_ids(
        df["track_id"],
        id_to_indices,
        embeddings,
        audio_model,
        audio_preprocessor=audio_preprocessor,
    )

    f1_meta = f1_score(y, meta_model.predict(X_meta), average="macro")
    f1_audio = f1_score(y, prob_audio.argmax(axis=1), average="macro")

    f1_equal = f1_score(
        y,
        _fuse_probabilities(prob_meta, prob_audio, 0.5).argmax(axis=1),
        average="macro",
    )

    if args.fusion_weight_meta is not None:
        weight_meta = float(args.fusion_weight_meta)
        f1_fused = f1_score(
            y,
            _fuse_probabilities(prob_meta, prob_audio, weight_meta).argmax(axis=1),
            average="macro",
        )
        strategy = "fixed"
    else:
        weight_meta, f1_fused = _search_best_fusion_weight(
            y,
            prob_meta,
            prob_audio,
            steps=args.fusion_weight_search_steps,
        )
        strategy = f"searched_{args.fusion_weight_search_steps}_steps"

    weight_audio = 1.0 - weight_meta

    print(
        f"Fusion ({meta_clf}+{audio_clf}) on {len(y)} val tracks with both modalities:"
    )
    print(f"  metadata : {f1_meta:.4f}")
    print(f"  audio    : {f1_audio:.4f}")
    print(f"  equal(0.5/0.5): {f1_equal:.4f}")
    print(
        f"  fusion(weighted, meta={weight_meta:.3f}, audio={weight_audio:.3f}): "
        f"{f1_fused:.4f}"
    )

    weights_payload = {
        "meta_classifier": meta_clf,
        "audio_classifier": audio_clf,
        "weight_meta": float(weight_meta),
        "weight_audio": float(weight_audio),
        "strategy": strategy,
        "val_macro_f1": float(f1_fused),
    }
    weights_path = _fusion_weights_path(out_dir, meta_clf, audio_clf)
    joblib.dump(weights_payload, weights_path)
    print(f"Saved fusion weights -> {weights_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train music genre classifier")
    parser.add_argument(
        "--modality",
        choices=["metadata", "audio", "fusion", "audio_fusion"],
        required=True,
        help="Which model to train",
    )
    parser.add_argument("--train_csv", default="data/splits/train.csv")
    parser.add_argument("--val_csv", default="data/splits/val.csv")
    parser.add_argument(
        "--embeddings",
        default="data/processed/embeddings/audio_embeddings.npz",
        help="Path to .npz produced by extract_audio.py (required for audio/fusion)",
    )
    parser.add_argument(
        "--classifier",
        choices=CLASSIFIER_CHOICES,
        default="lightgbm",
        help="Classifier used by metadata/audio training; default fallback for fusion",
    )
    parser.add_argument(
        "--fusion_meta_classifier",
        choices=CLASSIFIER_CHOICES,
        default=None,
        help="Optional classifier override for metadata model in fusion",
    )
    parser.add_argument(
        "--fusion_audio_classifier",
        choices=CLASSIFIER_CHOICES,
        default=None,
        help="Optional classifier override for audio model in fusion",
    )
    parser.add_argument(
        "--fusion_weight_meta",
        type=float,
        default=None,
        help=(
            "Fixed metadata fusion weight in [0,1]. "
            "If omitted, it is tuned on validation."
        ),
    )
    parser.add_argument(
        "--fusion_weight_search_steps",
        type=int,
        default=21,
        help="Number of candidate weights in [0,1] when tuning fusion weight",
    )
    parser.add_argument(
        "--audio_fusion_classifier_a",
        choices=CLASSIFIER_CHOICES,
        default="logreg",
        help="First audio classifier used by --modality audio_fusion",
    )
    parser.add_argument(
        "--audio_fusion_classifier_b",
        choices=CLASSIFIER_CHOICES,
        default="mlp",
        help="Second audio classifier used by --modality audio_fusion",
    )
    parser.add_argument(
        "--audio_fusion_weight_a",
        type=float,
        default=None,
        help=(
            "Fixed fusion weight for classifier A in [0,1]. "
            "If omitted, it is tuned on validation."
        ),
    )
    parser.add_argument(
        "--audio_fusion_weight_search_steps",
        type=int,
        default=21,
        help="Number of candidate weights in [0,1] when tuning audio-fusion weight",
    )
    parser.add_argument(
        "--audio_l2_norm",
        dest="audio_l2_norm",
        action="store_true",
        help="Apply row-wise L2 normalization to audio embeddings",
    )
    parser.add_argument(
        "--no_audio_l2_norm",
        dest="audio_l2_norm",
        action="store_false",
        help="Disable row-wise L2 normalization for audio embeddings",
    )
    parser.set_defaults(audio_l2_norm=True)
    parser.add_argument(
        "--audio_standardize",
        choices=AUDIO_STANDARDIZE_CHOICES,
        default="auto",
        help=(
            "Audio embedding standardization: 'auto' (on for logreg/mlp), "
            "'on', or 'off'"
        ),
    )
    parser.add_argument(
        "--audio_pca_components",
        type=float,
        default=0.0,
        help=(
            "Optional PCA applied to audio embeddings after normalization/"
            "standardization. Use 0 to disable, (0,1) for explained variance, "
            "or an integer >=1 for fixed dimensions."
        ),
    )
    parser.add_argument(
        "--audio_source_col",
        default="source",
        help="Column in train_csv identifying sample source (e.g. base/extra)",
    )
    parser.add_argument(
        "--audio_source_default_label",
        default="base",
        help="Default source label used when --audio_source_col is missing/NaN",
    )
    parser.add_argument(
        "--audio_source_extra_label",
        default="extra",
        help="Source label treated as extra set for train resampling",
    )
    parser.add_argument(
        "--audio_resample_extra_ratio",
        type=float,
        default=None,
        help=(
            "Optional target extra/base ratio on audio train embedding-rows. "
            "Example: 1.0 keeps extra ~= base; 0.5 keeps extra ~= half of base."
        ),
    )
    parser.add_argument(
        "--audio_resample_seed",
        type=int,
        default=42,
        help="Random seed for source-aware audio train resampling",
    )
    parser.add_argument(
        "--mlp_hidden_layers",
        default="512,256",
        help="Comma-separated hidden layer sizes for MLP (e.g. '512,256')",
    )
    parser.add_argument(
        "--mlp_lr",
        type=float,
        default=1e-3,
        help="MLP learning rate",
    )
    parser.add_argument(
        "--mlp_alpha",
        type=float,
        default=1e-4,
        help="MLP L2 regularization (alpha)",
    )
    parser.add_argument(
        "--mlp_batch_size",
        type=int,
        default=256,
        help="MLP batch size",
    )
    parser.add_argument(
        "--mlp_max_iter",
        type=int,
        default=300,
        help="MLP max iterations",
    )
    parser.add_argument(
        "--mlp_patience",
        type=int,
        default=20,
        help="MLP early-stopping patience (n_iter_no_change)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for MLP",
    )
    parser.add_argument(
        "--out_dir",
        default="models",
        help="Directory to write saved models and artefacts",
    )
    args = parser.parse_args()

    if args.fusion_weight_meta is not None and not (0.0 <= args.fusion_weight_meta <= 1.0):
        raise ValueError("--fusion_weight_meta must be in [0,1]")
    if args.fusion_weight_search_steps < 1:
        raise ValueError("--fusion_weight_search_steps must be >= 1")
    if args.audio_fusion_weight_a is not None and not (0.0 <= args.audio_fusion_weight_a <= 1.0):
        raise ValueError("--audio_fusion_weight_a must be in [0,1]")
    if args.audio_fusion_weight_search_steps < 1:
        raise ValueError("--audio_fusion_weight_search_steps must be >= 1")
    if args.audio_resample_extra_ratio is not None and args.audio_resample_extra_ratio < 0:
        raise ValueError("--audio_resample_extra_ratio must be >= 0")
    if args.audio_pca_components < 0:
        raise ValueError("--audio_pca_components must be >= 0")
    if args.audio_pca_components >= 1 and not float(args.audio_pca_components).is_integer():
        raise ValueError(
            "--audio_pca_components >= 1 must be an integer (e.g. 128, 256)"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    {
        "metadata": train_metadata,
        "audio": train_audio,
        "fusion": train_fusion,
        "audio_fusion": train_audio_fusion,
    }[args.modality](args, out_dir)
