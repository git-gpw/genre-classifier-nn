"""
Evaluate a trained genre classifier on the held-out test split.

Reports:
  - Macro F1 (overall)
  - Per-genre precision, recall, F1 (sklearn classification_report)
  - Saves a row-normalised confusion matrix heatmap as a PNG

Works for all modalities. For fusion/audio_fusion, corresponding model
artefacts must already be saved in --model_dir.
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

CLASSIFIER_CHOICES = ["lightgbm", "logreg", "mlp"]

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


# ---------------------------------------------------------------------------
# Helpers (mirrors train.py)
# ---------------------------------------------------------------------------


def load_split(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "duration_ms" in df.columns and "duration_s" not in df.columns:
        df["duration_s"] = df["duration_ms"] / 1000.0
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    return df.reset_index(drop=True)


def get_metadata_X(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df.reindex(columns=feature_cols).copy()
    return X.apply(pd.to_numeric, errors="coerce")


def load_embeddings(npz_path: str) -> tuple[dict[str, list[int]], np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    id_to_indices: dict[str, list[int]] = {}
    for i, tid in enumerate(data["track_ids"]):
        key = str(tid)
        id_to_indices.setdefault(key, []).append(i)
    return id_to_indices, data["embeddings"]


def assert_model_matches_label_encoder(model, le: LabelEncoder, model_name: str) -> None:
    """
    Fail fast when model artefacts and label encoder are from different runs.
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


def build_track_level_audio_eval_df(
    df: pd.DataFrame,
    known_genres: set[str],
    available_ids: set[str],
) -> pd.DataFrame:
    mask = df["genre"].isin(known_genres) & df["track_id"].astype(str).isin(available_ids)
    return df[mask].reset_index(drop=True)


def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


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


def _audio_preprocessor_path(model_dir: Path, classifier: str) -> Path:
    return model_dir / f"audio_preprocessor_{classifier}.joblib"


def _load_audio_preprocessor(model_dir: Path, classifier: str) -> dict:
    path = _audio_preprocessor_path(model_dir, classifier)
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


def audio_probs_for_track_ids(
    track_ids: pd.Series,
    id_to_indices: dict[str, list[int]],
    embeddings: np.ndarray,
    audio_model,
    audio_preprocessor: dict | None = None,
) -> np.ndarray:
    probs = []
    for tid in track_ids.astype(str):
        idxs = id_to_indices[tid]
        X_track = _apply_audio_preprocessor(embeddings[idxs], audio_preprocessor)
        track_probs = audio_model.predict_proba(X_track).mean(axis=0)
        probs.append(track_probs)
    return np.array(probs, dtype=np.float32)


def _fusion_weights_path(model_dir: Path, meta_clf: str, audio_clf: str) -> Path:
    return model_dir / f"fusion_weights_meta-{meta_clf}_audio-{audio_clf}.joblib"


def _audio_fusion_weights_path(model_dir: Path, audio_a_clf: str, audio_b_clf: str) -> Path:
    return model_dir / f"audio_fusion_weights_audioA-{audio_a_clf}_audioB-{audio_b_clf}.joblib"


def _resolve_fusion_weight_meta(
    args,
    model_dir: Path,
    meta_clf: str,
    audio_clf: str,
) -> tuple[float, str]:
    if args.fusion_weight_meta is not None:
        return float(args.fusion_weight_meta), "cli"

    weights_path = _fusion_weights_path(model_dir, meta_clf, audio_clf)
    if weights_path.exists():
        payload = joblib.load(weights_path)
        if "weight_meta" in payload:
            return float(payload["weight_meta"]), f"saved:{weights_path.name}"

    return 0.5, "default_equal"


def _resolve_audio_fusion_weight_a(
    args,
    model_dir: Path,
    audio_a_clf: str,
    audio_b_clf: str,
) -> tuple[float, str]:
    if args.audio_fusion_weight_a is not None:
        return float(args.audio_fusion_weight_a), "cli"

    weights_path = _audio_fusion_weights_path(model_dir, audio_a_clf, audio_b_clf)
    if weights_path.exists():
        payload = joblib.load(weights_path)
        if "weight_a" in payload:
            return float(payload["weight_a"]), f"saved:{weights_path.name}"

    return 0.5, "default_equal"


def _fuse_probabilities(
    prob_meta: np.ndarray,
    prob_audio: np.ndarray,
    weight_meta: float,
) -> np.ndarray:
    return weight_meta * prob_meta + (1.0 - weight_meta) * prob_audio


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: np.ndarray,
    out_path: Path,
) -> None:
    """Save a row-normalised (recall) confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    n = len(class_names)
    fig_size = max(8, n)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 1))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=0,
        vmax=1,
        linewidths=0.4,
    )
    ax.set_xlabel("Predicted genre", fontsize=12)
    ax.set_ylabel("True genre", fontsize=12)
    ax.set_title("Confusion matrix — row-normalised recall", fontsize=13)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved -> {out_path}")


def report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    le: LabelEncoder,
    tag: str,
    model_dir: Path,
) -> None:
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n{'=' * 60}")
    print(f"{tag}  |  test macro F1: {macro_f1:.4f}  ({len(y_true)} tracks)")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=le.classes_, digits=3))
    save_confusion_matrix(y_true, y_pred, le.classes_, model_dir / f"confusion_{tag}.png")


# ---------------------------------------------------------------------------
# Per-modality evaluation
# ---------------------------------------------------------------------------


def evaluate_metadata(args, model_dir: Path) -> None:
    test_df = load_split(args.test_csv)
    le: LabelEncoder = joblib.load(model_dir / "label_encoder.joblib")
    scaler: StandardScaler = joblib.load(model_dir / "metadata_scaler.joblib")
    feature_cols: list[str] = joblib.load(model_dir / "metadata_feature_cols.joblib")
    train_medians: pd.Series = joblib.load(model_dir / "metadata_train_medians.joblib")
    model = joblib.load(model_dir / f"metadata_{args.classifier}.joblib")
    assert_model_matches_label_encoder(model, le, "metadata model")

    known = set(le.classes_)
    test_df = test_df[test_df["genre"].isin(known)].reset_index(drop=True)

    X = get_metadata_X(test_df, feature_cols)
    X_s = scaler.transform(X.fillna(train_medians).fillna(0.0))
    y = le.transform(test_df["genre"])

    tag = f"metadata_{args.classifier}"
    report(y, model.predict(X_s), le, tag, model_dir)


def evaluate_audio(args, model_dir: Path) -> None:
    test_df = load_split(args.test_csv)
    le: LabelEncoder = joblib.load(model_dir / "label_encoder.joblib")
    model = joblib.load(model_dir / f"audio_{args.classifier}.joblib")
    audio_preprocessor = _load_audio_preprocessor(model_dir, args.classifier)
    assert_model_matches_label_encoder(model, le, "audio model")

    id_to_indices, embeddings = load_embeddings(args.embeddings)
    known = set(le.classes_)

    df = build_track_level_audio_eval_df(test_df, known, set(id_to_indices))
    prob_audio = audio_probs_for_track_ids(
        df["track_id"],
        id_to_indices,
        embeddings,
        model,
        audio_preprocessor=audio_preprocessor,
    )
    y = le.transform(df["genre"])

    tag = f"audio_{args.classifier}"
    report(y, prob_audio.argmax(axis=1), le, tag, model_dir)


def evaluate_fusion(args, model_dir: Path) -> None:
    test_df = load_split(args.test_csv)
    meta_clf = args.fusion_meta_classifier or args.classifier
    audio_clf = args.fusion_audio_classifier or args.classifier

    le: LabelEncoder = joblib.load(model_dir / "label_encoder.joblib")
    scaler: StandardScaler = joblib.load(model_dir / "metadata_scaler.joblib")
    feature_cols: list[str] = joblib.load(model_dir / "metadata_feature_cols.joblib")
    train_medians: pd.Series = joblib.load(model_dir / "metadata_train_medians.joblib")
    meta_model = joblib.load(model_dir / f"metadata_{meta_clf}.joblib")
    audio_model = joblib.load(model_dir / f"audio_{audio_clf}.joblib")
    audio_preprocessor = _load_audio_preprocessor(model_dir, audio_clf)
    assert_model_matches_label_encoder(meta_model, le, "metadata model")
    assert_model_matches_label_encoder(audio_model, le, "audio model")

    id_to_indices, embeddings = load_embeddings(args.embeddings)
    known = set(le.classes_)

    df = build_track_level_audio_eval_df(test_df, known, set(id_to_indices))

    X_meta_raw = get_metadata_X(df, feature_cols)
    X_meta = scaler.transform(X_meta_raw.fillna(train_medians).fillna(0.0))
    y = le.transform(df["genre"])

    prob_meta = meta_model.predict_proba(X_meta)
    prob_audio = audio_probs_for_track_ids(
        df["track_id"],
        id_to_indices,
        embeddings,
        audio_model,
        audio_preprocessor=audio_preprocessor,
    )

    weight_meta, weight_source = _resolve_fusion_weight_meta(args, model_dir, meta_clf, audio_clf)
    weight_audio = 1.0 - weight_meta
    prob_fused = _fuse_probabilities(prob_meta, prob_audio, weight_meta)

    tag = f"fusion_meta-{meta_clf}_audio-{audio_clf}"
    report(y, prob_fused.argmax(axis=1), le, tag, model_dir)

    # Also print individual model scores for comparison
    f1_meta = f1_score(y, meta_model.predict(X_meta), average="macro")
    f1_audio = f1_score(y, prob_audio.argmax(axis=1), average="macro")
    f1_fused = f1_score(y, prob_fused.argmax(axis=1), average="macro")
    print(f"\nFusion weights: meta={weight_meta:.3f}, audio={weight_audio:.3f} (source={weight_source})")
    print(f"Comparison on {len(y)} test tracks with both modalities:")
    print(f"  metadata : {f1_meta:.4f}")
    print(f"  audio    : {f1_audio:.4f}")
    print(f"  fusion   : {f1_fused:.4f}")


def evaluate_audio_fusion(args, model_dir: Path) -> None:
    test_df = load_split(args.test_csv)
    audio_a_clf = args.audio_fusion_classifier_a
    audio_b_clf = args.audio_fusion_classifier_b

    le: LabelEncoder = joblib.load(model_dir / "label_encoder.joblib")
    model_a = joblib.load(model_dir / f"audio_{audio_a_clf}.joblib")
    model_b = joblib.load(model_dir / f"audio_{audio_b_clf}.joblib")
    preproc_a = _load_audio_preprocessor(model_dir, audio_a_clf)
    preproc_b = _load_audio_preprocessor(model_dir, audio_b_clf)
    assert_model_matches_label_encoder(model_a, le, "audio model A")
    assert_model_matches_label_encoder(model_b, le, "audio model B")

    id_to_indices, embeddings = load_embeddings(args.embeddings)
    known = set(le.classes_)
    df = build_track_level_audio_eval_df(test_df, known, set(id_to_indices))
    if len(df) == 0:
        raise RuntimeError("No test tracks have audio embeddings and known genres.")

    y = le.transform(df["genre"])
    prob_a = audio_probs_for_track_ids(
        df["track_id"],
        id_to_indices,
        embeddings,
        model_a,
        audio_preprocessor=preproc_a,
    )
    prob_b = audio_probs_for_track_ids(
        df["track_id"],
        id_to_indices,
        embeddings,
        model_b,
        audio_preprocessor=preproc_b,
    )

    weight_a, weight_source = _resolve_audio_fusion_weight_a(
        args,
        model_dir,
        audio_a_clf,
        audio_b_clf,
    )
    weight_b = 1.0 - weight_a
    prob_fused = _fuse_probabilities(prob_a, prob_b, weight_a)

    tag = f"audio_fusion_audioA-{audio_a_clf}_audioB-{audio_b_clf}"
    report(y, prob_fused.argmax(axis=1), le, tag, model_dir)

    f1_a = f1_score(y, prob_a.argmax(axis=1), average="macro")
    f1_b = f1_score(y, prob_b.argmax(axis=1), average="macro")
    f1_fused = f1_score(y, prob_fused.argmax(axis=1), average="macro")
    print(
        f"\nAudio-fusion weights: {audio_a_clf}={weight_a:.3f}, "
        f"{audio_b_clf}={weight_b:.3f} (source={weight_source})"
    )
    print(f"Comparison on {len(y)} test tracks:")
    print(f"  {audio_a_clf:8s}: {f1_a:.4f}")
    print(f"  {audio_b_clf:8s}: {f1_b:.4f}")
    print(f"  fusion   : {f1_fused:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate genre classifier on test split")
    parser.add_argument(
        "--modality",
        choices=["metadata", "audio", "fusion", "audio_fusion"],
        required=True,
    )
    parser.add_argument("--test_csv", default="data/splits/test.csv")
    parser.add_argument(
        "--embeddings",
        default="data/processed/embeddings/audio_embeddings.npz",
        help="Required for audio and fusion modalities",
    )
    parser.add_argument(
        "--classifier",
        choices=CLASSIFIER_CHOICES,
        default="lightgbm",
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
            "Optional metadata fusion weight in [0,1]. "
            "If omitted, uses saved tuned weight when available."
        ),
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
            "Optional fusion weight for classifier A in [0,1]. "
            "If omitted, uses saved tuned weight when available."
        ),
    )
    parser.add_argument("--model_dir", default="models")
    args = parser.parse_args()

    if args.fusion_weight_meta is not None and not (0.0 <= args.fusion_weight_meta <= 1.0):
        raise ValueError("--fusion_weight_meta must be in [0,1]")
    if args.audio_fusion_weight_a is not None and not (0.0 <= args.audio_fusion_weight_a <= 1.0):
        raise ValueError("--audio_fusion_weight_a must be in [0,1]")

    evaluate_fn = {
        "metadata": evaluate_metadata,
        "audio": evaluate_audio,
        "fusion": evaluate_fusion,
        "audio_fusion": evaluate_audio_fusion,
    }[args.modality]
    evaluate_fn(args, Path(args.model_dir))
