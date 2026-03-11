"""
Evaluate a trained spectrogram CNN on one NPZ split and export track probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def load_npz(npz_path: str) -> dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    return {
        "X": np.asarray(data["X"], dtype=np.float32),
        "y": np.asarray(data["y"], dtype=np.int32),
        "track_ids": np.asarray(data["track_ids"], dtype=object),
        "class_names": np.asarray(data["class_names"], dtype=object).astype(str),
    }


def aggregate_track_probs(
    track_ids: np.ndarray,
    y_true_seg: np.ndarray,
    prob_seg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    uniq, inv = np.unique(track_ids.astype(str), return_inverse=True)
    n_tracks = len(uniq)
    n_classes = prob_seg.shape[1]
    prob_track = np.zeros((n_tracks, n_classes), dtype=np.float32)
    y_track = np.zeros(n_tracks, dtype=np.int32)
    for i in range(n_tracks):
        mask = inv == i
        prob_track[i] = prob_seg[mask].mean(axis=0)
        y_track[i] = int(y_true_seg[mask][0])
    pred_track = prob_track.argmax(axis=1)
    return uniq, y_track, pred_track, prob_track


def _save_confusion(cm: np.ndarray, class_names: np.ndarray, out_png: Path, title: str, normalize: bool) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if normalize:
        denom = np.maximum(cm.sum(axis=1, keepdims=True), 1)
        plot_cm = cm.astype(np.float32) / denom
        fmt = ".2f"
    else:
        plot_cm = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) - 1)))
    sns.heatmap(
        plot_cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.3,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate spectrogram CNN and export track probabilities")
    parser.add_argument("--model_path", required=True, help="Path to Keras model (.keras)")
    parser.add_argument("--split_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tag", default="test")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    import tensorflow as tf

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(args.model_path)
    ds = load_npz(args.split_npz)
    class_names = ds["class_names"]

    prob_seg = model.predict(ds["X"], batch_size=args.batch_size, verbose=0)
    pred_seg = prob_seg.argmax(axis=1)
    y_seg = ds["y"]

    segment_macro_f1 = float(f1_score(y_seg, pred_seg, average="macro"))
    cm_seg = confusion_matrix(y_seg, pred_seg)
    report_seg = classification_report(y_seg, pred_seg, target_names=class_names.tolist(), digits=3, output_dict=True)

    track_ids, y_track, pred_track, prob_track = aggregate_track_probs(ds["track_ids"], y_seg, prob_seg)
    track_macro_f1 = float(f1_score(y_track, pred_track, average="macro"))
    cm_track = confusion_matrix(y_track, pred_track)
    report_track = classification_report(
        y_track, pred_track, target_names=class_names.tolist(), digits=3, output_dict=True
    )

    pred_genre_track = class_names[pred_track]
    true_genre_track = class_names[y_track]
    confidence_track = prob_track[np.arange(len(prob_track)), pred_track]

    probs_df = pd.DataFrame(
        {
            "track_id": track_ids.astype(str),
            "true_genre": true_genre_track.astype(str),
            "pred_genre": pred_genre_track.astype(str),
            "confidence": confidence_track.astype(float),
        }
    )
    for i, g in enumerate(class_names):
        probs_df[f"prob_{g}"] = prob_track[:, i].astype(float)

    probs_csv = out_dir / f"track_probs_{args.tag}.csv"
    probs_df.to_csv(probs_csv, index=False)

    _save_confusion(
        cm_seg,
        class_names,
        out_dir / f"confusion_segment_{args.tag}.png",
        title=f"Segment Confusion ({args.tag})",
        normalize=False,
    )
    _save_confusion(
        cm_seg,
        class_names,
        out_dir / f"confusion_segment_{args.tag}_norm.png",
        title=f"Segment Confusion Normalized ({args.tag})",
        normalize=True,
    )
    _save_confusion(
        cm_track,
        class_names,
        out_dir / f"confusion_track_{args.tag}.png",
        title=f"Track Confusion ({args.tag})",
        normalize=False,
    )
    _save_confusion(
        cm_track,
        class_names,
        out_dir / f"confusion_track_{args.tag}_norm.png",
        title=f"Track Confusion Normalized ({args.tag})",
        normalize=True,
    )

    report = {
        "tag": args.tag,
        "model_path": str(Path(args.model_path).resolve()),
        "split_npz": str(Path(args.split_npz).resolve()),
        "rows_segments": int(len(y_seg)),
        "rows_tracks": int(len(y_track)),
        "segment_macro_f1": segment_macro_f1,
        "track_macro_f1": track_macro_f1,
        "classification_report_segment": report_seg,
        "classification_report_track": report_track,
        "track_probs_csv": str(probs_csv.resolve()),
    }
    (out_dir / f"report_{args.tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "tag": args.tag,
            "segment_macro_f1": segment_macro_f1,
            "track_macro_f1": track_macro_f1,
            "track_probs_csv": str(probs_csv),
        },
        indent=2,
    ))
