"""
Train a spectrogram CNN on cached NPZ splits.

Supports:
- standard training from scratch
- fine-tune from pretrained weights with optional warmup freeze epochs

NPZ format expected (from src.features.extract_spectrogram):
- X          : [n_segments, n_mels, n_frames, 1]
- y          : int labels
- track_ids  : segment-level track ids
- class_names: class names, consistent across splits
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def load_npz(npz_path: str) -> dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    out = {
        "X": np.asarray(data["X"], dtype=np.float32),
        "y": np.asarray(data["y"], dtype=np.int32),
        "track_ids": np.asarray(data["track_ids"], dtype=object),
        "class_names": np.asarray(data["class_names"], dtype=object),
    }
    return out


def assert_class_names_match(train_classes: np.ndarray, other_classes: np.ndarray, tag: str) -> None:
    if train_classes.shape != other_classes.shape or not np.array_equal(train_classes, other_classes):
        raise RuntimeError(f"class_names mismatch for {tag}")


def aggregate_track_probs(
    track_ids: np.ndarray,
    y_true_seg: np.ndarray,
    prob_seg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return y_track, pred_track, prob_track


def _build_model(tf, input_shape: tuple[int, ...], num_classes: int, l2_strength: float):
    L = tf.keras.layers
    R = tf.keras.regularizers
    model = tf.keras.Sequential(
        [
            L.Input(shape=input_shape, name="input_spec"),
            L.Conv2D(16, (3, 3), padding="same", kernel_regularizer=R.l2(l2_strength), name="conv1"),
            L.BatchNormalization(name="bn1"),
            L.ELU(name="elu1"),
            L.MaxPooling2D((2, 2), name="pool1"),
            L.Dropout(0.2, name="drop1"),
            L.Conv2D(32, (3, 3), padding="same", kernel_regularizer=R.l2(l2_strength), name="conv2"),
            L.BatchNormalization(name="bn2"),
            L.ELU(name="elu2"),
            L.MaxPooling2D((2, 2), name="pool2"),
            L.Dropout(0.25, name="drop2"),
            L.Conv2D(64, (3, 3), padding="same", kernel_regularizer=R.l2(l2_strength), name="conv3"),
            L.BatchNormalization(name="bn3"),
            L.ELU(name="elu3"),
            L.GlobalAveragePooling2D(name="gap"),
            L.Dropout(0.4, name="drop3"),
            L.Dense(64, kernel_regularizer=R.l2(l2_strength), name="dense_embed"),
            L.BatchNormalization(name="bn_embed"),
            L.ELU(name="elu_embed"),
            L.Dropout(0.5, name="drop_embed"),
            L.Dense(num_classes, activation="softmax", name="classifier"),
        ]
    )
    return model


def _spec_augment(tf, spectrogram, label):
    spec = spectrogram
    freq_bins = tf.shape(spec)[0]
    time_steps = tf.shape(spec)[1]

    t = tf.random.uniform((), 0, tf.maximum(1, time_steps // 8), dtype=tf.int32)
    t0 = tf.random.uniform((), 0, tf.maximum(1, time_steps - t), dtype=tf.int32)
    time_mask = tf.concat(
        [
            tf.ones([freq_bins, t0, 1], dtype=spec.dtype),
            tf.zeros([freq_bins, t, 1], dtype=spec.dtype),
            tf.ones([freq_bins, time_steps - t0 - t, 1], dtype=spec.dtype),
        ],
        axis=1,
    )
    spec = spec * time_mask

    f = tf.random.uniform((), 0, tf.maximum(1, freq_bins // 8), dtype=tf.int32)
    f0 = tf.random.uniform((), 0, tf.maximum(1, freq_bins - f), dtype=tf.int32)
    freq_mask = tf.concat(
        [
            tf.ones([f0, time_steps, 1], dtype=spec.dtype),
            tf.zeros([f, time_steps, 1], dtype=spec.dtype),
            tf.ones([freq_bins - f0 - f, time_steps, 1], dtype=spec.dtype),
        ],
        axis=0,
    )
    spec = spec * freq_mask
    spec = tf.image.random_brightness(spec, max_delta=0.05)
    spec = tf.clip_by_value(spec, 0.0, 1.0)
    return spec, label


def _make_dataset(
    tf,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    augment: bool,
    seed: int,
    shuffle: bool = False,
):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=seed)
    if augment:
        ds = ds.map(lambda a, b: _spec_augment(tf, a, b), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def _evaluate_split(
    tf,
    model,
    split_tag: str,
    data: dict[str, np.ndarray],
    class_names: np.ndarray,
) -> dict:
    X = data["X"]
    y = data["y"]
    track_ids = data["track_ids"]

    prob_seg = model.predict(X, verbose=0, batch_size=256)
    pred_seg = prob_seg.argmax(axis=1)
    seg_macro_f1 = float(f1_score(y, pred_seg, average="macro"))

    y_track, pred_track, prob_track = aggregate_track_probs(track_ids, y, prob_seg)
    track_macro_f1 = float(f1_score(y_track, pred_track, average="macro"))

    report_seg = classification_report(y, pred_seg, target_names=class_names.tolist(), digits=3, output_dict=True)
    report_track = classification_report(
        y_track, pred_track, target_names=class_names.tolist(), digits=3, output_dict=True
    )
    return {
        "split": split_tag,
        "segments": int(len(y)),
        "tracks": int(len(y_track)),
        "segment_macro_f1": seg_macro_f1,
        "track_macro_f1": track_macro_f1,
        "classification_report_segment": report_seg,
        "classification_report_track": report_track,
        "confusion_segment": confusion_matrix(y, pred_seg).tolist(),
        "confusion_track": confusion_matrix(y_track, pred_track).tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train spectrogram CNN")
    parser.add_argument("--train_npz", required=True)
    parser.add_argument("--val_npz", required=True)
    parser.add_argument("--test_npz", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--pretrained_weights", default="")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--no_class_weight", action="store_true")
    args = parser.parse_args()

    import os
    import random

    # TensorFlow is intentionally imported only when this script is executed.
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_npz(args.train_npz)
    val = load_npz(args.val_npz)
    class_names = train["class_names"].astype(str)
    assert_class_names_match(class_names, val["class_names"].astype(str), tag="val")

    test = None
    if args.test_npz:
        test = load_npz(args.test_npz)
        assert_class_names_match(class_names, test["class_names"].astype(str), tag="test")

    input_shape = tuple(train["X"].shape[1:])
    model = _build_model(tf, input_shape=input_shape, num_classes=len(class_names), l2_strength=args.l2)

    if args.pretrained_weights:
        pw = Path(args.pretrained_weights)
        if not pw.exists():
            raise FileNotFoundError(f"pretrained weights not found: {pw}")
        try:
            model.load_weights(str(pw), by_name=True, skip_mismatch=True)
            print(f"Loaded pretrained weights (by_name=True, skip_mismatch=True): {pw}")
        except ValueError as exc:
            msg = str(exc)
            # Keras 3 disallows by_name=True for modern .weights.h5 files.
            if "by_name" in msg and "legacy '.h5' or '.hdf5'" in msg:
                model.load_weights(str(pw), skip_mismatch=True)
                print(f"Loaded pretrained weights (by_name=False, skip_mismatch=True): {pw}")
            else:
                raise

    def compile_model(lr: float):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    train_ds = _make_dataset(
        tf,
        train["X"],
        train["y"],
        args.batch_size,
        augment=(not args.no_augment),
        seed=args.seed,
        shuffle=True,
    )
    val_ds = _make_dataset(tf, val["X"], val["y"], args.batch_size, augment=False, seed=args.seed, shuffle=False)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history: dict[str, list] = {}
    class_weight = None
    if not args.no_class_weight:
        counts = np.bincount(train["y"], minlength=len(class_names)).astype(np.float64)
        nonzero = counts > 0
        if np.any(nonzero):
            total = float(np.sum(counts[nonzero]))
            n_cls = float(np.sum(nonzero))
            class_weight = {}
            for i, c in enumerate(counts):
                if c > 0:
                    class_weight[i] = float(total / (n_cls * c))
            print("Using class_weight:", class_weight)

    if args.freeze_backbone_epochs > 0 and args.pretrained_weights:
        for layer in model.layers:
            if layer.name != "classifier":
                layer.trainable = False
        compile_model(lr=args.learning_rate * 0.5)
        warmup_epochs = min(args.freeze_backbone_epochs, args.epochs)
        h1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=warmup_epochs,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )
        history = {k: list(v) for k, v in h1.history.items()}

        if args.epochs > warmup_epochs:
            for layer in model.layers:
                layer.trainable = True
            compile_model(lr=args.learning_rate)
            h2 = model.fit(
                train_ds,
                validation_data=val_ds,
                initial_epoch=warmup_epochs,
                epochs=args.epochs,
                class_weight=class_weight,
                callbacks=callbacks,
                verbose=1,
            )
            for k, v in h2.history.items():
                history.setdefault(k, [])
                history[k].extend(list(v))
    else:
        compile_model(lr=args.learning_rate)
        h = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )
        history = {k: list(v) for k, v in h.history.items()}

    best_model_path = out_dir / "best_model.keras"
    if best_model_path.exists():
        model = tf.keras.models.load_model(best_model_path)

    model.save(out_dir / "final_model.keras")
    model.save_weights(out_dir / "model.weights.h5")
    joblib.dump(class_names.tolist(), out_dir / "class_names.joblib")

    history_json = out_dir / "history.json"
    history_json.write_text(json.dumps(history, indent=2), encoding="utf-8")

    report = {
        "class_names": class_names.tolist(),
        "input_shape": list(input_shape),
        "train_npz": str(Path(args.train_npz).resolve()),
        "val_npz": str(Path(args.val_npz).resolve()),
        "test_npz": str(Path(args.test_npz).resolve()) if args.test_npz else "",
        "pretrained_weights": args.pretrained_weights,
        "epochs_requested": int(args.epochs),
        "freeze_backbone_epochs": int(args.freeze_backbone_epochs),
        "learning_rate": float(args.learning_rate),
        "no_augment": bool(args.no_augment),
        "use_class_weight": bool(not args.no_class_weight),
        "class_weight": class_weight if class_weight is not None else {},
        "train": _evaluate_split(tf, model, "train", train, class_names),
        "val": _evaluate_split(tf, model, "val", val, class_names),
    }
    if test is not None:
        report["test"] = _evaluate_split(tf, model, "test", test, class_names)

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "out_dir": str(out_dir.resolve()),
            "class_count": len(class_names),
            "val_segment_macro_f1": report["val"]["segment_macro_f1"],
            "val_track_macro_f1": report["val"]["track_macro_f1"],
            "test_segment_macro_f1": report.get("test", {}).get("segment_macro_f1", None),
            "test_track_macro_f1": report.get("test", {}).get("track_macro_f1", None),
        },
        indent=2,
    ))
