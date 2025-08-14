"""
Train an EfficientNetB0 classifier on Kaggle flower datasets and export a model
that the advanced classifier can consume.

Outputs:
- models/flower_efficientnet.keras
- models/flower_labels.json (index -> class_name)

This script will:
1) Optionally download the datasets via kagglehub (paths printed if present)
2) Prepare a combined dataset directory with class subfolders
3) Train EfficientNetB0
4) Save model and labels
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

# Optional kagglehub import for downloading
try:
    import kagglehub  # type: ignore
except Exception:
    kagglehub = None

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
COMBINED_DATA = DATA_ROOT / "combined_flowers"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 8  # a bit more training for better accuracy
SEED = 42


def maybe_download_datasets() -> List[Path]:
    paths: List[Path] = []
    if kagglehub is None:
        print("kagglehub not installed or failed to import; skipping download.")
        return paths
    try:
        p1 = Path(kagglehub.dataset_download("l3llff/flowers"))
        print("Downloaded l3llff/flowers to:", p1)
        paths.append(p1)
    except Exception as e:
        print("Could not download l3llff/flowers:", e)
    try:
        p2 = Path(kagglehub.dataset_download("batoolabbas91/flower-photos-by-the-tensorflow-team"))
        print("Downloaded tensorflow-team flower photos to:", p2)
        paths.append(p2)
    except Exception as e:
        print("Could not download tensorflow-team dataset:", e)
    return paths


def ensure_dirs():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    COMBINED_DATA.mkdir(parents=True, exist_ok=True)


def normalize_class_name(name: str) -> str:
    name = name.strip().lower().replace(" ", "_")
    # Map common names to standard set
    mapping = {
        "daisy": "daisy",
        "dandelion": "dandelion",
        "rose": "rose",
        "sunflower": "sunflower",
        "tulip": "tulip",
        # Variants in some datasets
        "roses": "rose",
        "sunflowers": "sunflower",
        "tulips": "tulip",
        "daisies": "daisy",
        "dandelions": "dandelion",
    }
    return mapping.get(name, name)


def collect_images_from_source(src_dir: Path, dst_dir: Path):
    """Copy images from a Kaggle dataset folder structure into dst_dir with
    normalized class subfolders.
    """
    import shutil

    # Heuristics: look for class subfolders directly under src_dir, or under a nested folder
    candidates = [p for p in src_dir.glob("**/*") if p.is_dir()]

    # Filter to likely class dirs (contain images)
    class_dirs = []
    for d in candidates:
        has_images = any(f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"} for f in d.glob("*"))
        if has_images:
            class_dirs.append(d)

    for cdir in class_dirs:
        cls_name = normalize_class_name(cdir.name)
        out_dir = dst_dir / cls_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in cdir.glob("*"):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                # Avoid name collisions
                dst_path = out_dir / f"{cdir.name}__{f.name}"
                if not dst_path.exists():
                    try:
                        shutil.copy2(f, dst_path)
                    except Exception:
                        pass


def prepare_combined_dataset(download_paths: List[Path]):
    # Clear existing combined dir if empty-ish
    if any(COMBINED_DATA.iterdir()):
        print("Combined dataset directory not empty; reusing existing content.")
        return

    for p in download_paths:
        if p.exists():
            collect_images_from_source(p, COMBINED_DATA)
    print("Prepared combined dataset at:", COMBINED_DATA)


def build_model(num_classes: int) -> tf.keras.Model:
    # Ensure channels_last to avoid unexpected single-channel builds
    try:
        K.set_image_data_format("channels_last")
    except Exception:
        pass

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    # Lightweight data augmentation to help generalization
    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ], name="augment")

    # Try ImageNet weights; if there's a shape mismatch on some environments, fall back to random init
    weights_loaded = True
    try:
        base = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet", pooling="avg")
    except Exception as e:
        print("EfficientNet ImageNet weights failed (", str(e), ") — falling back to weights=None")
        base = EfficientNetB0(include_top=False, input_tensor=inputs, weights=None, pooling="avg")
        weights_loaded = False

    # If no pretrained weights, allow the base to learn features
    base.trainable = not weights_loaded

    # EfficientNet expects inputs scaled to [-1, 1]
    x = aug(inputs)
    x = layers.Rescaling(1./127.5, offset=-1)(x)
    x = base(x, training=base.trainable)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    # Use a lower LR if training the backbone from scratch
    lr = 1e-3 if weights_loaded else 5e-4
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # Attach a flag so caller can decide to fine-tune
    model._efnet_weights_loaded = weights_loaded  # type: ignore[attr-defined]
    model._efnet_base = base  # type: ignore[attr-defined]
    return model


essential_classes = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

def main():
    ensure_dirs()
    dl_paths = maybe_download_datasets()
    prepare_combined_dataset(dl_paths)

    # Use ALL classes present in COMBINED_DATA (no restriction),
    # so the model/labels reflect the full dataset you provide.
    class_filter = None

    # Build datasets (optionally restricting to filtered classes)
    ds_kwargs = dict(
        validation_split=0.2,
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    if class_filter:
        print("Training on subset of classes:", class_filter)
        ds_kwargs["class_names"] = class_filter

    train_ds = image_dataset_from_directory(
        COMBINED_DATA,
        subset="training",
        **ds_kwargs,
    )
    val_ds = image_dataset_from_directory(
        COMBINED_DATA,
        subset="validation",
        **ds_kwargs,
    )

    # Only keep classes we care about if present
    class_names = list(train_ds.class_names)
    print("Detected classes:", class_names)

    # Train model
    model = build_model(num_classes=len(class_names))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # Callbacks: early stopping to avoid long runs with no progress
    callbacks = [EarlyStopping(patience=2, restore_best_weights=True, monitor="val_accuracy")] 
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    # Optional fine-tuning: unfreeze top layers when pretrained weights are available
    if getattr(model, "_efnet_weights_loaded", False):
        base = getattr(model, "_efnet_base")
        # Unfreeze last 40 layers except BatchNorm
        trainable_count = 0
        for layer in base.layers[-40:]:
            if isinstance(layer, BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
                trainable_count += 1
        print(f"Fine-tuning top layers of EfficientNet (trainable layers: {trainable_count})")
        # Recompile with a lower LR
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        ft_callbacks = [EarlyStopping(patience=2, restore_best_weights=True, monitor="val_accuracy")]
        model.fit(train_ds, validation_data=val_ds, epochs=4, callbacks=ft_callbacks)

    # Save
    model_path = MODELS_DIR / "flower_efficientnet.keras"
    labels_path = MODELS_DIR / "flower_labels.json"
    model.save(model_path)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({i: name for i, name in enumerate(class_names)}, f, ensure_ascii=False, indent=2)

    print("Saved model to:", model_path)
    print("Saved labels to:", labels_path)


if __name__ == "__main__":
    main()
