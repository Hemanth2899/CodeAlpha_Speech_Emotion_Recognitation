import argparse
import json
import random
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "ravdess"
MODEL_PATH = BASE_DIR / "emotion_model.keras"
LABELS_PATH = BASE_DIR / "labels.json"

SAMPLE_RATE = 22050
DURATION = 3.0
N_MFCC = 40
MAX_FRAMES = 130
RANDOM_SEED = 42

# RAVDESS emotion codes
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def extract_features(audio_path: Path) -> np.ndarray:
    """Load audio and return a fixed-size normalized MFCC feature matrix."""
    audio, _ = librosa.load(
        str(audio_path),
        sr=SAMPLE_RATE,
        duration=DURATION,
        mono=True,
    )
    target_samples = int(SAMPLE_RATE * DURATION)

    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    else:
        audio = audio[:target_samples]

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=2048,
        hop_length=512,
    )

    if mfcc.shape[1] < MAX_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_FRAMES - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :MAX_FRAMES]

    # Normalize each recording's MFCC features for more stable training.
    mean = np.mean(mfcc)
    std = np.std(mfcc)
    mfcc = (mfcc - mean) / (std + 1e-8)
    return mfcc.astype(np.float32)


def find_ravdess_files():
    """Find audio-only, speech-channel RAVDESS WAV files."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR}\n"
            "Run: python download_dataset.py"
        )

    audio_files = []
    for path in DATA_DIR.rglob("*.wav"):
        parts = path.stem.split("-")
        # RAVDESS naming: modality-vocal_channel-emotion-intensity-statement-repetition-actor
        # Modality 03 = audio-only; vocal channel 01 = speech.
        if len(parts) == 7 and parts[0] == "03" and parts[1] == "01":
            if parts[2] in EMOTIONS:
                audio_files.append(path)

    if not audio_files:
        raise RuntimeError(
            "No valid RAVDESS speech WAV files found. Expected files inside "
            "data/ravdess/Actor_01, data/ravdess/Actor_02, etc. "
            "Run: python download_dataset.py"
        )
    return sorted(audio_files)


def load_dataset():
    paths = find_ravdess_files()
    features, labels = [], []
    skipped = 0

    print(f"Reading {len(paths)} RAVDESS audio files...")
    for index, path in enumerate(paths, start=1):
        try:
            parts = path.stem.split("-")
            emotion = EMOTIONS[parts[2]]
            features.append(extract_features(path))
            labels.append(emotion)
        except Exception as exc:
            skipped += 1
            print(f"Skipping {path.name}: {exc}")

        if index % 100 == 0:
            print(f"Processed {index}/{len(paths)} files")

    if not features:
        raise RuntimeError("Could not extract features from the dataset.")

    x = np.asarray(features, dtype=np.float32)[..., np.newaxis]
    class_names = sorted(set(labels))
    label_to_id = {name: i for i, name in enumerate(class_names)}
    y = np.asarray([label_to_id[name] for name in labels], dtype=np.int64)

    print(f"Usable recordings: {len(x)} (skipped: {skipped})")
    print("Classes:", ", ".join(class_names))
    return x, y, class_names


def build_model(num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(N_MFCC, MAX_FRAMES, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.20),

        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    x, y, class_names = load_dataset()
    counts = np.bincount(y)
    can_stratify = len(counts) > 1 and np.min(counts) >= 2

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y if can_stratify else None,
    )

    model = build_model(len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    model.fit(
        x_train,
        y_train,
        validation_split=0.15,
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    print(f"\nTest loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    # Save label order so prediction uses the same class mapping.
    LABELS_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    model.save(MODEL_PATH)
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved labels: {LABELS_PATH}")


def predict_emotion(audio_path: str):
    path = Path(audio_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Train it first with: python main.py --train"
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    features = extract_features(path)[np.newaxis, ..., np.newaxis]
    probabilities = model.predict(features, verbose=0)[0]
    best_index = int(np.argmax(probabilities))

    print(f"\nAudio file: {path.name}")
    print(f"Predicted emotion: {class_names[best_index].upper()}")
    print(f"Model score: {probabilities[best_index] * 100:.2f}%")
    print("\nScores:")
    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_names[i]:10s} {probabilities[i] * 100:6.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Speech Emotion Recognition using MFCC features and a CNN."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train using RAVDESS.")
    group.add_argument("--predict", metavar="AUDIO_FILE", help="Predict emotion from a WAV/audio file.")
    args = parser.parse_args()

    if args.train:
        train_model()
    else:
        predict_emotion(args.predict)


if __name__ == "__main__":
    main()
