"""Optional GloVe + Conv1D + LSTM model, mirroring the TensorFlow walkthrough.

Kept out of the default path: it needs TensorFlow plus an ~820 MB GloVe download
and, on this corpus, scores below the TF-IDF baseline in `model.py`.

    pip install -r requirements-deep.txt
    python -m fnd.cli deep-train
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .data import Split

GLOVE_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
GLOVE_MEMBER = "glove.6B.50d.txt"
GLOVE_PATH = Path(__file__).resolve().parent.parent / "data" / GLOVE_MEMBER
DEEP_MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "deep.keras"

EMBED_DIM = 50
MAX_LEN = 54  # walkthrough's sequence length
OOV_TOKEN = "<OOV>"


def ensure_glove(path: Path = GLOVE_PATH) -> Path:
    """Download and extract glove.6B.50d.txt if absent (~820 MB zip)."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(GLOVE_URL, timeout=600) as response:  # noqa: S310 - fixed https URL
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    path.write_bytes(archive.read(GLOVE_MEMBER))
    return path


def load_embeddings(vocabulary: dict[str, int], glove_path: Path = GLOVE_PATH) -> np.ndarray:
    """Build an embedding matrix; words absent from GloVe stay zero."""
    vectors = np.zeros((len(vocabulary) + 1, EMBED_DIM), dtype="float32")
    with open(glove_path, encoding="utf8") as handle:
        for line in handle:
            word, _, rest = line.partition(" ")
            index = vocabulary.get(word)
            if index is not None:
                vectors[index] = np.fromstring(rest, sep=" ", dtype="float32")
    return vectors


def build_model(embeddings: np.ndarray):
    """Embedding (frozen) -> Dropout -> Conv1D -> MaxPool -> LSTM -> sigmoid."""
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Conv1D, Dense, Dropout, Embedding, MaxPooling1D

    model = Sequential(
        [
            Embedding(
                embeddings.shape[0],
                EMBED_DIM,
                weights=[embeddings],
                input_length=MAX_LEN,
                trainable=False,
            ),
            Dropout(0.2),
            Conv1D(64, 5, activation="relu"),
            MaxPooling1D(pool_size=4),
            LSTM(64),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def train(split: Split, epochs: int = 50, batch_size: int = 128, seed: int = 42):
    """Fit the deep model. Early stopping on because 50 flat epochs overfit hard."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    tf.keras.utils.set_random_seed(seed)
    ensure_glove()

    tokenizer = Tokenizer(oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(split.x_train)

    def encode(texts):
        return pad_sequences(
            tokenizer.texts_to_sequences(texts),
            maxlen=MAX_LEN,
            padding="post",
            truncating="post",
        )

    x_train, x_test = encode(split.x_train), encode(split.x_test)
    y_train = (split.y_train == "FAKE").to_numpy(dtype="float32")
    y_test = (split.y_test == "FAKE").to_numpy(dtype="float32")

    model = build_model(load_embeddings(tokenizer.word_index))
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy")],
        verbose=2,
    )
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return model, tokenizer, {"val_loss": float(loss), "val_accuracy": float(accuracy)}, history
