"""Dataset loading for the fake-news corpus (title/text/label)."""

from __future__ import annotations

import hashlib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Same corpus used by the GeeksforGeeks TensorFlow walkthrough (`news.csv`:
# id/title/text/label with 6335 balanced FAKE/REAL rows), mirrored on GitHub.
DATA_URL = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv"
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "news.csv"

REQUIRED_COLUMNS = ("title", "text", "label")
VALID_LABELS = frozenset({"FAKE", "REAL"})
MIN_ROWS = 100
MIN_CHARS = 20  # shorter than this carries no usable signal


class DataError(RuntimeError):
    """Raised when the corpus is missing, malformed, or unusable."""


@dataclass(frozen=True)
class Split:
    """Immutable train/test partition of the corpus."""

    x_train: pd.Series
    x_test: pd.Series
    y_train: pd.Series
    y_test: pd.Series

    def __post_init__(self) -> None:
        if len(self.x_train) == 0 or len(self.x_test) == 0:
            raise DataError("split produced an empty partition")


def download(path: Path = DEFAULT_PATH, url: str = DATA_URL, force: bool = False) -> Path:
    """Fetch the corpus to `path` unless already cached. Returns the path."""
    if path.exists() and not force:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https URL
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise DataError(f"could not download corpus from {url}: {exc}") from exc

    if len(payload) < 1_000_000:
        raise DataError(f"downloaded corpus is implausibly small ({len(payload)} bytes)")

    tmp.write_bytes(payload)
    tmp.replace(path)
    return path


def checksum(path: Path = DEFAULT_PATH) -> str:
    """SHA-256 of the corpus file, for reproducibility notes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frame(path: Path = DEFAULT_PATH, auto_download: bool = True) -> pd.DataFrame:
    """Load and validate the corpus into a clean DataFrame."""
    path = Path(path)
    if not path.exists():
        if not auto_download:
            raise DataError(f"corpus not found at {path}; run `fnd download` first")
        download(path)

    try:
        frame = pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataError(f"corpus at {path} is not readable CSV: {exc}") from exc

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataError(f"corpus is missing required columns: {missing}")

    clean = (
        frame.loc[:, list(REQUIRED_COLUMNS)]
        .assign(
            title=lambda f: f["title"].fillna("").astype(str).str.strip(),
            text=lambda f: f["text"].fillna("").astype(str).str.strip(),
            label=lambda f: f["label"].astype(str).str.strip().str.upper(),
        )
        .loc[lambda f: f["label"].isin(VALID_LABELS)]
        .loc[lambda f: (f["title"].str.len() + f["text"].str.len()) >= MIN_CHARS]
        .drop_duplicates(subset=["title", "text"])
        .reset_index(drop=True)
    )

    if len(clean) < MIN_ROWS:
        raise DataError(f"only {len(clean)} usable rows after cleaning; expected >= {MIN_ROWS}")
    if clean["label"].nunique() < 2:
        raise DataError("corpus contains a single class after cleaning")
    return clean


def to_documents(frame: pd.DataFrame) -> pd.Series:
    """Join title and body into one document per article."""
    return (frame["title"] + "\n\n" + frame["text"]).rename("document")


def make_split(frame: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> Split:
    """Stratified train/test split over joined documents."""
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    documents = to_documents(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        documents,
        frame["label"],
        test_size=test_size,
        random_state=seed,
        stratify=frame["label"],
    )
    return Split(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
