"""TF-IDF + linear SVM classifier: train, evaluate, persist, predict."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .data import MIN_CHARS, Split

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "baseline.joblib"
POSITIVE_LABEL = "FAKE"


class ModelError(RuntimeError):
    """Raised when a model is unusable or an input is unclassifiable."""


@dataclass(frozen=True)
class Report:
    """Held-out evaluation results."""

    accuracy: float
    f1: float
    roc_auc: float
    confusion: list[list[int]]
    detail: str
    cross_val: list[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"accuracy : {self.accuracy:.4f}",
            f"f1({POSITIVE_LABEL}) : {self.f1:.4f}",
            f"roc_auc  : {self.roc_auc:.4f}",
        ]
        if self.cross_val:
            scores = np.asarray(self.cross_val)
            lines.append(f"cv f1    : {scores.mean():.4f} +/- {scores.std():.4f}")
        lines += ["", "confusion matrix [rows=true FAKE,REAL | cols=pred FAKE,REAL]:"]
        lines += [f"  {row}" for row in self.confusion]
        lines += ["", self.detail]
        return "\n".join(lines)


def build_pipeline(seed: int = 42) -> Pipeline:
    """Word + character n-gram TF-IDF into a probability-calibrated linear SVM.

    Character n-grams cost little at this corpus size and keep the model honest
    against spacing tricks and misspellings that shatter word tokens.
    """
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    lowercase=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    max_features=100_000,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
        ]
    )
    classifier = CalibratedClassifierCV(
        LinearSVC(C=1.0, class_weight="balanced", random_state=seed),
        cv=5,
        method="sigmoid",
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def _validate_documents(documents: Sequence[str]) -> list[str]:
    if len(documents) == 0:
        raise ModelError("no documents supplied")
    cleaned = []
    for index, document in enumerate(documents):
        if not isinstance(document, str):
            raise ModelError(f"document {index} is {type(document).__name__}, expected str")
        stripped = document.strip()
        if len(stripped) < MIN_CHARS:
            raise ModelError(
                f"document {index} has {len(stripped)} chars; need >= {MIN_CHARS} to classify"
            )
        cleaned.append(stripped)
    return cleaned


def train(split: Split, seed: int = 42, cross_validate: bool = False) -> tuple[Pipeline, Report]:
    """Fit on the training partition and score against the held-out partition."""
    pipeline = build_pipeline(seed=seed)
    pipeline.fit(split.x_train, split.y_train)

    scores: list[float] = []
    if cross_validate:
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        scores = cross_val_score(
            build_pipeline(seed=seed),
            split.x_train,
            split.y_train,
            cv=folds,
            scoring="f1_macro",
            n_jobs=-1,
        ).tolist()

    return pipeline, evaluate(pipeline, split.x_test, split.y_test, cross_val=scores)


def evaluate(
    pipeline: Pipeline,
    documents: Iterable[str],
    labels: Iterable[str],
    cross_val: Sequence[float] = (),
) -> Report:
    """Score a fitted pipeline against labelled documents."""
    documents = list(documents)
    labels = list(labels)
    predictions = pipeline.predict(documents)
    class_order = list(pipeline.classes_)
    positive_column = class_order.index(POSITIVE_LABEL)
    probabilities = pipeline.predict_proba(documents)[:, positive_column]
    truth = [1 if label == POSITIVE_LABEL else 0 for label in labels]

    return Report(
        accuracy=float(accuracy_score(labels, predictions)),
        f1=float(f1_score(labels, predictions, pos_label=POSITIVE_LABEL)),
        roc_auc=float(roc_auc_score(truth, probabilities)),
        confusion=confusion_matrix(labels, predictions, labels=["FAKE", "REAL"]).tolist(),
        detail=classification_report(labels, predictions, digits=4),
        cross_val=list(cross_val),
    )


def save(pipeline: Pipeline, path: Path = DEFAULT_MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path, compress=3)
    return path


def load(path: Path = DEFAULT_MODEL_PATH) -> Pipeline:
    path = Path(path)
    if not path.exists():
        raise ModelError(f"no model at {path}; run `fnd train` first")
    pipeline = joblib.load(path)
    if not hasattr(pipeline, "predict_proba"):
        raise ModelError(f"artifact at {path} is not a probability-capable classifier")
    return pipeline


def predict(pipeline: Pipeline, documents: Sequence[str]) -> list[dict[str, float | str]]:
    """Classify documents. Returns label plus P(FAKE) per document."""
    cleaned = _validate_documents(documents)
    positive_column = list(pipeline.classes_).index(POSITIVE_LABEL)
    probabilities = pipeline.predict_proba(cleaned)[:, positive_column]
    return [
        {
            "label": POSITIVE_LABEL if probability >= 0.5 else "REAL",
            "p_fake": round(float(probability), 4),
        }
        for probability in probabilities
    ]
