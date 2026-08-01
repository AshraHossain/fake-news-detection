"""Self-check: `python tests/test_fnd.py`. No network, no framework.

Covers the branches that would silently ruin results if broken: cleaning,
stratified splitting, calibration/label ordering, and input validation.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fnd import data, model  # noqa: E402

FAKE_PHRASES = ["shocking truth exposed", "doctors hate this", "secret plot revealed"]
REAL_PHRASES = ["the committee said monday", "officials reported figures", "the agency confirmed"]


def synthetic_frame(per_class: int = 60) -> pd.DataFrame:
    rows = []
    for index in range(per_class):
        rows.append(
            {
                "title": f"{FAKE_PHRASES[index % 3]} number {index}",
                "text": f"{FAKE_PHRASES[(index + 1) % 3]} " * 12,
                "label": "FAKE",
            }
        )
        rows.append(
            {
                "title": f"{REAL_PHRASES[index % 3]} report {index}",
                "text": f"{REAL_PHRASES[(index + 1) % 3]} " * 12,
                "label": "REAL",
            }
        )
    return pd.DataFrame(rows)


def test_cleaning_drops_junk() -> None:
    frame = synthetic_frame()
    dirty = pd.concat(
        [
            frame,
            frame.iloc[[0]],  # duplicate
            pd.DataFrame([{"title": "hi", "text": "", "label": "REAL"}]),  # too short
            pd.DataFrame([{"title": "x" * 50, "text": "y" * 50, "label": "MAYBE"}]),  # bad label
        ],
        ignore_index=True,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "news.csv"
        dirty.to_csv(path, index=False)
        clean = data.load_frame(path, auto_download=False)

    assert len(clean) == len(frame), f"expected {len(frame)} rows, got {len(clean)}"
    assert set(clean["label"]) == {"FAKE", "REAL"}


def test_missing_columns_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.csv"
        pd.DataFrame({"title": ["a" * 50], "label": ["FAKE"]}).to_csv(path, index=False)
        try:
            data.load_frame(path, auto_download=False)
        except data.DataError as exc:
            assert "text" in str(exc)
        else:
            raise AssertionError("missing 'text' column should raise DataError")


def test_split_is_stratified_and_disjoint() -> None:
    split = data.make_split(synthetic_frame(), test_size=0.25, seed=7)
    assert len(split.x_test) == 30, len(split.x_test)
    assert split.y_test.value_counts().min() == 15, split.y_test.value_counts().to_dict()
    assert not set(split.x_train.index) & set(split.x_test.index), "train/test overlap"


def test_train_predict_roundtrip() -> None:
    split = data.make_split(synthetic_frame(), test_size=0.25, seed=7)
    pipeline, report = model.train(split, seed=7)

    assert report.accuracy > 0.9, f"separable corpus should be easy, got {report.accuracy}"
    assert 0.0 <= report.roc_auc <= 1.0
    assert sum(sum(row) for row in report.confusion) == len(split.x_test)

    with tempfile.TemporaryDirectory() as tmp:
        path = model.save(pipeline, Path(tmp) / "m.joblib")
        reloaded = model.load(path)

    hits = model.predict(reloaded, [FAKE_PHRASES[0] * 10, REAL_PHRASES[0] * 10])
    assert hits[0]["label"] == "FAKE", hits[0]
    assert hits[1]["label"] == "REAL", hits[1]
    # p_fake must track the label, i.e. calibration column is not inverted.
    assert hits[0]["p_fake"] > 0.5 > hits[1]["p_fake"], hits


def test_predict_rejects_bad_input() -> None:
    split = data.make_split(synthetic_frame(), test_size=0.25, seed=7)
    pipeline, _ = model.train(split, seed=7)

    for bad in ([], ["too short"], [None]):
        try:
            model.predict(pipeline, bad)
        except model.ModelError:
            continue
        raise AssertionError(f"expected ModelError for {bad!r}")


def test_missing_model_file() -> None:
    try:
        model.load(Path("/nonexistent/model.joblib"))
    except model.ModelError as exc:
        assert "train" in str(exc)
    else:
        raise AssertionError("loading a missing model should raise ModelError")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
