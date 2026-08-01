# Fake News Detection

Binary FAKE/REAL classifier over the 6,335-article `news.csv` corpus (title + body).

## Results

Held-out test set (20%, stratified, seed 42):

| Model | Accuracy | F1 (FAKE) | ROC-AUC | Train time |
|---|---|---|---|---|
| TF-IDF (word 1-2 + char 3-5) → calibrated LinearSVC | **0.9492** | 0.9493 | 0.9895 | ~55 s CPU |
| GloVe-50d → Conv1D → LSTM (the walkthrough's model) | ~0.75 val | — | — | minutes + 820 MB download |

5-fold CV on the training split: macro-F1 `0.9572 ± 0.0062`.

The deep model is included in `fnd/deep.py` for reference, but the linear
baseline wins here by ~20 points. Frozen 50-dimensional embeddings truncated to
54 tokens throw away most of the article, and 54 tokens is roughly a headline.

## Usage

```bash
pip install -r requirements.txt
```

```bash
python -m fnd.cli download
```

```bash
python -m fnd.cli train --cross-validate
```

```bash
python -m fnd.cli predict --file article.txt
```

`predict` also accepts `--text "..."` or piped stdin, and emits
`{"label": "FAKE", "p_fake": 0.98}`.

```bash
python tests/test_fnd.py
```

### Docker

The build trains the model, so the image ships ready to classify:

```bash
docker build -t fnd .
```

```bash
docker run --rm fnd predict --text "SHOCKING: secret documents prove everything you know is a lie."
```

Any subcommand works — the entrypoint is the CLI:

```bash
docker run --rm fnd evaluate --data /app/data/news.csv
```

`evaluate` and `train` need the corpus, which is not in the runtime image; mount
it with `-v "$PWD/data:/app/data"`. `predict` needs nothing mounted.

Optional deep model:

```bash
pip install -r requirements-deep.txt && python -m fnd.cli deep-train
```

## Layout

| File | Role |
|---|---|
| `fnd/data.py` | download, validate, clean, stratified split |
| `fnd/model.py` | pipeline, train/evaluate/save/load/predict |
| `fnd/deep.py` | optional GloVe + Conv1D + LSTM |
| `fnd/cli.py` | `download` / `train` / `evaluate` / `predict` / `deep-train` |
| `tests/test_fnd.py` | assert-based self-check, no network |

## What "robust" means here

Robustness lives in the boundaries, not in extra layers:

- Corpus validated on load — required columns, label whitelist, minimum length,
  de-duplication. A malformed CSV raises `DataError` rather than training on junk.
- Splits are stratified and asserted disjoint in the test suite; the reported
  number is always held-out, never training accuracy.
- Character n-grams sit alongside word n-grams so spacing tricks and misspellings
  (`F R E E`, `vacciiine`) don't shatter the feature space.
- Probabilities are calibrated (`CalibratedClassifierCV`), so `p_fake` is usable
  as a threshold rather than a raw margin.
- `predict` refuses non-strings and documents under 20 characters instead of
  guessing from noise.

## Known limits

- **The corpus is the ceiling.** It is 2016 US-election-era English news from a
  fixed set of outlets. Part of the 95% is the model learning *publisher style*,
  not falsehood. Expect a large drop on any other domain, era, or language.
- **No claim verification.** This is stylometry. It cannot tell you whether a
  well-written statement is true — only whether it reads like articles labelled
  FAKE in this dataset. Do not deploy it as an arbiter of truth.
- Retrain before using on new data, and re-measure; don't carry the 0.9492 over.

## Provenance

Pipeline design (GloVe 50d, `maxlen=54`, Conv1D-64/kernel-5, MaxPool-4, LSTM-64,
sigmoid, adam/binary-crossentropy) follows the architecture described in the
[GeeksforGeeks TensorFlow walkthrough](https://www.geeksforgeeks.org/nlp/fake-news-detection-model-using-tensorflow-in-python/).
Hyperparameters are facts about a configuration; all code here is written fresh,
not copied. Corpus mirrored from
[lutzhamel/fake-news](https://github.com/lutzhamel/fake-news).
