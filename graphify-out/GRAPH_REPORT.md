# Graph Report - .  (2026-07-31)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 111 nodes · 192 edges · 7 communities
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 12 edges (avg confidence: 0.6)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `98808ea8`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Dataset Loading|Dataset Loading]]
- [[_COMMUNITY_CLI Interface|CLI Interface]]
- [[_COMMUNITY_Neural Model Architecture|Neural Model Architecture]]
- [[_COMMUNITY_Test Suite|Test Suite]]
- [[_COMMUNITY_HTTP Prediction API|HTTP Prediction API]]
- [[_COMMUNITY_ML Pipeline Training|ML Pipeline Training]]
- [[_COMMUNITY_Project Documentation|Project Documentation]]

## God Nodes (most connected - your core abstractions)
1. `Fake News Detection` - 17 edges
2. `Split` - 14 edges
3. `Pipeline` - 7 edges
4. `train()` - 7 edges
5. `Namespace` - 6 edges
6. `DataError` - 6 edges
7. `load_frame()` - 6 edges
8. `train()` - 6 edges
9. `ModelError` - 6 edges
10. `Report` - 6 edges

## Surprising Connections (you probably didn't know these)
- `Fake News Detection` --references--> `tests.test_fnd`  [EXTRACTED]
  README.md → tests/test_fnd.py
- `CI Workflow` --calls--> `tests.test_fnd`  [EXTRACTED]
  .github/workflows/ci.yml → tests/test_fnd.py
- `Conv1D-LSTM Classifier` --calls--> `tensorflow`  [INFERRED]
  README.md → requirements-deep.txt
- `Split` --uses--> `Split`  [INFERRED]
  fnd/deep.py → fnd/data.py
- `Path` --uses--> `Split`  [INFERRED]
  fnd/deep.py → fnd/data.py

## Import Cycles
- None detected.

## Communities (7 total, 0 thin omitted)

### Community 0 - "Dataset Loading"
Cohesion: 0.15
Nodes (22): Immutable train/test partition of the corpus., Split, build_pipeline(), evaluate(), load(), ModelError, predict(), Path (+14 more)

### Community 1 - "CLI Interface"
Cohesion: 0.10
Nodes (21): CI Workflow, Conv1D-LSTM Classifier, fastapi, GeeksforGeeks Fake News Detection Walkthrough, joblib, LinearSVC Classifier, numpy, pandas (+13 more)

### Community 2 - "Neural Model Architecture"
Cohesion: 0.15
Nodes (17): fake-news Corpus (lutzhamel), checksum(), DataError, download(), load_frame(), make_split(), DataFrame, Path (+9 more)

### Community 3 - "Test Suite"
Cohesion: 0.22
Nodes (12): build_model(), ensure_glove(), load_embeddings(), Path, Split, Optional GloVe + Conv1D + LSTM model, mirroring the TensorFlow walkthrough.  Kep, Download and extract glove.6B.50d.txt if absent (~820 MB zip)., Build an embedding matrix; words absent from GloVe stay zero. (+4 more)

### Community 4 - "HTTP Prediction API"
Cohesion: 0.27
Nodes (11): ArgumentParser, build_parser(), cmd_deep_train(), cmd_download(), cmd_evaluate(), cmd_predict(), cmd_train(), main() (+3 more)

### Community 5 - "ML Pipeline Training"
Cohesion: 0.31
Nodes (8): BaseModel, Article, health(), _pipeline(), predict(), Prediction, FastAPI wrapper over the trained pipeline. Run: uvicorn fnd.api:app., Fake news detection: TF-IDF + linear SVM baseline, optional GloVe/LSTM model.

### Community 6 - "Project Documentation"
Cohesion: 0.33
Nodes (8): DataFrame, Self-check: `python tests/test_fnd.py`. No network, no framework.  Covers the br, synthetic_frame(), test_cleaning_drops_junk(), test_missing_columns_rejected(), test_predict_rejects_bad_input(), test_split_is_stratified_and_disjoint(), test_train_predict_roundtrip()

## Knowledge Gaps
- **10 isolated node(s):** `ArgumentParser`, `Series`, `Results`, `Docker`, `HTTP API` (+5 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Fake News Detection` connect `CLI Interface` to `Dataset Loading`, `Neural Model Architecture`, `Test Suite`, `HTTP Prediction API`, `ML Pipeline Training`?**
  _High betweenness centrality (0.232) - this node is a cross-community bridge._
- **Why does `Split` connect `Dataset Loading` to `Neural Model Architecture`, `Test Suite`?**
  _High betweenness centrality (0.150) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `Split` (e.g. with `Path` and `Split`) actually correct?**
  _`Split` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Fake news detection: TF-IDF + linear SVM baseline, optional GloVe/LSTM model.`, `FastAPI wrapper over the trained pipeline. Run: uvicorn fnd.api:app.`, `ArgumentParser` to the rest of the system?**
  _34 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `CLI Interface` be split into smaller, more focused modules?**
  _Cohesion score 0.10276679841897234 - nodes in this community are weakly interconnected._