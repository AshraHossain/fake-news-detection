"""Command line entry point: download, train, evaluate, predict."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import data, model


def _read_documents(args: argparse.Namespace) -> list[str]:
    if args.file:
        path = Path(args.file)
        if not path.exists():
            raise model.ModelError(f"no such file: {path}")
        return [path.read_text(encoding="utf8", errors="replace")]
    if args.text:
        return [args.text]
    piped = sys.stdin.read()
    if not piped.strip():
        raise model.ModelError("nothing to classify: pass --text, --file, or pipe stdin")
    return [piped]


def cmd_download(args: argparse.Namespace) -> int:
    path = data.download(Path(args.data), force=args.force)
    print(f"corpus: {path}\nsha256: {data.checksum(path)}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    frame = data.load_frame(Path(args.data))
    split = data.make_split(frame, test_size=args.test_size, seed=args.seed)
    print(f"corpus: {len(frame)} rows | train {len(split.x_train)} | test {len(split.x_test)}")

    pipeline, report = model.train(split, seed=args.seed, cross_validate=args.cross_validate)
    path = model.save(pipeline, Path(args.model))
    print(report.summary())
    print(f"\nsaved: {path}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    frame = data.load_frame(Path(args.data))
    split = data.make_split(frame, test_size=args.test_size, seed=args.seed)
    report = model.evaluate(model.load(Path(args.model)), split.x_test, split.y_test)
    print(report.summary())
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    results = model.predict(model.load(Path(args.model)), _read_documents(args))
    print(json.dumps(results if len(results) > 1 else results[0], indent=2))
    return 0


def cmd_deep_train(args: argparse.Namespace) -> int:
    from . import deep

    frame = data.load_frame(Path(args.data))
    split = data.make_split(frame, test_size=args.test_size, seed=args.seed)
    net, _tokenizer, metrics, _history = deep.train(split, epochs=args.epochs, seed=args.seed)
    deep.DEEP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    net.save(deep.DEEP_MODEL_PATH)
    print(json.dumps(metrics, indent=2))
    print(f"saved: {deep.DEEP_MODEL_PATH}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fnd", description="Fake news detection")
    parser.add_argument("--data", default=str(data.DEFAULT_PATH), help="corpus CSV path")
    parser.add_argument("--model", default=str(model.DEFAULT_MODEL_PATH), help="model artifact")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="fetch the corpus")
    download.add_argument("--force", action="store_true")
    download.set_defaults(func=cmd_download)

    train = subparsers.add_parser("train", help="train the TF-IDF baseline")
    train.add_argument("--cross-validate", action="store_true", help="add 5-fold CV on train")
    train.set_defaults(func=cmd_train)

    subparsers.add_parser("evaluate", help="score a saved model").set_defaults(func=cmd_evaluate)

    predict = subparsers.add_parser("predict", help="classify an article")
    predict.add_argument("--text")
    predict.add_argument("--file")
    predict.set_defaults(func=cmd_predict)

    deep_train = subparsers.add_parser("deep-train", help="train the GloVe+CNN+LSTM model")
    deep_train.add_argument("--epochs", type=int, default=50)
    deep_train.set_defaults(func=cmd_deep_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (data.DataError, model.ModelError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
