"""FastAPI wrapper over the trained pipeline. Run: uvicorn fnd.api:app."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import model

app = FastAPI(title="Fake News Detection", version="0.1.0")


class Article(BaseModel):
    text: str = Field(min_length=1, description="Article title and/or body")


class Prediction(BaseModel):
    label: str
    p_fake: float


@lru_cache(maxsize=1)
def _pipeline():
    # Loaded once on first request, then cached; save() writes it at build time.
    return model.load()


@app.get("/health")
def health() -> dict[str, str]:
    try:
        _pipeline()
    except model.ModelError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(article: Article) -> Prediction:
    try:
        result = model.predict(_pipeline(), [article.text])[0]
    except model.ModelError as exc:
        # Too-short / unclassifiable input is the caller's fault, not a 500.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Prediction(**result)
