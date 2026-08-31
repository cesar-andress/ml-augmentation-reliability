"""Learner wrappers with common fit / predict_proba interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class BaseLearner(ABC):
    name: str
    family: str

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BaseLearner":
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of positive class (label=1), shape (n,)."""
        ...


class XGBoostLearner(BaseLearner):
    name = "xgboost"
    family = "gbdt"

    def __init__(self, params: dict[str, Any]):
        self.params = dict(params)
        self.model = None
        self.mark = self.params.pop("mark", "")

    def fit(self, X, y):
        from xgboost import XGBClassifier

        p = {
            "device": self.params.get("device", "cuda"),
            "tree_method": self.params.get("tree_method", "hist"),
            "max_depth": self.params.get("max_depth", 4),
            "n_estimators": self.params.get("n_estimators", 50),
            "learning_rate": self.params.get("learning_rate", 0.1),
            "subsample": self.params.get("subsample", 0.8),
            "colsample_bytree": self.params.get("colsample_bytree", 0.8),
            "eval_metric": self.params.get("eval_metric", "logloss"),
            "random_state": self.params.get("random_state", 0),
            "n_jobs": 1,
        }
        self.model = XGBClassifier(**p)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        proba = self.model.predict_proba(X)
        # positive class column for label 1
        classes = list(self.model.classes_)
        return proba[:, classes.index(1)].astype(np.float64)


class CatBoostLearner(BaseLearner):
    name = "catboost"
    family = "gbdt"

    def __init__(self, params: dict[str, Any], cat_features: list[str] | None = None):
        self.params = dict(params)
        self.cat_features = cat_features or []
        self.model = None
        self.mark = self.params.pop("mark", "")

    def fit(self, X, y):
        from catboost import CatBoostClassifier

        p = {
            "task_type": self.params.get("task_type", "GPU"),
            "iterations": self.params.get("iterations", 50),
            "depth": self.params.get("depth", 4),
            "learning_rate": self.params.get("learning_rate", 0.1),
            "verbose": self.params.get("verbose", False),
            "random_seed": self.params.get("random_seed", 0),
            "allow_writing_files": False,
        }
        self.model = CatBoostClassifier(**p)
        cat_idx = [X.columns.get_loc(c) for c in self.cat_features if c in X.columns]
        self.model.fit(X, y, cat_features=cat_idx)
        return self

    def predict_proba(self, X):
        proba = self.model.predict_proba(X)
        classes = list(self.model.classes_)
        # CatBoost may use string/int classes
        if 1 in classes:
            j = classes.index(1)
        else:
            j = 1 if proba.shape[1] > 1 else 0
        return proba[:, j].astype(np.float64)


class TabPFNLearner(BaseLearner):
    name = "tabpfn"
    family = "tfm"

    MAX_ROWS = 10_000
    MAX_FEATURES = 100

    def __init__(self, params: dict[str, Any]):
        self.params = dict(params)
        self.model = None

    def fit(self, X, y):
        import torch
        from tabpfn import TabPFNClassifier

        n_rows, n_feat = X.shape
        if n_rows > self.MAX_ROWS:
            raise AssertionError(f"TabPFN row limit exceeded: {n_rows} > {self.MAX_ROWS}")
        if n_feat > self.MAX_FEATURES:
            raise AssertionError(f"TabPFN feature limit exceeded: {n_feat} > {self.MAX_FEATURES}")

        model_path = Path(self.params["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"TabPFN checkpoint missing: {model_path}")

        prec = self.params.get("inference_precision", "float32")
        if prec == "float32":
            inference_precision = torch.float32
        else:
            inference_precision = prec

        self.model = TabPFNClassifier(
            n_estimators=int(self.params.get("n_estimators", 8)),
            model_path=str(model_path),
            device="cuda",
            ignore_pretraining_limits=bool(self.params.get("ignore_pretraining_limits", False)),
            inference_precision=inference_precision,
            n_preprocessing_jobs=int(self.params.get("n_preprocessing_jobs", 1)),
            random_state=int(self.params.get("random_state", 0)),
            softmax_temperature=float(self.params.get("softmax_temperature", 1.0)),
            # critical: do not auto-scale away from frozen n_estimators
            auto_scale_n_estimators=False,
        )
        self.model.fit(X.to_numpy(), y)
        return self

    def predict_proba(self, X):
        proba = self.model.predict_proba(X.to_numpy())
        classes = list(self.model.classes_)
        return proba[:, classes.index(1)].astype(np.float64)


class TabICLLearner(BaseLearner):
    name = "tabicl"
    family = "tfm"

    def __init__(self, params: dict[str, Any]):
        self.params = dict(params)
        self.model = None

    def fit(self, X, y):
        from tabicl import TabICLClassifier

        model_path = Path(self.params["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"TabICL checkpoint missing: {model_path}")

        self.model = TabICLClassifier(
            n_estimators=int(self.params.get("n_estimators", 8)),
            model_path=str(model_path),
            checkpoint_version=str(self.params.get("checkpoint_version", "tabicl-classifier-v1-20250208.ckpt")),
            use_amp=bool(self.params.get("use_amp", False)),
            batch_size=int(self.params.get("batch_size", 8)),
            average_logits=bool(self.params.get("average_logits", True)),
            random_state=int(self.params.get("random_state", 42)),
            softmax_temperature=float(self.params.get("softmax_temperature", 1.0)),
            device="cuda",
            allow_auto_download=False,
        )
        self.model.fit(X.to_numpy(), y)
        return self

    def predict_proba(self, X):
        proba = self.model.predict_proba(X.to_numpy())
        classes = list(self.model.classes_)
        return proba[:, classes.index(1)].astype(np.float64)


def build_learner(name: str, cfg: dict[str, Any], cat_features: list[str] | None = None) -> BaseLearner:
    if name == "xgboost":
        return XGBoostLearner(cfg["xgboost"])
    if name == "catboost":
        return CatBoostLearner(cfg["catboost"], cat_features=cat_features)
    if name == "tabpfn":
        return TabPFNLearner(cfg["tabpfn"])
    if name == "tabicl":
        return TabICLLearner(cfg["tabicl"])
    raise ValueError(f"unknown learner: {name}")
