# app/services/ml_ranker.py
from __future__ import annotations
from typing import List, Dict, Any
import os

import numpy as np
import joblib

from app.core.config import get_settings

settings = get_settings()


class MlRanker:
    """
    Simple wrapper quanh model sklearn (LogisticRegression, XGBoost, ...)

    QUY ƯỚC FEATURE:
    -----------------
    - Mỗi candidate là 1 dict có các key:
        "faiss_score": float
        "graph_score": float
        "total_kcal":  float

    - Khi train model trong train_model.py, anh PHẢI dùng đúng thứ tự:
        X = [[faiss_score, graph_score, total_kcal], ...]
    """

    # Thứ tự feature cố định, dùng cho cả train & infer
    FEATURE_KEYS = ["faiss_score", "graph_score", "total_kcal"]

    def __init__(self, model=None):
        self.model = model

    @classmethod
    def load(cls) -> "MlRanker":
        """
        Load model từ file settings.ML_MODEL_PATH nếu có.
        Nếu chưa có file, coi như chưa dùng ML (rơi vào mode "fallback 0.5").
        """
        if os.path.exists(settings.ML_MODEL_PATH):
            model = joblib.load(settings.ML_MODEL_PATH)
            return cls(model=model)
        return cls(model=None)

    def has_model(self) -> bool:
        return self.model is not None

    def _features_to_matrix(self, features: List[Dict[str, Any]]) -> np.ndarray:
        """
        Map list[dict] -> numpy array (N x D) theo FEATURE_KEYS.
        Thiếu key nào thì mặc định 0.0
        """
        X = []
        for f in features:
            row = []
            for key in self.FEATURE_KEYS:
                val = f.get(key, 0.0)
                # tránh None
                if val is None:
                    val = 0.0
                row.append(float(val))
            X.append(row)
        if not X:
            return np.empty((0, len(self.FEATURE_KEYS)), dtype="float32")
        return np.array(X, dtype="float32")

    def score_batch(self, features: List[Dict[str, Any]]) -> np.ndarray:
        """
        Trả về 1 vector score [0,1] (xác suất được chọn).

        - Nếu chưa có model.pkl -> trả về 0.5 cho tất cả.
        - Nếu có model:
            + Nếu có predict_proba -> dùng cột thứ 2 (p(class=1)).
            + Nếu không có -> dùng predict (coi như score).
        """
        n = len(features)
        if not self.model:
            # fallback: chưa train model, coi như trung lập
            return np.full(shape=(n,), fill_value=0.5, dtype="float32")

        if n == 0:
            return np.empty((0,), dtype="float32")

        X = self._features_to_matrix(features)

        # sklearn classifier thường có predict_proba
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            # lấy xác suất class 1
            return proba[:, 1].astype("float32")

        # fallback: nếu model không có predict_proba
        pred = self.model.predict(X)
        # ép sang float
        return np.array(pred, dtype="float32")
