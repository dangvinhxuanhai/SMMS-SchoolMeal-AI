# app/services/rag_index.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

from app.core.config import get_settings

settings = get_settings()
_local_embedder = SentenceTransformer(
    settings.EMBEDDING_MODEL_NAME
)

@dataclass
class Candidate:
    idx: int
    food_id: int
    food_name: str
    is_main_dish: bool
    total_kcal: float
    faiss_score: float

class RagIndex:
    def __init__(self, index: faiss.Index, metadata: pd.DataFrame):
        self.index = index
        self.metadata = metadata

    def _meta_path_for_school(school_id: str) -> str:
        base: Path = settings.METADATA_PATH
        return str(base.with_name(f"{base.stem}_{school_id}{base.suffix}"))


    def _index_path_for_school(school_id: str) -> str:
        base: Path = settings.FAISS_INDEX_PATH
        return str(base.with_name(f"{base.stem}_{school_id}{base.suffix}"))

    @classmethod
    def load_for_school(cls, school_id: str) -> "RagIndex":
        meta_path = cls._meta_path_for_school(school_id)
        index_path = cls._index_path_for_school(school_id)

        if not os.path.exists(meta_path) or not os.path.exists(index_path):
            raise RuntimeError("Missing FAISS index or metadata")

        metadata = pd.read_csv(meta_path)
        index = faiss.read_index(index_path)
        return cls(index=index, metadata=metadata)

    @classmethod
    def try_load_for_school(cls, school_id: str):
        try:
            return cls.load_for_school(school_id)
        except Exception:
            return None

    def encode_query(self, text: str) -> np.ndarray:
        # Encode query bằng local SentenceTransformer
        emb = _local_embedder.encode(text)
        # đảm bảo đúng dtype cho FAISS
        emb = np.array(emb, dtype="float32")
        # normalize L2 (giống FAISS IndexFlatIP)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        return emb.reshape(1, -1)


    def build_query_text(
        self,
        main_ingredient_ids: List[int],
        side_ingredient_ids: List[int],
        avoid_allergen_ids: List[int],
        max_main_kcal: Optional[float],
        max_side_kcal: Optional[float],
    ) -> str:
        # MVP: chỉ encode vài thông tin cơ bản; sau này anh có thể enrich thêm
        return (
            f"Main dish with ingredients ids {main_ingredient_ids}, "
            f"side dish with ingredients ids {side_ingredient_ids}, "
            f"avoid allergens {avoid_allergen_ids}, "
            f"max main kcal {max_main_kcal}, max side kcal {max_side_kcal}."
        )

    def search_candidates(self, query_text: str, k: int = 200) -> List[Candidate]:
        q_emb = self.encode_query(query_text)
        scores, idxs = self.index.search(q_emb, k)

        candidates: List[Candidate] = []
        for i, s in zip(idxs[0], scores[0]):
            if i < 0:
                continue
            row = self.metadata.iloc[i]
            candidates.append(
                Candidate(
                    idx=int(i),
                    food_id=int(row["FoodId"]),
                    food_name=str(row["FoodName"]),
                    is_main_dish=bool(row["IsMainDish"]),
                    total_kcal=float(row["TotalKcal"]) if not pd.isna(row["TotalKcal"]) else 0.0,
                    faiss_score=float(s),
                )
            )
        return candidates