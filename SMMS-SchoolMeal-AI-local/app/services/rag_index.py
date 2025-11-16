# app/services/rag_index.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
import os
import faiss
from sentence_transformers import SentenceTransformer
from app.core.config import get_settings

settings = get_settings()

@dataclass
class Candidate:
    idx: int
    food_id: int
    food_name: str
    is_main_dish: bool
    total_kcal: float
    faiss_score: float

class RagIndex:
    def __init__(self, index: faiss.Index, metadata: pd.DataFrame, embedder: SentenceTransformer):
        self.index = index
        self.metadata = metadata
        self.embedder = embedder

    @classmethod
    def load_for_school(cls, school_id: str) -> "RagIndex":
        suffix = f"_{school_id}"
        meta_path  = settings.METADATA_PATH.replace(".csv",  f"{suffix}.csv")
        index_path = settings.FAISS_INDEX_PATH.replace(".faiss", f"{suffix}.faiss")

        if not os.path.exists(meta_path):
            raise RuntimeError(f"Metadata file not found for SchoolId={school_id}: {meta_path}")
        if not os.path.exists(index_path):
            raise RuntimeError(f"FAISS index file not found for SchoolId={school_id}: {index_path}")

        metadata = pd.read_csv(meta_path)
        index = faiss.read_index(index_path)
        embedder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        return cls(index=index, metadata=metadata, embedder=embedder)

    def encode_query(self, text: str) -> np.ndarray:
        emb = self.embedder.encode([text], normalize_embeddings=True)
        return emb.astype("float32")
    
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

    def search_candidates(
        self,
        query_text: str,
        k: int = 200
    ) -> List[Candidate]:
        q_emb = self.encode_query(query_text)
        scores, idxs = self.index.search(q_emb, k)
        idxs = idxs[0]
        scores = scores[0]

        candidates: List[Candidate] = []
        for i, s in zip(idxs, scores):
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