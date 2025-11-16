# scripts/train_model.py
"""
Train ranking model for AI menu recommendation.

- Đọc dữ liệu từ:
    rag.MenuRecommendSessions (SessionId, UserId, RequestJson, ...)
    rag.MenuRecommendResults  (SessionId, FoodId, IsMain, IsChosen, ...)

- Dùng lại:
    + RAG index (embeddings + metadata)
    + Ingredient graph

- Build features cho mỗi (SessionId, FoodId):
    faiss_score:  q_emb · food_emb
    graph_score:  overlap ingredient giữa món và input
    total_kcal:   từ metadata (food_metadata.csv)

- Label:
    IsChosen (0/1)

- Train:
    sklearn pipeline: StandardScaler + LogisticRegression

- Save:
    model vào settings.ML_MODEL_PATH (vd: data/ranking_model.pkl)
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from app.core.config import get_settings
from app.services.rag_index import RagIndex
from app.services.graph_service import IngredientGraph

settings = get_settings()


def load_db_engine():
    """Tạo SQLAlchemy engine kết nối SQL Server."""
    engine = create_engine(settings.SQLSERVER_CONN_STR, pool_pre_ping=True)
    return engine


def fetch_logs(engine) -> pd.DataFrame:
    """
    Lấy log từ rag.MenuRecommendSessions + rag.MenuRecommendResults.

    Anh có thể thêm WHERE CreatedAt >= ... nếu muốn giới hạn theo thời gian.
    """
    sql = text(
        """
        SELECT
            r.SessionId,
            r.FoodId,
            r.IsMain,
            r.IsChosen,
            s.RequestJson
        FROM rag.MenuRecommendResults r
        JOIN rag.MenuRecommendSessions s
            ON r.SessionId = s.SessionId
        -- WHERE s.CreatedAt >= DATEADD(DAY, -60, SYSDATETIME())  -- optional, giới hạn 60 ngày gần nhất
        """
    )

    df = pd.read_sql(sql, engine)
    return df


def build_food_index_maps(rag_index: RagIndex) -> Dict[int, int]:
    """
    Tạo map FoodId -> index trong embeddings.

    rag_index.metadata: DataFrame với cột FoodId (trùng với DB)
    embeddings: đọc từ EMBEDDINGS_PATH
    """
    meta = rag_index.metadata
    foodid_to_idx: Dict[int, int] = {}
    for i, row in meta.iterrows():
        foodid_to_idx[int(row["FoodId"])] = i
    return foodid_to_idx


def get_total_kcal_for_food(food_id: int, metadata: pd.DataFrame) -> float:
    row = metadata.loc[metadata["FoodId"] == food_id]
    if row.empty:
        return 0.0
    val = row["TotalKcal"].values[0]
    if pd.isna(val):
        return 0.0
    return float(val)


def parse_request_json(request_json: str) -> Dict[str, Any]:
    """Parse JSON send từ .NET sang Python lúc recommend."""
    try:
        data = json.loads(request_json)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}


def build_dataset(
    df_logs: pd.DataFrame,
    rag_index: RagIndex,
    graph: IngredientGraph,
    food_embeddings: np.ndarray,
) -> (np.ndarray, np.ndarray):
    """
    Từ log -> build X, y.

    X: [faiss_score, graph_score, total_kcal]
    y: IsChosen
    """
    metadata = rag_index.metadata
    foodid_to_idx = build_food_index_maps(rag_index)

    # cache theo SessionId: query embedding + ingredient list
    session_cache: Dict[Any, Dict[str, Any]] = {}

    X_rows: List[List[float]] = []
    y_rows: List[int] = []

    for _, row in df_logs.iterrows():
        session_id = row["SessionId"]
        food_id = int(row["FoodId"])
        is_main = bool(row["IsMain"])
        is_chosen = bool(row["IsChosen"])
        request_json = row["RequestJson"]

        # Lấy context cho session
        if session_id not in session_cache:
            req_obj = parse_request_json(request_json)

            main_ing_ids = req_obj.get("main_ingredient_ids", []) or []
            side_ing_ids = req_obj.get("side_ingredient_ids", []) or []
            avoid_allergens = req_obj.get("avoid_allergen_ids", []) or []
            max_main_kcal = req_obj.get("max_main_kcal")
            max_side_kcal = req_obj.get("max_side_kcal")

            # build query_text & embedding giống lúc recommend
            query_text = rag_index.build_query_text(
                main_ing_ids,
                side_ing_ids,
                avoid_allergens,
                max_main_kcal,
                max_side_kcal,
            )
            q_emb = rag_index.encode_query(query_text)[0]  # shape (d,)

            session_cache[session_id] = {
                "q_emb": q_emb,
                "main_ing_ids": main_ing_ids,
                "side_ing_ids": side_ing_ids,
            }

        sctx = session_cache[session_id]

        # tìm index của food trong embedding
        if food_id not in foodid_to_idx:
            # món không có trong metadata/index -> bỏ qua
            continue
        idx = foodid_to_idx[food_id]
        food_vec = food_embeddings[idx]

        # faiss_score = dot(q_emb, food_emb)
        q_emb = sctx["q_emb"]
        faiss_score = float(np.dot(q_emb, food_vec))

        # graph_score: overlap nguyên liệu theo main/side
        ing_ids = sctx["main_ing_ids"] if is_main else sctx["side_ing_ids"]
        graph_score = float(graph.ingredient_overlap_score(food_id, ing_ids))

        # total_kcal từ metadata
        total_kcal = get_total_kcal_for_food(food_id, metadata)

        X_rows.append([faiss_score, graph_score, total_kcal])
        y_rows.append(1 if is_chosen else 0)

    if not X_rows:
        raise RuntimeError("Không tạo được sample nào để train (X_rows empty).")

    X = np.array(X_rows, dtype="float32")
    y = np.array(y_rows, dtype="int32")
    return X, y


def train_and_save_model(X: np.ndarray, y: np.ndarray):
    """Train LogisticRegression + save model ra file."""
    # kiểm tra đủ 2 class chưa
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        print("⚠️  Không đủ dữ liệu 2 class (IsChosen) để train model. Bỏ qua.")
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Pipeline: chuẩn hóa + LogisticRegression
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_proba)
        print(f"Validation AUC: {auc:.4f}")
    else:
        y_pred = model.predict(X_val)
        auc = roc_auc_score(y_val, y_pred)
        print(f"Validation AUC (dùng predict): {auc:.4f}")

    os.makedirs(settings.DATA_DIR, exist_ok=True)
    joblib.dump(model, settings.ML_MODEL_PATH)
    print(f"✅ Saved model to {settings.ML_MODEL_PATH}")


def main():
    print("=== Train ranking model for menu recommendation ===")

    # 1. Kết nối DB & lấy log
    engine = load_db_engine()
    df_logs = fetch_logs(engine)
    print(f"Loaded {len(df_logs)} log rows from rag.MenuRecommend*")

    # 2. Load RAG index + graph + embeddings
    print("Loading RAG index, metadata & embeddings...")
    rag_index = RagIndex.load()
    graph = IngredientGraph.load()
    food_embeddings = np.load(settings.EMBEDDINGS_PATH)

    # 3. Build dataset X, y
    X, y = build_dataset(df_logs, rag_index, graph, food_embeddings)
    print(f"Built dataset: X shape = {X.shape}, positive rate = {y.mean():.3f}")

    # 4. Train & save model
    train_and_save_model(X, y)


if __name__ == "__main__":
    main()
