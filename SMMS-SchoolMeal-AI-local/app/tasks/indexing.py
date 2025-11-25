# app/tasks/indexing.py
import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from app.core.config import get_settings

settings = get_settings()


def _get_engine():
    return create_engine(settings.SQLSERVER_CONN_STR)


def build_index_for_school(school_id: str) -> int:
    """
    Build FAISS index + metadata cho 1 trường.
    Trả về số món ăn được index.
    """
    engine = _get_engine()

    sql = """
        WITH base AS (
            SELECT
                f.FoodId,
                f.FoodName,
                f.IsMainDish,
                f.SchoolId,
                fi.IngredientId,
                ing.IngredientName,
                ai.AllergenId,
                al.AllergenName,
                fi.QuantityGram * ing.EnergyKcal / 100.0 AS Kcal
            FROM nutrition.FoodItems f
            JOIN nutrition.FoodItemIngredients fi
                ON fi.FoodId = f.FoodId
            JOIN nutrition.Ingredients ing
                ON ing.IngredientId = fi.IngredientId
            LEFT JOIN nutrition.AllergeticIngredients ai
                ON ai.IngredientId = ing.IngredientId
            LEFT JOIN nutrition.Allergens al
                ON al.AllergenId = ai.AllergenId
            WHERE f.IsActive = 1
            AND f.SchoolId = :school_id
        ),
        kcal AS (
            SELECT
                FoodId,
                SUM(Kcal) AS TotalKcal
            FROM base
            GROUP BY FoodId
        ),
        ing_agg AS (
            SELECT
                FoodId,
                STRING_AGG(IngredientName, ', ') AS IngredientNames
            FROM (
                SELECT DISTINCT FoodId, IngredientName
                FROM base
            ) d
            GROUP BY FoodId
        ),
        allergen_agg AS (
            SELECT
                FoodId,
                STRING_AGG(AllergenName, ', ') AS AllergenNames
            FROM (
                SELECT DISTINCT FoodId, AllergenName
                FROM base
                WHERE AllergenName IS NOT NULL
            ) d
            GROUP BY FoodId
        )
        SELECT
            f.FoodId,
            f.FoodName,
            f.IsMainDish,
            f.SchoolId,
            ia.IngredientNames,
            aa.AllergenNames,
            k.TotalKcal
        FROM nutrition.FoodItems f
        LEFT JOIN kcal         k  ON k.FoodId  = f.FoodId
        LEFT JOIN ing_agg      ia ON ia.FoodId = f.FoodId
        LEFT JOIN allergen_agg aa ON aa.FoodId = f.FoodId
        WHERE f.IsActive = 1
        AND f.SchoolId = :school_id
        ORDER BY f.FoodId;
        """

    df = pd.read_sql(text(sql), _get_engine(), params={"school_id": school_id})

    if df.empty:
        return 0

    # build docs cho embedder
    docs = []
    for _, row in df.iterrows():
        ing = row["IngredientNames"] or ""
        aller = row["AllergenNames"] or ""
        text_doc = (
            f"{row['FoodName']} - main dish: {bool(row['IsMainDish'])}, "
            f"ingredients: {ing}, allergens: {aller}, "
            f"total kcal: {row['TotalKcal']}"
        )
        docs.append(text_doc)

    model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    emb = model.encode(docs, normalize_embeddings=True).astype("float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    os.makedirs(settings.DATA_DIR, exist_ok=True)
    suffix = f"_{school_id}"
    meta_path = settings.METADATA_PATH.replace(".csv", f"{suffix}.csv")
    index_path = settings.FAISS_INDEX_PATH.replace(".faiss", f"{suffix}.faiss")
    emb_path = settings.EMBEDDINGS_PATH.replace(".npy", f"{suffix}.npy")

    df.to_csv(meta_path, index=False, encoding="utf-8-sig")
    faiss.write_index(index, index_path)
    np.save(emb_path, emb)

    return len(df)
