# scripts/build_index.py
import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

from app.core.config import get_settings

settings = get_settings()


def fetch_foods(engine, school_id: str) -> pd.DataFrame:
    """
    Lấy món ăn + tổng kcal + danh sách nguyên liệu + danh sách dị ứng liên quan.
    """
    sql = """
    SELECT
        f.FoodId,
        f.FoodName,
        f.IsMainDish,
        f.SchoolId,
        STRING_AGG(ing.IngredientName, ', ')        AS IngredientNames,
        STRING_AGG(al.AllergenName, ', ')           AS AllergenNames,
        SUM(fi.QuantityGram * ing.EnergyKcal / 100.0)        AS TotalKcal
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
    GROUP BY f.FoodId, f.FoodName, f.IsMainDish, f.SchoolId
    ORDER BY f.FoodId;
    """
    return pd.read_sql(text(sql), engine, params={"school_id": school_id})


def build_docs(df: pd.DataFrame) -> list[str]:
    """
    Mỗi món -> 1 câu mô tả đầy đủ: tên, main/side, kcal, ingredients, allergens.
    """
    docs: list[str] = []
    for _, row in df.iterrows():
        ing_names = row.get("IngredientNames") or ""
        all_names = row.get("AllergenNames") or ""

        parts = [
            f"Food: {row['FoodName']}.",
            f"Main dish: {bool(row['IsMainDish'])}.",
            f"Total calories: {row['TotalKcal']}.",
        ]
        if ing_names:
            parts.append(f"Ingredients: {ing_names}.")
        if all_names:
            parts.append(f"May contain allergens: {all_names}.")

        docs.append(" ".join(parts))

    return docs


def main():
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    engine = create_engine(settings.SQLSERVER_CONN_STR)

    school_id = os.getenv("SCHOOL_ID")
    if not school_id:
        raise RuntimeError("Vui lòng set biến môi trường SCHOOL_ID (GUID của trường)")

    df = fetch_foods(engine, school_id)
    print(f"Fetched {len(df)} foods for SchoolId = {school_id}")
    if df.empty:
        print("⚠️  Không có món ăn nào để build index cho trường này.")
        return

    docs = build_docs(df)

    model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    emb = model.encode(docs, normalize_embeddings=True).astype("float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    suffix = f"_{school_id}"
    meta_path = settings.METADATA_PATH.replace(".csv", f"{suffix}.csv")
    index_path = settings.FAISS_INDEX_PATH.replace(".faiss", f"{suffix}.faiss")
    emb_path = settings.EMBEDDINGS_PATH.replace(".npy", f"{suffix}.npy")

    df.to_csv(meta_path, index=False, encoding="utf-8-sig")
    faiss.write_index(index, index_path)
    np.save(emb_path, emb)

    print(f"✅ Built index for {len(df)} foods.")
    print(f"  - Metadata:    {meta_path}")
    print(f"  - FAISS index: {index_path}")
    print(f"  - Embeddings:  {emb_path}")


if __name__ == "__main__":
    main()
