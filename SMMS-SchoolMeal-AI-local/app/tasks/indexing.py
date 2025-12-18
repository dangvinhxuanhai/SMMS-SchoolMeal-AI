# app/tasks/indexing.py
import os
import traceback
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from pathlib import Path

from app.core.config import get_settings
from app.tasks.graph_build import build_graph_for_school
from app.tasks.train_model import train_ranking_model

settings = get_settings()
local_embedder = SentenceTransformer(
    settings.EMBEDDING_MODEL_NAME
)

def embed_text(doc: str) -> list[float]:
    """
    Embed text bằng OpenAI hoặc Local model.
    """
    # ---- ưu tiên OpenAI nếu bật ----
    if settings.USE_OPENAI_EMBEDDINGS:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            res = client.embeddings.create(
                model=settings.OPENAI_EMBEDDING_MODEL,
                input=doc,
            )
            return res.data[0].embedding
        except Exception as ex:
            print("⚠ OpenAI embedding failed, fallback to local:", ex)

    # ---- fallback: LOCAL embedding (MIỄN PHÍ) ----
    return local_embedder.encode(doc).tolist()


def _get_engine():
    return create_engine(settings.SQLSERVER_CONN_STR)


def get_pending_schools():
    engine = _get_engine()
    
    sql = """
        SELECT SchoolId
        FROM school.Schools
        WHERE NeedRebuildAiIndex = 0 AND IsActive = 1
    """
    with engine.begin() as conn:
        rows = conn.execute(text(sql)).fetchall()
    return [str(r.SchoolId) for r in rows]


def build_ai_for_pending_schools():
    school_ids = get_pending_schools()
    results = []

    for sid in school_ids:
        try:
            print("\n==============================")
            print(f"[AI] START build for school {sid}")

            n_food = build_index_for_school(sid)
            n_nodes, n_edges = build_graph_for_school(sid)

            print(f"[AI] BUILD OK for school {sid}")
            print(f"     - indexed_food = {n_food}")
            print(f"     - graph_nodes  = {n_nodes}")
            print(f"     - graph_edges  = {n_edges}")

            with _get_engine().begin() as conn:
                conn.execute(
                    text("""
                        UPDATE school.Schools
                        SET NeedRebuildAiIndex = 1
                        WHERE SchoolId = :sid
                    """),
                    {"sid": sid},
                )

            print(f"[AI] UPDATED school.NeedRebuildAiIndex = 1 for {sid}")

            results.append({
                "school_id": sid,
                "status": "ok",
                "indexed_food": n_food,
                "graph_nodes": n_nodes,
                "graph_edges": n_edges,
            })

        except Exception as ex:
            print("\n❌ [AI][ERROR] build failed for school:", sid)
            print("Exception:", ex)
            print("Traceback:")
            traceback.print_exc()

            results.append({
                "school_id": sid,
                "status": "error",
                "message": str(ex),
            })

    # ---- TRAIN MODEL ----
    try:
        print("\n==============================")
        print("[AI] START training ranking model")

        train_ranking_model()

        print("[AI] TRAIN ranking model OK")

        results.append({
            "school_id": None,
            "status": "ok",
            "stage": "train_model",
            "message": "Trained ranking model successfully.",
        })

    except Exception as ex:
        print("\n❌ [AI][ERROR] training ranking model failed")
        print("Exception:", ex)
        print("Traceback:")
        traceback.print_exc()

        results.append({
            "school_id": None,
            "status": "error",
            "stage": "train_model",
            "message": str(ex),
        })

    return results


def build_index_for_school(school_id: str) -> int:
    """
    Build FAISS index + metadata cho 1 trường.
    Trả về số món ăn được index.
    """
    print(f"\n[INDEX] Start build_index_for_school: {school_id}")

    try:
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
                SELECT FoodId, SUM(Kcal) AS TotalKcal
                FROM base
                GROUP BY FoodId
            ),
            ing_agg AS (
                SELECT FoodId, STRING_AGG(IngredientName, ', ') AS IngredientNames
                FROM (
                    SELECT DISTINCT FoodId, IngredientName
                    FROM base
                ) d
                GROUP BY FoodId
            ),
            allergen_agg AS (
                SELECT FoodId, STRING_AGG(AllergenName, ', ') AS AllergenNames
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

        df = pd.read_sql(text(sql), engine, params={"school_id": school_id})
        print(f"[INDEX] Query rows = {len(df)}")

        os.makedirs(settings.DATA_DIR, exist_ok=True)

        suffix = f"_{school_id}"

        base_meta: Path = settings.METADATA_PATH
        base_index: Path = settings.FAISS_INDEX_PATH
        base_emb: Path = settings.EMBEDDINGS_PATH

        meta_path = base_meta.with_name(f"{base_meta.stem}{suffix}{base_meta.suffix}")
        index_path = base_index.with_name(f"{base_index.stem}{suffix}{base_index.suffix}")
        emb_path = base_emb.with_name(f"{base_emb.stem}{suffix}{base_emb.suffix}")

        print("[INDEX] Paths:")
        print("  meta :", meta_path)
        print("  index:", index_path)
        print("  emb  :", emb_path)

        if df.empty:
            print("[INDEX] No food data → create EMPTY index")

            dim = 3072  # text-embedding-3-large
            index = faiss.IndexFlatIP(emb.shape[1])
            emb = np.zeros((0, dim), dtype="float32")

            df.to_csv(meta_path, index=False, encoding="utf-8-sig")
            faiss.write_index(index, str(index_path))
            np.save(emb_path, emb)

            print("[INDEX] Empty index files CREATED")
            return 0

        docs = []
        for _, row in df.iterrows():
            docs.append(
                f"{row['FoodName']} - main dish: {bool(row['IsMainDish'])}, "
                f"ingredients: {row['IngredientNames'] or ''}, "
                f"allergens: {row['AllergenNames'] or ''}, "
                f"total kcal: {row['TotalKcal']}"
            )

        embeddings = []
        for i, doc in enumerate(docs):
            try:
                embeddings.append(embed_text(doc))
            except Exception as ex:
                print(f"❌ Embed failed at doc {i}")
                raise

        emb = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(emb)

        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)

        df.to_csv(meta_path, index=False, encoding="utf-8-sig")
        faiss.write_index(index, str(index_path))
        np.save(emb_path, emb)

        print(f"[INDEX] Index files CREATED for school {school_id}")
        return len(df)

    except Exception as ex:
        print("\n❌ [INDEX][FATAL ERROR] build_index_for_school failed")
        print("Exception:", ex)
        print("Traceback:")
        traceback.print_exc()
        raise
