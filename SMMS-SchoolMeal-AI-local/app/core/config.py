# app/core/config.py
import os
from functools import lru_cache

class Settings:
    # đọc từ env để dễ deploy
    SQLSERVER_CONN_STR: str = os.getenv(
        "SQLSERVER_CONN_STR",
        "mssql+pyodbc://haidang:123@XUANHAI/EduMeal"
        "?driver=ODBC+Driver+17+for+SQL+Server"
        "&TrustServerCertificate=yes"
    )

    # đường dẫn file index/metadata
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    FAISS_INDEX_PATH: str = os.path.join(DATA_DIR, "food_index.faiss")
    EMBEDDINGS_PATH: str = os.path.join(DATA_DIR, "food_embeddings.npy")
    METADATA_PATH: str = os.path.join(DATA_DIR, "food_metadata.csv")
    ML_MODEL_PATH: str = os.path.join(DATA_DIR, "ranking_model.pkl")

    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings()
