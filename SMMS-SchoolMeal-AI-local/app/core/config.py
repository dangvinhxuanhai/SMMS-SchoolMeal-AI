# app/core/config.py
import os
from functools import lru_cache
from pathlib import Path


class Settings:
    # =========================
    # DATABASE (SQL SERVER)
    # =========================
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: str = os.getenv("DB_PORT", "1433")
    DB_NAME: str = os.getenv("DB_NAME", "EduMeal")
    DB_USER: str = os.getenv("DB_USER", "sa")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "YourStrong@Passw0rd1")
    DB_DRIVER: str = os.getenv(
        "DB_DRIVER",
        "ODBC Driver 18 for SQL Server"
    )

    @property
    def SQLSERVER_CONN_STR(self) -> str:
        """
        SQLAlchemy + pyodbc connection string
        """
        return (
            f"mssql+pyodbc://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST},{self.DB_PORT}/{self.DB_NAME}"
            f"?driver={self.DB_DRIVER.replace(' ', '+')}"
            f"&TrustServerCertificate=yes"
            f"&Encrypt=no"
        )

    # =========================
    # DATA / AI FILE PATHS
    # =========================
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "data"))

    FAISS_INDEX_PATH: Path = DATA_DIR / "food_index.faiss"
    EMBEDDINGS_PATH: Path = DATA_DIR / "food_embeddings.npy"
    METADATA_PATH: Path = DATA_DIR / "food_metadata.csv"
    ML_MODEL_PATH: Path = DATA_DIR / "ranking_model.pkl"

    # =========================
    # AI / EMBEDDING
    # =========================
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2"
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
