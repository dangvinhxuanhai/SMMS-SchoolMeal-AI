# app/core/config.py
import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus  # <--- THÊM DÒNG NÀY
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # =========================
    # DATABASE (SQL SERVER)
    # =========================
    DB_HOST: str = os.getenv("DB_HOST", "XUANHAI") 
    DB_PORT: str = os.getenv("DB_PORT", "1433") 
    DB_NAME: str = os.getenv("DB_NAME", "EduMeal")
    DB_USER: str = os.getenv("DB_USER", "haidang") 
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "123")
    DB_DRIVER: str = os.getenv(
        "DB_DRIVER",
        "ODBC Driver 17 for SQL Server"
    )
    OPENAI_API_KEY: str = os.getenv(
        "OPENAI_API_KEY",
        ""  # fallback an toàn
    )
    OPENAI_EMBEDDING_MODEL: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "davinci-codex"
    )
    USE_OPENAI_EMBEDDINGS: bool = (
        os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
    )
    
    @property
    def SQLSERVER_CONN_STR(self) -> str:
        """
        SQLAlchemy + pyodbc connection string
        """
        # Tự động mã hóa user và password (xử lý @, /, :, v.v.)
        encoded_user = quote_plus(self.DB_USER)
        encoded_password = quote_plus(self.DB_PASSWORD)
        
        # Mã hóa tên driver (thay khoảng trắng bằng +)
        encoded_driver = quote_plus(self.DB_DRIVER)

        return (
            f"mssql+pyodbc://{encoded_user}:{encoded_password}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
            f"?driver={encoded_driver}"
            f"&Encrypt=no"
            f"&TrustServerCertificate=yes"
            f"&Connection Timeout=30"
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
