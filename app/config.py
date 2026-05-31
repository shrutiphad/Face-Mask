

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
  
    
    qdrant_url: str = "http://localhost:6333"

    # Required only for Qdrant Cloud — leave empty for local/docker
    qdrant_api_key: str = ""

    # The Qdrant collection we create at startup
    collection_name: str = "faces"


    insightface_model_pack: str = "buffalo_s"

    # Where InsightFace stores downloaded model files
    insightface_model_dir: str = "/tmp/insightface_models"

    
    det_size: int = 640

    match_threshold: float = 0.35

   
    # 128 gives >99 % recall@1 for collections under 10 M vectors.
    hnsw_ef_search: int = 128

    # Number of results to return per /search call
    top_k: int = 1

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# lru_cache means Settings() is instantiated exactly once per process.
# Subsequent calls return the same object — no repeated disk I/O for .env.
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()