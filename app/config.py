from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    GEMINI_API_KEY: str
    CHROMA_DATA_PATH: str = "./chroma_data"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Semantic Chunking Profiles config
    CHUNK_PROFILES: dict = {
        "standard": {"size": 1000, "overlap": 200},
        "legal": {"size": 800, "overlap": 300},
        "technical": {"size": 1500, "overlap": 150},
    }


settings = Settings()
