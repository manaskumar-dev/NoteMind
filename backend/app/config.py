from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All secrets/config live in .env — never hardcoded, never logged."""

    secret_key: str
    access_token_expire_minutes: int = 60
    database_url: str = "sqlite:///./notes.db"
    gemini_api_key: str
    gemini_model: str = "gemini-3.1-flash-lite"
    cors_origins: str = "http://localhost:5500"
    max_pdf_size_mb: int = 5
    ai_rate_limit_per_minute: int = 5

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    class Config:
        env_file = ".env"


settings = Settings()
