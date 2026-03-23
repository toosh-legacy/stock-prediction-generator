from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app.config import get_settings

settings = get_settings()

# Support both Postgres (production) and SQLite (local dev without Docker)
def _make_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url  # sqlite+aiosqlite:// or already async-prefixed

_url = _make_url(settings.database_url)
_is_sqlite = _url.startswith("sqlite")

engine = create_async_engine(
    _url,
    echo=settings.environment == "development",
    # pool_pre_ping not supported by SQLite async driver
    **({} if _is_sqlite else {"pool_pre_ping": True}),
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
