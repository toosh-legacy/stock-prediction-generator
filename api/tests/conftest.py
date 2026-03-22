"""
pytest configuration and shared fixtures.
Uses an in-memory SQLite database so no external Postgres is needed for tests.
"""
import os
import pytest
import pytest_asyncio

# Override env vars before any app module is imported
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-testing-only")
os.environ.setdefault("API_KEY_SALT", "test-salt")
os.environ.setdefault("REDIS_URL", "")  # Disable redis in tests

from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

from app.main import app
from app.database import Base, get_db
from app.models.api_key import APIKey
from passlib.context import CryptContext
import uuid

TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


async def override_get_db():
    async with TestSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


app.dependency_overrides[get_db] = override_get_db


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_database():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()


@pytest_asyncio.fixture
async def db_session():
    async with TestSessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def valid_api_key(db_session: AsyncSession) -> str:
    """Create a test API key and return the raw key string."""
    raw_key = "sk-test-valid-key-12345"
    key_hash = pwd_context.hash(raw_key)
    record = APIKey(id=str(uuid.uuid4()), key_hash=key_hash, name="Test Key", tier="free")
    db_session.add(record)
    await db_session.commit()
    return raw_key


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
