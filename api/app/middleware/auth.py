from fastapi import Security, HTTPException, status, Depends
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from passlib.context import CryptContext
from app.models.api_key import APIKey
from app.database import get_db

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


async def verify_api_key(
    api_key: str = Security(API_KEY_HEADER),
    db: AsyncSession = Depends(get_db)
) -> APIKey:
    """Verify API key by comparing bcrypt hash against all active keys."""
    result = await db.execute(select(APIKey).where(APIKey.is_active == True))
    keys = result.scalars().all()
    for key_record in keys:
        if pwd_context.verify(api_key, key_record.key_hash):
            return key_record
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key. Get one at /v1/auth/keys",
        headers={"WWW-Authenticate": "ApiKey"},
    )
