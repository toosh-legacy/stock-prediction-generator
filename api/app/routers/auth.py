from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from passlib.context import CryptContext
from app.database import get_db
from app.models.api_key import APIKey
import secrets
import uuid

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class CreateKeyRequest(BaseModel):
    name: str


class CreateKeyResponse(BaseModel):
    api_key: str
    key_id: str
    message: str = "Store this key securely — it will not be shown again."


@router.post("/keys", response_model=CreateKeyResponse, summary="Generate a new API key")
async def create_api_key(body: CreateKeyRequest, db: AsyncSession = Depends(get_db)):
    """
    Generate a new API key. The raw key is returned once and never stored in plaintext.
    """
    raw_key = f"sk-{secrets.token_urlsafe(32)}"
    key_hash = pwd_context.hash(raw_key)
    record = APIKey(id=str(uuid.uuid4()), key_hash=key_hash, name=body.name, tier="free")
    db.add(record)
    await db.commit()
    return CreateKeyResponse(api_key=raw_key, key_id=record.id)


@router.get("/keys", summary="List all API key IDs (no raw keys shown)")
async def list_api_keys(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(APIKey).where(APIKey.is_active == True))
    keys = result.scalars().all()
    return [
        {
            "key_id": k.id,
            "name": k.name,
            "tier": k.tier,
            "requests_count": k.requests_count,
            "created_at": str(k.created_at),
        }
        for k in keys
    ]


@router.delete("/keys/{key_id}", summary="Revoke an API key")
async def revoke_api_key(key_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(APIKey).where(APIKey.id == key_id))
    key = result.scalar_one_or_none()
    if not key:
        raise HTTPException(status_code=404, detail="Key not found")
    key.is_active = False
    await db.commit()
    return {"message": "Key revoked"}
