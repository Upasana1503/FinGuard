import hashlib
import secrets

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from .db import get_db
from .models import ApiKey

KEY_PREFIX = "fg_"


def generate_api_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def get_current_api_key(
    x_api_key: str = Header(..., description="Your FinGuard API key"),
    db: Session = Depends(get_db),
) -> ApiKey:
    key_hash = hash_key(x_api_key)
    api_key = db.query(ApiKey).filter(ApiKey.key_hash == key_hash, ApiKey.is_active == True).first()  # noqa: E712
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    return api_key
