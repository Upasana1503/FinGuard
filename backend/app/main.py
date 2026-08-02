"""
FinGuard API — activation-based guardrail for finance/cybersecurity LLM agents.

Endpoints:
    POST /v1/signup    -- create an account, get an API key (shown once)
    POST /v1/check     -- check a prompt, requires X-API-Key header
    GET  /v1/usage      -- usage summary for the calling key
    GET  /health        -- liveness check, no auth

The model loads ONCE at startup (see guardrail_core.inference.get_engine),
not per-request -- request latency is one forward pass, not a cold model load.
"""

import datetime
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from .auth import generate_api_key, get_current_api_key, hash_key
from .db import Base, engine, get_db
from .guardrail_core.inference import get_engine
from .models import ApiKey, UsageLog, User
from .rate_limit import check_rate_limit

Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine_ = get_engine()  # load model once at startup, not on first request
    # Kernel-compilation cost (MPS/CUDA backends) isn't fully paid off by a
    # single call -- empirically takes ~3 calls to settle (30s, then ~6s,
    # ~1.5s, then <100ms steady state). Absorb all of it at startup so it
    # never lands on a real user's request.
    for _ in range(3):
        engine_.check("warmup")
    yield


app = FastAPI(title="FinGuard API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # portfolio project: open CORS. Restrict to your frontend's origin in production.
    allow_methods=["*"],
    allow_headers=["*"],
)


class SignupRequest(BaseModel):
    email: EmailStr


class SignupResponse(BaseModel):
    api_key: str
    message: str = "Save this key now -- it will not be shown again."


class CheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


class PolicyAttribution(BaseModel):
    category: str
    policy_label: str
    policy_reference: str
    activation_cosine_similarity: float


class CheckResponse(BaseModel):
    flagged: bool
    flag_confidence: float
    policy_attribution: list[PolicyAttribution]
    disclaimer: str
    latency_ms: float


class UsageResponse(BaseModel):
    total_requests: int
    flagged_requests: int
    avg_latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/signup", response_model=SignupResponse)
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if user is None:
        user = User(email=payload.email)
        db.add(user)
        db.flush()

    raw_key = generate_api_key()
    api_key = ApiKey(user_id=user.id, key_hash=hash_key(raw_key), key_prefix=raw_key[:11])
    db.add(api_key)
    db.commit()

    return SignupResponse(api_key=raw_key)


@app.post("/v1/check", response_model=CheckResponse)
def check_prompt(
    payload: CheckRequest,
    api_key: ApiKey = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):
    check_rate_limit(api_key.id)

    engine_ = get_engine()
    result = engine_.check(payload.text)

    db.add(UsageLog(
        api_key_id=api_key.id,
        flagged=result["flagged"],
        latency_ms=result["latency_ms"],
    ))
    db.commit()

    return CheckResponse(**result)


@app.get("/v1/usage", response_model=UsageResponse)
def usage(
    api_key: ApiKey = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):
    logs = db.query(UsageLog).filter(UsageLog.api_key_id == api_key.id)
    total = logs.count()
    if total == 0:
        return UsageResponse(total_requests=0, flagged_requests=0, avg_latency_ms=0.0)

    flagged = logs.filter(UsageLog.flagged == True).count()  # noqa: E712
    avg_latency = db.query(func.avg(UsageLog.latency_ms)).filter(UsageLog.api_key_id == api_key.id).scalar()

    return UsageResponse(
        total_requests=total,
        flagged_requests=flagged,
        avg_latency_ms=round(float(avg_latency or 0), 2),
    )
