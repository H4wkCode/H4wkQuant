"""
H4wkQuant - Authentication System
JWT (HMAC-SHA256) auth with no external dependencies.
Default user: admin/admin123
"""
import hashlib
import hmac
import base64
import json
import time
import secrets
from typing import Optional

import redis.asyncio as redis
from fastapi import Depends, HTTPException, Request
from loguru import logger

from shared.config.settings import settings


SECRET_KEY = settings.panel_secret_key.encode()
TOKEN_EXPIRY = 86400  # 24 hours


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
    return f"{salt}:{h.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, hashed = stored.split(":")
        h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
        return hmac.compare_digest(h.hex(), hashed)
    except Exception:
        return False


def create_jwt(payload: dict) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).rstrip(b"=").decode()
    payload["exp"] = int(time.time()) + TOKEN_EXPIRY
    payload["iat"] = int(time.time())
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    signing_input = f"{header}.{body}"
    sig = hmac.new(SECRET_KEY, signing_input.encode(), hashlib.sha256).digest()
    signature = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
    return f"{header}.{body}.{signature}"


def decode_jwt(token: str) -> Optional[dict]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header, body, signature = parts
        signing_input = f"{header}.{body}"
        expected_sig = hmac.new(SECRET_KEY, signing_input.encode(), hashlib.sha256).digest()
        expected = base64.urlsafe_b64encode(expected_sig).rstrip(b"=").decode()
        if not hmac.compare_digest(expected, signature):
            return None
        # Pad base64
        padded = body + "=" * (4 - len(body) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


async def ensure_default_user(redis_client: redis.Redis):
    """Create default admin user if not exists"""
    users = await redis_client.hgetall("q:panel:users")
    if not users:
        hashed = hash_password("admin123")
        await redis_client.hset("q:panel:users", "admin", hashed)
        logger.info("Default admin user created")


async def get_current_user(request: Request) -> dict:
    """FastAPI dependency for auth"""
    auth_header = request.headers.get("Authorization", "")
    token = None
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
    if not token:
        token = request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Token required")
    payload = decode_jwt(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload
