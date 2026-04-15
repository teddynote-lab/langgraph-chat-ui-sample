import os

import jwt
from langgraph_sdk import Auth

JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"

auth = Auth()


@auth.authenticate
async def authenticate(authorization: str | None):
    """JWT 토큰 검증"""

    if not authorization:
        raise Auth.exceptions.HTTPException(status_code=401,
                                            detail="Unauthorized")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise Auth.exceptions.HTTPException(status_code=401,
                                            detail="Invalid token")

    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.InvalidTokenError:
        raise Auth.exceptions.HTTPException(status_code=401,
                                            detail="Invalid token")

    return {
        "identity": payload.get("sub"),
        "role": payload.get("role"),
        "email": payload.get("email", ""),
    }

@auth.on.threads.create
@auth.on.threads.read
@auth.on.threads.update
@auth.on.threads.delete
@auth.on.threads.search
async def filter_by_owner(ctx: Auth.types.AuthContext, value: dict) -> dict:
    """사용자별 스레드 격리"""
    metadata = value.setdefault("metadata", {})
    metadata["owner"] = ctx.user.identity
    return {"owner": ctx.user.identity}

