"""ParkSence API server — replaces LM Studio for parking sign analysis."""
import base64
import os
from datetime import date as DateType
from io import BytesIO

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy.orm import Session
from PIL import Image

import models
from models import VehicleType
import auth
import analyzer
from database import engine, get_db

# Create tables on startup
models.Base.metadata.create_all(bind=engine)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    if analyzer._IS_APPLE_SILICON:
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, analyzer._load_local_model)
    else:
        import logging
        logging.getLogger(__name__).info(
            "Non-Apple-Silicon host — mlx-vlm skipped. "
            f"Ollama ({analyzer.OLLAMA_MODEL}) or HF cloud will be used."
        )
    yield

app = FastAPI(title="ParkSence API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth schemas ──────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    vehicle_type: VehicleType = VehicleType.car
    is_disabled: bool = False
    has_resident_permit: bool = False
    resident_zone: str = ""

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ProfileUpdate(BaseModel):
    vehicle_type: VehicleType | None = None
    is_disabled: bool | None = None
    has_resident_permit: bool | None = None
    resident_zone: str | None = None


class UserOut(BaseModel):
    id: int
    email: str
    vehicle_type: VehicleType
    is_disabled: bool
    has_resident_permit: bool
    resident_zone: str

    model_config = {"from_attributes": True, "use_enum_values": True}


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.post("/api/auth/register", response_model=TokenOut, status_code=201)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.email == req.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = models.User(
        email=req.email,
        hashed_password=auth.hash_password(req.password),
        vehicle_type=req.vehicle_type,
        is_disabled=req.is_disabled,
        has_resident_permit=req.has_resident_permit,
        resident_zone=req.resident_zone,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = auth.create_access_token(user.id)
    return TokenOut(access_token=token, user=UserOut.model_validate(user))


@app.post("/api/auth/login", response_model=TokenOut)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == req.email).first()
    if not user or not auth.verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = auth.create_access_token(user.id)
    return TokenOut(access_token=token, user=UserOut.model_validate(user))


@app.get("/api/auth/me", response_model=UserOut)
def get_me(current_user: models.User = Depends(auth.get_current_user)):
    return current_user


@app.put("/api/auth/me", response_model=UserOut)
def update_me(
    req: ProfileUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    if req.vehicle_type is not None:
        current_user.vehicle_type = req.vehicle_type
    if req.is_disabled is not None:
        current_user.is_disabled = req.is_disabled
    if req.has_resident_permit is not None:
        current_user.has_resident_permit = req.has_resident_permit
    if req.resident_zone is not None:
        current_user.resident_zone = req.resident_zone
    db.commit()
    db.refresh(current_user)
    return current_user


# ── Analysis endpoint ─────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    day: str = Form(...),
    time: str = Form(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    """Analyze a parking sign image for the authenticated user."""
    # Read + resize image
    raw = await image.read()
    img = Image.open(BytesIO(raw)).convert("RGB")

    max_dim = 1024
    if max(img.width, img.height) > max_dim:
        scale = max_dim / max(img.width, img.height)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    image_b64 = base64.b64encode(buf.getvalue()).decode()

    today = DateType.today().strftime("%-d %B %Y")

    try:
        result = analyzer.analyze_image(
            image_b64=image_b64,
            day=day,
            time=time,
            date=today,
            vehicle_type=current_user.vehicle_type.value,
            is_disabled=current_user.is_disabled,
            has_resident_permit=current_user.has_resident_permit,
            resident_zone=current_user.resident_zone,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}
