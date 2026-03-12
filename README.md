# ParkSence

**Never guess a parking sign again.** Point your phone at any Swedish parking sign, and ParkSence tells you instantly - can you park here, right now, for your vehicle.

## Overview

ParkSence is a three-part system built for anyone navigating Stockholm's (or any Swedish city's) maze of parking signs. Native Android and iOS apps stream the camera feed, lock onto the sign, and fire it to a FastAPI backend that runs a multimodal vision model. The model reads every sign, plate, and symbol on the pole and returns a simple verdict: park / don't park / uncertain - personalised to your vehicle type, disability status, and resident permit zone.

## Short Demo

https://github.com/user-attachments/assets/59de0738-8cb1-4dbb-bc36-f4b1cd2fe975

---

## Features

- **Instant Sign Analysis** - Point, lock, and get a verdict in seconds using a local VLM (Qwen3-VL)
- **Personalised Verdicts** - Accounts for vehicle type (car, EV, motorcycle, truck, bus), disability permits, and resident zone
- **Swedish Parking Rules** - Understands the full 3-tier time format (weekday / Saturday / Sunday), zone extent plates, hard-block signs (♿, EV, Boende, Taxi, etc.)
- **Local-First Inference** - Runs Qwen3-VL-8B 4-bit quantized via mlx-vlm natively on Apple Silicon, zero cloud cost
- **HF Cloud Fallback** - Automatically falls back to Qwen2.5-VL-72B on Hugging Face if the local model is unavailable
- **Auth & Profiles** - JWT-based register/login, persistent vehicle profile stored server-side
- **Native Android App** - CameraX live preview with a scan overlay, haptic feedback, and a clean result card
- **Native iOS App** - AVFoundation camera pipeline, SwiftUI overlay, full feature parity with Android

---

## How It Works

1. **Android / iOS app** captures a frame when the camera locks on a sign, attaches the current day and time, and POSTs it to the server.
2. **FastAPI server** receives the image, resizes it, and passes it to the analyzer along with the user's vehicle profile pulled from the database.
3. **Analyzer** runs a structured two-phase prompt: first listing every sign/plate visible, then applying Swedish parking rules step-by-step.
4. **VLM** (local mlx or HF cloud) returns JSON - `signs`, `notes`, `can_park`, `message` - which the app renders as a colour-coded verdict card.

---

## Project Structure

```
server/
  main.py         # FastAPI app - auth + /api/analyze endpoint
  analyzer.py     # VLM inference (local mlx + HF cloud fallback)
  auth.py         # JWT helpers (bcrypt + jose)
  models.py       # SQLAlchemy ORM models (User, VehicleType)
  database.py     # SQLite engine + session factory
  pyproject.toml  # Server dependencies
android/
  app/src/main/java/com/parksence/
    MainActivity.kt         # Camera + scan flow
    api/ApiClient.kt        # HTTP client
    auth/                   # Login / Register / Profile screens
    detection/ColorDetector.kt
    parser/ParkingParser.kt
    classifier/SignClassifier.kt
ios/ParkSence/ParkSence/
  MainView.swift            # Camera + scan flow
  API/ApiClient.swift       # URLSession HTTP client
  Auth/                     # Login / Register / Profile screens
  Camera/                   # AVFoundation pipeline
  Detection/ColorDetector.swift
  Parser/ParkingParser.swift
  Classifier/SignClassifier.swift
  UI/                       # ScanOverlayView, VerdictCard, DesignSystem
desktop/          # Deprecated Streamlit prototype
```

---

## Requirements

- Python 3.11+ with `uv`
- Apple Silicon Mac (for local mlx inference) **or** a Hugging Face token (for cloud fallback)
- Android Studio / Android 8.0+ device
- Xcode 16+ / iOS 16+ device (for the iOS app)

---

## Installation

### Server

```sh
git clone https://github.com/VAnkataBot/ParkSence.git
cd ParkSence/server
```

Install dependencies:

```sh
uv venv .venv
uv sync
```

Activate the environment:

```sh
source .venv/bin/activate
```

The server auto-detects which inference backend to use — no manual setup required:

| OS | Backend | Setup needed |
|---|---|---|
| macOS Apple Silicon | mlx-vlm (automatic) | Nothing — model downloads on first run |
| Linux / Windows / macOS Intel | transformers (automatic) | Nothing — model downloads on first run |
| Any OS | HF cloud fallback | Set `HF_TOKEN` env var |

On Linux/Windows the server downloads `Qwen/Qwen3-VL-8B-Instruct` (~16 GB fp16, or ~5 GB with 4-bit via `bitsandbytes`) to `server/model/` on first run. A CUDA-capable GPU is recommended; CPU inference works but is slow.

Configure environment variables (optional):

```env
LLM_MODEL=mlx-community/Qwen3-VL-8B-Instruct-4bit   # macOS: override mlx model
TRANSFORMERS_MODEL=Qwen/Qwen3-VL-8B-Instruct          # Linux/Windows: override transformers model
HF_TOKEN=hf_...                                        # enables cloud fallback
HF_MODEL=Qwen/Qwen2.5-VL-72B-Instruct                 # cloud fallback model
SECRET_KEY=your-secret-key                             # JWT signing key
```

Start the server:

```sh
uvicorn main:app --host 0.0.0.0 --port 8000
```

On first run the model (~5–16 GB depending on quantization) is downloaded automatically to `server/model/`.

### Android App

1. Open `android/` in Android Studio.
2. In `ApiClient.kt`, set `serverUrl` to your server's address.
3. Build and run on a physical device (camera required).

### iOS App

1. Open `ios/ParkSence/ParkSence.xcodeproj` in Xcode.
2. In `API/ApiClient.swift`, set `serverUrl` to your server's address.
3. Build and run on a physical device (camera required for scanning).

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/register` | Create account |
| `POST` | `/api/auth/login` | Login → JWT token |
| `GET` | `/api/auth/me` | Get current user profile |
| `PUT` | `/api/auth/me` | Update vehicle profile |
| `POST` | `/api/analyze` | Analyze a parking sign image |
| `GET` | `/health` | Server health check |

**`/api/analyze` request** - `multipart/form-data`:
- `image` - JPEG/PNG photo of the sign
- `day` - day name in Swedish (e.g. `Måndag`)
- `time` - current time (`HH:MM`)

**Response:**
```json
{
  "signs": ["No parking Mon–Fri 7–19", "Residents zone A exempt"],
  "notes": ["Restriction active on weekdays during business hours"],
  "can_park": false,
  "message": "No parking - restriction active now"
}
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Android | Kotlin, CameraX, ViewBinding, Coroutines |
| iOS | Swift, SwiftUI, AVFoundation, Vision |
| Server | FastAPI, SQLAlchemy, SQLite |
| AI (local/macOS) | mlx-vlm, Qwen3-VL-8B-Instruct-4bit (Apple Silicon) |
| AI (local/Linux-Win) | transformers + torch, Qwen3-VL-8B-Instruct (auto-download) |
| AI (cloud) | HuggingFace Inference API, Qwen2.5-VL-72B |
| Auth | JWT (python-jose), bcrypt (passlib) |
| Package mgmt | uv |
