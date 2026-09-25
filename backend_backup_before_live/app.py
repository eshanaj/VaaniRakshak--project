from pathlib import Path

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
)

from fastapi.middleware.cors import CORSMiddleware

import os
import shutil
import asyncio
import uuid

from inference import (
    predict_audio,
    SUPPORTED_EXTENSIONS,
)

app = FastAPI(
    title="VaaniRakshak API",
    version="1.0"
)

# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# UPLOADS
# ============================================================

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def home():

    return {
        "message": "VaaniRakshak Backend is Running",
        "status": "healthy",
    }


# ============================================================
# UPLOAD ANALYSIS
# ============================================================

@app.post("/analyze-clip")
async def analyze_clip(
    file: UploadFile = File(...)
):

    original_name = file.filename or "audio"

    extension = (
        Path(original_name)
        .suffix
        .lower()
    )

    if extension not in SUPPORTED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio format. "
                "Supported formats: "
                + ", ".join(
                    sorted(
                        SUPPORTED_EXTENSIONS
                    )
                )
            ),
        )

    # Unique filename prevents collisions
    temp_name = (
        f"{uuid.uuid4().hex}"
        f"{extension}"
    )

    file_path = os.path.join(
        UPLOAD_FOLDER,
        temp_name
    )

    try:

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        result = predict_audio(
            file_path
        )

        # Show user's original filename
        result["filename"] = original_name

        return result

    except HTTPException:

        raise

    except Exception as e:

        print(
            f"[ERROR] Analysis failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        # Remove temporary upload
        if os.path.exists(file_path):

            try:
                os.remove(file_path)

            except Exception as e:

                print(
                    f"[WARNING] Could not "
                    f"delete temporary file: {e}"
                )


# ============================================================
# LIVE CALL DEMO
# ============================================================

@app.websocket("/ws/live")
async def live_websocket(
    websocket: WebSocket
):

    await websocket.accept()

    dummy_scores = [
        12,
        28,
        47,
        63,
        81,
        94
    ]

    try:

        for score in dummy_scores:

            if score < 35:

                decision = "Proceed"

            elif score < 70:

                decision = "Call Back"

            else:

                decision = "Escalate"

            await websocket.send_json(
                {
                    "confidence": score,
                    "decision": decision,
                }
            )

            await asyncio.sleep(2)

    except WebSocketDisconnect:

        print(
            "Live demo disconnected."
        )

    finally:

        try:
            await websocket.close()
        except Exception:
            pass
