import sys
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import torch

# ============================================================
# PROJECT PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
AASIST_DIR = PROJECT_DIR / "aasist"

# IMPORTANT:
# We continue using the original model checkpoint.
# Upload inference is NOT touched.
MODEL_PATH = BACKEND_DIR / "best.pth"

# ============================================================
# AASIST SETTINGS
# ============================================================

SAMPLE_RATE = 16000
AASIST_SAMPLES = 64600
WINDOW_SECONDS = AASIST_SAMPLES / SAMPLE_RATE

# ============================================================
# IMPORT AASIST
# ============================================================

sys.path.append(str(AASIST_DIR))

from models.AASIST import Model

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# MODEL CONFIG
# ============================================================

model_config = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [
        70,
        [1, 32],
        [32, 32],
        [32, 64],
        [64, 64],
    ],
    "gat_dims": [64, 32],
    "pool_ratios": [
        0.5,
        0.7,
        0.5,
        0.5,
    ],
    "temperatures": [
        2.0,
        2.0,
        100.0,
        100.0,
    ],
}

# ============================================================
# LOAD MODEL
# ============================================================

model = Model(model_config).to(device)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Live model checkpoint not found:\n{MODEL_PATH}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

if "model" in checkpoint:
    state_dict = checkpoint["model"]
else:
    state_dict = checkpoint

state_dict = {
    k.replace("module.", "", 1): v
    for k, v in state_dict.items()
}

model.load_state_dict(state_dict, strict=True)
model.eval()

print(f"[LIVE] AASIST loaded from:")
print(MODEL_PATH)
print(f"[LIVE] Device: {device}")
print(f"[LIVE] Window size: {WINDOW_SECONDS:.2f} seconds")
print("[LIVE] Preprocessing: DC removal + amplitude normalization")


# ============================================================
# LIVE AUDIO PREPROCESSING
# ============================================================

def preprocess_live_audio(audio):
    """
    Lightweight preprocessing for microphone/live audio.

    This is intentionally separate from the existing upload
    inference pipeline.

    Steps:
    1. Convert to float32
    2. Remove NaN/Inf
    3. Remove DC offset
    4. Normalize amplitude
    """

    audio = np.asarray(audio, dtype=np.float32)

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if len(audio) == 0:
        raise ValueError("Live recording contains no audio.")

    # Remove DC offset.
    audio = audio - np.mean(audio)

    # Amplitude normalization.
    peak = np.max(np.abs(audio))

    if peak > 1e-8:
        audio = audio / peak

    return audio.astype(np.float32)


# ============================================================
# PREPARE ONE AASIST WINDOW
# ============================================================

def prepare_live_window(audio):

    audio = preprocess_live_audio(audio)

    if len(audio) == 0:
        raise ValueError("Live recording contains no audio.")

    # If shorter than AASIST input length, repeat the audio.
    if len(audio) < AASIST_SAMPLES:

        repeats = (AASIST_SAMPLES // len(audio)) + 1

        audio = np.tile(
            audio,
            repeats
        )

        audio = audio[:AASIST_SAMPLES]

        return audio

    # If longer than one AASIST window,
    # take the center portion.
    if len(audio) > AASIST_SAMPLES:

        start = (
            len(audio) - AASIST_SAMPLES
        ) // 2

        audio = audio[
            start:start + AASIST_SAMPLES
        ]

    return audio.astype(np.float32)


# ============================================================
# RUN AASIST ON ONE WINDOW
# ============================================================

def run_model_on_window(audio):

    audio = prepare_live_window(audio)

    x = torch.tensor(
        audio,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        _, logits = model(x)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    cloned_probability = float(
        probabilities[0].item()
    )

    real_probability = float(
        probabilities[1].item()
    )

    return (
        cloned_probability,
        real_probability
    )


# ============================================================
# LIVE AUDIO PREDICTION
# ============================================================

def predict_live_audio(audio_path):

    """
    Separate inference function for microphone/live audio.

    IMPORTANT:
    This function does NOT call predict_audio()
    from inference.py.
    """

    audio_path = Path(audio_path)

    print(
        f"\n[LIVE] Analyzing: "
        f"{audio_path.name}"
    )

    temp_wav = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_file:

            temp_wav = Path(
                temp_file.name
            )

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
                str(temp_wav),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
        )

        audio, sr = librosa.load(
            str(temp_wav),
            sr=SAMPLE_RATE,
            mono=True
        )

    except Exception as e:

        raise RuntimeError(
            f"Could not decode live audio: {e}"
        )

    finally:

        if temp_wav is not None:

            temp_wav.unlink(
                missing_ok=True
            )

    if audio is None or len(audio) == 0:

        raise ValueError(
            "Live audio is empty."
        )

    duration = len(audio) / SAMPLE_RATE

    print(
        f"[LIVE] Audio duration: "
        f"{duration:.2f}s"
    )

    # ========================================================
    # CREATE 50% OVERLAPPING WINDOWS
    # ========================================================

    windows = []

    if len(audio) <= AASIST_SAMPLES:

        windows.append(audio)

    else:

        hop = AASIST_SAMPLES // 2

        start = 0

        while (
            start + AASIST_SAMPLES
            <= len(audio)
        ):

            windows.append(
                audio[
                    start:start + AASIST_SAMPLES
                ]
            )

            start += hop

        # Include final portion.
        if start < len(audio):

            windows.append(
                audio[-AASIST_SAMPLES:]
            )

    print(
        f"[LIVE] Number of analysis windows: "
        f"{len(windows)}"
    )

    # ========================================================
    # RUN MODEL
    # ========================================================

    cloned_scores = []
    real_scores = []

    for index, window in enumerate(windows):

        cloned_probability, real_probability = (
            run_model_on_window(window)
        )

        cloned_scores.append(
            cloned_probability
        )

        real_scores.append(
            real_probability
        )

        print(
            f"[LIVE] Window {index + 1}: "
            f"CLONED={cloned_probability:.3f}, "
            f"REAL={real_probability:.3f}"
        )

    # ========================================================
    # AVERAGE WINDOWS
    # ========================================================

    cloned_probability = float(
        np.mean(cloned_scores)
    )

    real_probability = float(
        np.mean(real_scores)
    )

    if real_probability >= cloned_probability:

        label = "real"
        confidence = real_probability

    else:

        label = "cloned"
        confidence = cloned_probability

    print(
        f"[LIVE] FINAL -> "
        f"{label.upper()} "
        f"({confidence:.3f})"
    )

    return {

        "prediction": label,

        "confidence": round(
            confidence,
            4
        ),

        "cloned_probability": round(
            cloned_probability,
            4
        ),

        "real_probability": round(
            real_probability,
            4
        ),

        "duration": round(
            duration,
            2
        ),

        "windows": len(windows),

        "live": True,
    }