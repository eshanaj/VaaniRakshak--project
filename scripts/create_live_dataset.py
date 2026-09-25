from pathlib import Path
import random

import librosa
import numpy as np
import soundfile as sf


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

CLONED_DIR = PROJECT_DIR / "dataset" / "cloned"
REAL_DIR = PROJECT_DIR / "dataset" / "real"

OUTPUT_CLONED_DIR = (
    PROJECT_DIR / "dataset" / "live_training" / "cloned"
)
OUTPUT_REAL_DIR = (
    PROJECT_DIR / "dataset" / "live_training" / "real"
)

OUTPUT_CLONED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REAL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# AUDIO SETTINGS
# ============================================================

TARGET_SR = 16000

# Number of acoustic versions generated from each source file.
VARIATIONS_PER_FILE = 3


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(42)
np.random.seed(42)


# ============================================================
# AUDIO AUGMENTATION FUNCTIONS
# ============================================================

def normalize(audio):
    audio = np.asarray(audio, dtype=np.float32)

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    peak = np.max(np.abs(audio))

    if peak > 1e-8:
        audio = audio / peak

    return audio.astype(np.float32)


def add_room_reverb(audio, sr):
    """
    Simple synthetic room/reverberation effect.

    This is NOT a physical simulation of a real room.
    It is only an acoustic augmentation for training.
    """

    delay_ms = random.uniform(20, 80)
    delay_samples = int(sr * delay_ms / 1000)

    if delay_samples >= len(audio):
        return audio

    decay = random.uniform(0.15, 0.40)

    output = audio.copy()

    output[delay_samples:] += (
        audio[:-delay_samples] * decay
    )

    second_delay = delay_samples * 2

    if second_delay < len(audio):
        output[second_delay:] += (
            audio[:-second_delay] * decay * 0.5
        )

    return normalize(output)


def add_background_noise(audio):
    """
    Adds low-level microphone/environment noise.
    """

    noise_level = random.uniform(0.003, 0.015)

    noise = np.random.normal(
        0,
        noise_level,
        size=len(audio),
    ).astype(np.float32)

    return normalize(audio + noise)


def apply_volume_change(audio):
    gain = random.uniform(0.55, 1.0)
    return normalize(audio * gain)


def simulate_microphone(audio, sr):
    """
    Approximate some characteristics of a consumer
    microphone/speaker recording chain.

    This is intentionally conservative.
    """

    # Slight bandwidth limitation.
    audio = librosa.effects.preemphasis(
        audio,
        coef=random.uniform(0.85, 0.97),
    )

    # Room/reflection effect.
    audio = add_room_reverb(audio, sr)

    # Background/environment noise.
    audio = add_background_noise(audio)

    # Microphone distance / volume variation.
    audio = apply_volume_change(audio)

    return normalize(audio)


# ============================================================
# LOAD + PROCESS
# ============================================================

def process_file(input_path, output_dir, label):
    try:
        audio, sr = librosa.load(
            str(input_path),
            sr=TARGET_SR,
            mono=True,
        )

        if audio is None or len(audio) == 0:
            print(f"[SKIP] Empty: {input_path.name}")
            return 0

        generated = 0

        for variation in range(1, VARIATIONS_PER_FILE + 1):

            augmented = simulate_microphone(
                audio.copy(),
                TARGET_SR,
            )

            output_name = (
                f"{input_path.stem}"
                f"_live_{variation}.wav"
            )

            output_path = output_dir / output_name

            sf.write(
                str(output_path),
                augmented,
                TARGET_SR,
                subtype="PCM_16",
            )

            generated += 1

        return generated

    except Exception as e:
        print(
            f"[ERROR] {label}: "
            f"{input_path.name} -> {e}"
        )
        return 0


# ============================================================
# MAIN
# ============================================================

def main():

    cloned_files = list(
        CLONED_DIR.rglob("*.wav")
    )

    real_files = list(
        REAL_DIR.rglob("*.mp3")
    )

    print()
    print("=" * 60)
    print("VaaniRakshak LIVE DATASET GENERATOR")
    print("=" * 60)

    print(f"Cloned source files : {len(cloned_files)}")
    print(f"Real source files   : {len(real_files)}")
    print(
        f"Variations per file: "
        f"{VARIATIONS_PER_FILE}"
    )

    print()
    print("Output:")
    print(OUTPUT_CLONED_DIR)
    print(OUTPUT_REAL_DIR)
    print()

    cloned_count = 0

    for index, file in enumerate(
        cloned_files,
        start=1,
    ):

        print(
            f"[CLONED {index}/{len(cloned_files)}] "
            f"{file.name}"
        )

        cloned_count += process_file(
            file,
            OUTPUT_CLONED_DIR,
            "CLONED",
        )

    real_count = 0

    for index, file in enumerate(
        real_files,
        start=1,
    ):

        print(
            f"[REAL {index}/{len(real_files)}] "
            f"{file.name}"
        )

        real_count += process_file(
            file,
            OUTPUT_REAL_DIR,
            "REAL",
        )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"Generated cloned files: {cloned_count}"
    )

    print(
        f"Generated real files:   {real_count}"
    )

    print()
    print(
        "Original dataset was not modified."
    )


if __name__ == "__main__":
    main()