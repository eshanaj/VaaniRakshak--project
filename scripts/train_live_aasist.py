from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
AASIST_DIR = PROJECT_DIR / "aasist"

LIVE_CLONED = PROJECT_DIR / "dataset" / "live_training" / "cloned"
LIVE_REAL = PROJECT_DIR / "dataset" / "live_training" / "real"

MODEL_PATH = PROJECT_DIR / "backend" / "best.pth"
LIVE_MODEL_PATH = PROJECT_DIR / "backend" / "live_best.pth"

# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000
AASIST_SAMPLES = 64600

BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 1e-5

VAL_RATIO = 0.2

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# IMPORT AASIST
# ============================================================

import sys

sys.path.append(str(AASIST_DIR))

from models.AASIST import Model


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("=" * 60)
print("VAANIRAKSHAK LIVE AASIST TRAINING")
print("=" * 60)
print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA is not available.")
    print("Training will be slow on CPU.")


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
# LOAD AUDIO
# ============================================================

def load_audio(path):

    import librosa

    audio, sr = librosa.load(
        str(path),
        sr=SAMPLE_RATE,
        mono=True,
    )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return audio


# ============================================================
# PAD / CROP
# ============================================================

def prepare_audio(audio, training=False):

    length = len(audio)

    if length == 0:
        return np.zeros(
            AASIST_SAMPLES,
            dtype=np.float32,
        )

    if length >= AASIST_SAMPLES:

        if training:

            start = random.randint(
                0,
                length - AASIST_SAMPLES,
            )

        else:

            start = (
                length - AASIST_SAMPLES
            ) // 2

        audio = audio[
            start:start + AASIST_SAMPLES
        ]

    else:

        repeats = (
            AASIST_SAMPLES // length
        ) + 1

        audio = np.tile(
            audio,
            repeats,
        )[:AASIST_SAMPLES]

    return audio.astype(np.float32)


# ============================================================
# DATASET
# ============================================================

class LiveDataset(Dataset):

    def __init__(
        self,
        samples,
        training=False,
    ):

        self.samples = samples
        self.training = training

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        audio = load_audio(path)

        audio = prepare_audio(
            audio,
            training=self.training,
        )

        tensor = torch.tensor(
            audio,
            dtype=torch.float32,
        )

        return tensor, label


# ============================================================
# COLLECT FILES
# ============================================================

cloned_files = sorted(
    LIVE_CLONED.glob("*.wav")
)

real_files = sorted(
    LIVE_REAL.glob("*.wav")
)

print()
print(f"Cloned files: {len(cloned_files)}")
print(f"Real files:   {len(real_files)}")


# ============================================================
# SOURCE-AWARE SPLIT
# ============================================================

# Each original source produced three files:
#
# original_live_1.wav
# original_live_2.wav
# original_live_3.wav
#
# We keep all variations of the same source together.

def get_source_name(path):

    name = path.stem

    if "_live_" in name:

        return name.rsplit(
            "_live_",
            1,
        )[0]

    return name


def split_by_source(files):

    groups = {}

    for path in files:

        source = get_source_name(path)

        groups.setdefault(
            source,
            [],
        ).append(path)

    sources = list(groups.keys())

    random.shuffle(sources)

    split_index = int(
        len(sources) * (1 - VAL_RATIO)
    )

    train_sources = set(
        sources[:split_index]
    )

    train_files = []
    val_files = []

    for source, paths in groups.items():

        if source in train_sources:

            train_files.extend(paths)

        else:

            val_files.extend(paths)

    return train_files, val_files


cloned_train, cloned_val = split_by_source(
    cloned_files
)

real_train, real_val = split_by_source(
    real_files
)


train_samples = (
    [(p, 0) for p in cloned_train]
    + [(p, 1) for p in real_train]
)

val_samples = (
    [(p, 0) for p in cloned_val]
    + [(p, 1) for p in real_val]
)


random.shuffle(train_samples)
random.shuffle(val_samples)


print()
print(f"Training samples:   {len(train_samples)}")
print(f"Validation samples: {len(val_samples)}")


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = LiveDataset(
    train_samples,
    training=True,
)

val_dataset = LiveDataset(
    val_samples,
    training=False,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)


# ============================================================
# CREATE MODEL
# ============================================================

model = Model(
    model_config
).to(device)


# ============================================================
# LOAD EXISTING AASIST
# ============================================================

print()
print("Loading existing AASIST checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
)

if "model" in checkpoint:

    state_dict = checkpoint["model"]

else:

    state_dict = checkpoint


state_dict = {
    key.replace(
        "module.",
        "",
        1,
    ): value
    for key, value in state_dict.items()
}


model.load_state_dict(
    state_dict,
    strict=True,
)

print("Existing AASIST loaded.")

# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(
        [1.0, 1.0],
        dtype=torch.float32,
        device=device,
    )
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0

for epoch in range(1, EPOCHS + 1):

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for audio, labels in train_loader:

        audio = audio.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad()

        _, logits = model(audio)

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )

        optimizer.step()

        train_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)


    train_loss /= train_total

    train_accuracy = (
        train_correct /
        train_total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    cloned_correct = 0
    cloned_total = 0

    real_correct = 0
    real_total = 0

    with torch.no_grad():

        for audio, labels in val_loader:

            audio = audio.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            _, logits = model(audio)

            loss = criterion(
                logits,
                labels,
            )

            val_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = logits.argmax(
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

            cloned_mask = labels == 0

            real_mask = labels == 1

            cloned_total += (
                cloned_mask.sum().item()
            )

            real_total += (
                real_mask.sum().item()
            )

            cloned_correct += (
                (
                    predictions[
                        cloned_mask
                    ] == 0
                ).sum().item()
            )

            real_correct += (
                (
                    predictions[
                        real_mask
                    ] == 1
                ).sum().item()
            )


    val_loss /= val_total

    val_accuracy = (
        val_correct /
        val_total
    )

    cloned_accuracy = (
        cloned_correct /
        cloned_total
        if cloned_total
        else 0
    )

    real_accuracy = (
        real_correct /
        real_total
        if real_total
        else 0
    )


    print()
    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"  CLONED accuracy: "
        f"{cloned_accuracy * 100:.2f}%"
    )

    print(
        f"  REAL accuracy: "
        f"{real_accuracy * 100:.2f}%"
    )


    # ========================================================
    # SAVE BEST LIVE MODEL
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            LIVE_MODEL_PATH,
        )

        print(
            f"Saved live model -> "
            f"{LIVE_MODEL_PATH}"
        )


print()
print("=" * 60)
print("LIVE TRAINING COMPLETE")
print("=" * 60)
print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)
print(
    f"Live checkpoint: "
    f"{LIVE_MODEL_PATH}"
)