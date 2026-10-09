from pathlib import Path

import pandas as pd
import numpy as np


# =========================================================
# PATH
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

SOURCE_MANIFEST = ROOT / "manifests" / "source_manifest.csv"
OUTPUT_PLAN = ROOT / "manifests" / "acquisition_plan.csv"


# =========================================================
# LOAD SOURCE MANIFEST
# =========================================================

src = pd.read_csv(SOURCE_MANIFEST)


# =========================================================
# BASIC VALIDATION
# =========================================================

required_columns = {
    "source_file_id",
    "file_path",
    "speaker_id",
    "attack_id",
    "label",
    "dataset",
    "split",
    "duration_s",
    "sha256",
}

missing_columns = required_columns - set(src.columns)

if missing_columns:
    raise ValueError(
        f"source_manifest.csv tidak memiliki kolom: "
        f"{sorted(missing_columns)}"
    )


# =========================================================
# CHECK AVAILABLE LABEL COUNTS
# =========================================================

label_counts_before = (
    src["label"]
    .value_counts()
    .sort_index()
)

print("=" * 70)
print("SOURCE MANIFEST BEFORE SAMPLING")
print("=" * 70)

print()
print("Available source per label:")
print(label_counts_before)

print()

for label in (0, 1):
    available = int(label_counts_before.get(label, 0))

    if available < 60:
        raise ValueError(
            f"Jumlah source untuk label {label} tidak cukup. "
            f"Dibutuhkan 60, tersedia {available}."
        )


# =========================================================
# SAMPLE 60 BONA FIDE + 60 SPOOF
# =========================================================
#
# Ini mempertahankan struktur kode dari panduan:
#
# src = src.groupby("label", group_keys=False).sample(
#     n=60,
#     random_state=2026
# )
#
# random_state=2026 memastikan sampling reproducible.
# =========================================================

src = (
    src.groupby(
        "label",
        group_keys=False
    )
    .sample(
        n=60,
        random_state=2026
    )
    .reset_index(drop=True)
)


# =========================================================
# ACQUISITION CONFIGURATION
# =========================================================
#
# Konfigurasi mengikuti panduan.
# Tidak mengubah desain eksperimen.
# =========================================================

configs = [
    {
        "config_id": "C1",
        "room": "A",
        "playback": "speaker_A",
        "mic": "mic_A",
        "distance": 0.5,
        "angle": 0,
    },
    {
        "config_id": "C2",
        "room": "A",
        "playback": "speaker_B",
        "mic": "mic_B",
        "distance": 1.0,
        "angle": 45,
    },
    {
        "config_id": "C3",
        "room": "B",
        "playback": "speaker_A",
        "mic": "mic_unseen",
        "distance": 0.5,
        "angle": 45,
    },
    {
        "config_id": "C4",
        "room": "B",
        "playback": "speaker_B",
        "mic": "mic_unseen",
        "distance": 1.0,
        "angle": 0,
    },
]


# =========================================================
# BUILD ACQUISITION PLAN
# =========================================================
#
# Setiap source mendapat:
# - 2 konfigurasi
# - 2 repetisi per konfigurasi
#
# Jadi:
# 120 source × 2 configs × 2 repetitions = 480 recordings
# =========================================================

rows = []

for i, row in src.iterrows():

    chosen = [
        configs[i % len(configs)],
        configs[(i + 1) % len(configs)]
    ]

    for config in chosen:

        for rep in (1, 2):

            rows.append(
                {
                    **row.to_dict(),
                    **config,
                    "repetition": rep
                }
            )


# =========================================================
# CREATE DATAFRAME
# =========================================================

plan = pd.DataFrame(rows)


# =========================================================
# SAVE ACQUISITION PLAN
# =========================================================

OUTPUT_PLAN.parent.mkdir(
    parents=True,
    exist_ok=True
)

plan.to_csv(
    OUTPUT_PLAN,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print()
print("=" * 70)
print("ACQUISITION PLAN SUMMARY")
print("=" * 70)


# =========================================================
# SELECTED SOURCE BY LABEL
# =========================================================

print()
print("Selected source:")
print(
    src["label"]
    .value_counts()
    .sort_index()
)


# =========================================================
# UNIQUE SOURCE
# =========================================================

print()
print("Total unique source:")
print(
    plan["source_file_id"]
    .nunique()
)


# =========================================================
# TOTAL RECORDINGS
# =========================================================

print()
print("Total planned recordings:")
print(
    len(plan)
)


# =========================================================
# RECORDINGS PER SOURCE
# =========================================================

recordings_per_source = (
    plan.groupby("source_file_id")
    .size()
)

print()
print("Recordings per source:")
print(
    recordings_per_source
    .value_counts()
    .sort_index()
)


# =========================================================
# CONFIGURATION DISTRIBUTION
# =========================================================

print()
print("Configurations:")
print(
    plan["config_id"]
    .value_counts()
    .sort_index()
)


# =========================================================
# LABEL x CONFIGURATION
# =========================================================

print()
print("Label x configuration:")
print(
    pd.crosstab(
        plan["label"],
        plan["config_id"]
    )
)


# =========================================================
# ROOM DISTRIBUTION
# =========================================================

print()
print("Room:")
print(
    plan["room"]
    .value_counts()
    .sort_index()
)


# =========================================================
# MICROPHONE DISTRIBUTION
# =========================================================

print()
print("Microphone:")
print(
    plan["mic"]
    .value_counts()
    .sort_index()
)


# =========================================================
# PLAYBACK DEVICE DISTRIBUTION
# =========================================================

print()
print("Playback device:")
print(
    plan["playback"]
    .value_counts()
    .sort_index()
)


# =========================================================
# SPLIT AUDIT - SELECTED SOURCE
# =========================================================
#
# Bagian ini ditambahkan untuk audit source split.
# Tidak mengubah sampling atau konfigurasi.
# =========================================================

print()
print("Selected source per split:")
print(
    src["split"]
    .value_counts()
    .sort_index()
)


# =========================================================
# LABEL x SPLIT
# =========================================================

print()
print("Label x split:")
print(
    pd.crosstab(
        src["split"],
        src["label"]
    )
)


# =========================================================
# PLANNED RECORDINGS PER SPLIT
# =========================================================

print()
print("Planned recordings per split:")
print(
    plan["split"]
    .value_counts()
    .sort_index()
)


# =========================================================
# SPLIT x CONFIGURATION
# =========================================================

print()
print("Split x configuration:")
print(
    pd.crosstab(
        plan["split"],
        plan["config_id"]
    )
)


# =========================================================
# LABEL x ROOM
# =========================================================

print()
print("Label x room:")
print(
    pd.crosstab(
        plan["label"],
        plan["room"]
    )
)


# =========================================================
# LABEL x MICROPHONE
# =========================================================

print()
print("Label x microphone:")
print(
    pd.crosstab(
        plan["label"],
        plan["mic"]
    )
)


# =========================================================
# LABEL x PLAYBACK DEVICE
# =========================================================

print()
print("Label x playback:")
print(
    pd.crosstab(
        plan["label"],
        plan["playback"]
    )
)


# =========================================================
# HARD VALIDATION
# =========================================================

print()
print("=" * 70)
print("VALIDATION")
print("=" * 70)


# 120 unique sources
assert plan["source_file_id"].nunique() == 120, (
    "Jumlah unique source harus 120."
)


# 480 recording plans
assert len(plan) == 480, (
    "Jumlah planned recordings harus 480."
)


# 60 source per label
selected_label_counts = (
    src["label"]
    .value_counts()
)

assert selected_label_counts.get(0, 0) == 60, (
    "Jumlah bona fide harus 60."
)

assert selected_label_counts.get(1, 0) == 60, (
    "Jumlah spoof harus 60."
)


# Each source exactly 4 recordings
assert (recordings_per_source == 4).all(), (
    "Setiap source harus memiliki tepat 4 planned recordings."
)


# Each configuration exactly 120 rows
config_counts = (
    plan["config_id"]
    .value_counts()
)

for config_id in ["C1", "C2", "C3", "C4"]:

    assert config_counts.get(config_id, 0) == 120, (
        f"{config_id} harus memiliki tepat 120 recordings."
    )


# Each label x configuration = 60
label_config = pd.crosstab(
    plan["label"],
    plan["config_id"]
)

for label in (0, 1):

    for config_id in ["C1", "C2", "C3", "C4"]:

        assert label_config.loc[label, config_id] == 60, (
            f"Label {label} pada {config_id} harus 60."
        )


# Room balance
room_counts = (
    plan["room"]
    .value_counts()
)

assert room_counts.get("A", 0) == 240, (
    "Room A harus memiliki 240 recordings."
)

assert room_counts.get("B", 0) == 240, (
    "Room B harus memiliki 240 recordings."
)


# Playback balance
playback_counts = (
    plan["playback"]
    .value_counts()
)

assert playback_counts.get("speaker_A", 0) == 240, (
    "speaker_A harus memiliki 240 recordings."
)

assert playback_counts.get("speaker_B", 0) == 240, (
    "speaker_B harus memiliki 240 recordings."
)


# =========================================================
# FINAL
# =========================================================

print()
print("VALIDATION PASSED")

print()
print("Acquisition plan saved to:")
print(OUTPUT_PLAN)

print("=" * 70)