from pathlib import Path
import subprocess
import sys

import pandas as pd


# =========================================================
# PROJECT PATH
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

PLAY_AND_RECORD = ROOT / "src" / "play_and_record.py"

ACQUISITION_PLAN = (
    ROOT
    / "manifests"
    / "acquisition_plan.csv"
)

REAL_MANIFEST = (
    ROOT
    / "manifests"
    / "manifest_real_replay.csv"
)


# =========================================================
# FINAL C2 CONFIGURATION
# =========================================================

TARGET_CONFIG = "C2"

EXPECTED_ROOM = "A"
EXPECTED_PLAYBACK = "speaker_B"
EXPECTED_MIC = "mic_B"

EXPECTED_DISTANCE = 1.0
EXPECTED_ANGLE = 45.0


# =========================================================
# DEVICE CONFIGURATION
# =========================================================
#
# GANTI setelah cek:
#
# python -m sounddevice
#
# Contoh sementara:
# =========================================================

INPUT_DEVICE = 5
OUTPUT_DEVICE = 7


# =========================================================
# LEVEL CONFIGURATION
# =========================================================
#
# Jangan otomatis menyalin level C1.
# Level C2 perlu dipilotkan karena perangkat berbeda.
#
# Setelah pilot aman, tetapkan level final di sini.
# =========================================================

MIC_LEVEL = 70
SPEAKER_LEVEL = 50

LEVEL_NOTE = "mic70_speaker50"

AGC_STATUS = "unknown"


# =========================================================
# BATCH CONFIGURATION
# =========================================================

BATCH_SIZE = 20

EXPECTED_C2_TOTAL = 120


# =========================================================
# RECORDING ID
# =========================================================

def build_recording_id(row):

    return (
        f"{row['source_file_id']}"
        f"__{row['config_id']}"
        f"__rep{int(row['repetition'])}"
    )


# =========================================================
# LOAD ACQUISITION PLAN
# =========================================================

def load_acquisition_plan():

    if not ACQUISITION_PLAN.exists():

        raise FileNotFoundError(
            f"acquisition_plan.csv tidak ditemukan:\n"
            f"{ACQUISITION_PLAN}"
        )

    plan = pd.read_csv(
        ACQUISITION_PLAN
    )

    required_columns = {
        "source_file_id",
        "label",
        "split",
        "config_id",
        "room",
        "playback",
        "mic",
        "distance",
        "angle",
        "repetition",
    }

    missing = (
        required_columns
        - set(plan.columns)
    )

    if missing:

        raise ValueError(
            "Acquisition plan kehilangan kolom:\n"
            f"{sorted(missing)}"
        )

    c2 = (
        plan[
            plan[
                "config_id"
            ].eq(
                TARGET_CONFIG
            )
        ]
        .copy()
    )

    if len(c2) != EXPECTED_C2_TOTAL:

        raise RuntimeError(
            f"Expected C2 = {EXPECTED_C2_TOTAL}, "
            f"actual = {len(c2)}"
        )

    return plan, c2


# =========================================================
# VALIDATE C2 PLAN
# =========================================================

def validate_c2_plan(
    c2: pd.DataFrame,
):

    print()
    print("=" * 70)
    print("VALIDATING C2 ACQUISITION PLAN")
    print("=" * 70)

    assert (
        c2["config_id"]
        == TARGET_CONFIG
    ).all()

    assert (
        c2["room"]
        .astype(str)
        == EXPECTED_ROOM
    ).all(), (
        "Room C2 tidak konsisten."
    )

    assert (
        c2["playback"]
        == EXPECTED_PLAYBACK
    ).all(), (
        "Playback C2 tidak konsisten."
    )

    assert (
        c2["mic"]
        == EXPECTED_MIC
    ).all(), (
        "Microphone C2 tidak konsisten."
    )

    assert (
        c2["distance"]
        .astype(float)
        .sub(
            EXPECTED_DISTANCE
        )
        .abs()
        .lt(1e-9)
        .all()
    ), (
        "Distance C2 tidak konsisten."
    )

    assert (
        c2["angle"]
        .astype(float)
        .sub(
            EXPECTED_ANGLE
        )
        .abs()
        .lt(1e-9)
        .all()
    ), (
        "Angle C2 tidak konsisten."
    )

    print()
    print("C2 acquisition plan: PASSED")

    print()
    print(
        f"Total C2 recording planned: "
        f"{len(c2)}"
    )

    print()
    print("Label distribution:")

    print(
        c2[
            "label"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print("Split distribution:")

    print(
        c2[
            "split"
        ]
        .value_counts()
        .sort_index()
    )


# =========================================================
# LOAD EXISTING MANIFEST
# =========================================================

def load_existing_manifest():

    if not REAL_MANIFEST.exists():
        return pd.DataFrame()

    try:

        return pd.read_csv(
            REAL_MANIFEST
        )

    except pd.errors.EmptyDataError:

        return pd.DataFrame()


# =========================================================
# VALIDATE C1 STILL PRESENT
# =========================================================

def validate_c1_complete(
    existing: pd.DataFrame,
):

    if existing.empty:

        raise RuntimeError(
            "Manifest kosong. "
            "C1 seharusnya sudah selesai."
        )

    c1 = (
        existing[
            existing[
                "config_id"
            ].eq(
                "C1"
            )
        ]
    )

    if len(c1) != 120:

        raise RuntimeError(
            f"C1 belum lengkap. "
            f"Current C1 = {len(c1)}/120."
        )

    print()
    print(
        "C1 prerequisite: 120/120"
    )


# =========================================================
# VALIDATE EXISTING C2
# =========================================================

def validate_existing_c2(
    existing: pd.DataFrame,
):

    if existing.empty:
        return

    c2 = (
        existing[
            existing[
                "config_id"
            ].eq(
                TARGET_CONFIG
            )
        ]
        .copy()
    )

    if c2.empty:

        print()
        print(
            "Existing C2 recording: 0"
        )

        return

    print()
    print("=" * 70)
    print("VALIDATING EXISTING C2")
    print("=" * 70)

    print()
    print(
        f"Existing C2 recording: "
        f"{len(c2)}"
    )

    if (
        c2[
            "recording_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate recording_id pada C2."
        )

    if not (
        c2[
            "level"
        ]
        .astype(str)
        .eq(
            LEVEL_NOTE
        )
        .all()
    ):

        print()
        print(
            c2[
                "level"
            ]
            .value_counts(
                dropna=False
            )
        )

        raise RuntimeError(
            f"C2 existing tidak semuanya "
            f"menggunakan {LEVEL_NOTE}."
        )

    if not (
        c2[
            "input_device_id"
        ]
        .astype(int)
        .eq(
            INPUT_DEVICE
        )
        .all()
    ):

        raise RuntimeError(
            "Input device C2 berubah."
        )

    if not (
        c2[
            "output_device_id"
        ]
        .astype(int)
        .eq(
            OUTPUT_DEVICE
        )
        .all()
    ):

        raise RuntimeError(
            "Output device C2 berubah."
        )

    print()
    print(
        "Existing C2 validation: PASSED"
    )


# =========================================================
# PREPARE C2
# =========================================================

def prepare_c2(
    c2: pd.DataFrame,
):

    c2 = c2.copy()

    c2[
        "recording_id"
    ] = c2.apply(
        build_recording_id,
        axis=1,
    )

    return c2


# =========================================================
# COMPLETED IDS
# =========================================================

def get_completed_ids(
    existing: pd.DataFrame,
):

    if existing.empty:
        return set()

    return set(
        existing[
            "recording_id"
        ]
        .dropna()
        .astype(str)
    )


# =========================================================
# SELECT NEXT BATCH
# =========================================================

def select_next_batch(
    c2: pd.DataFrame,
    completed: set,
):

    pending = (
        c2[
            ~c2[
                "recording_id"
            ]
            .isin(
                completed
            )
        ]
        .copy()
        .sort_index()
    )

    batch = (
        pending
        .head(
            BATCH_SIZE
        )
        .copy()
    )

    return pending, batch


# =========================================================
# SHOW PROGRESS
# =========================================================

def show_progress(
    c2: pd.DataFrame,
    completed: set,
):

    completed_c2 = (
        c2[
            "recording_id"
        ]
        .isin(
            completed
        )
        .sum()
    )

    remaining = (
        len(c2)
        - completed_c2
    )

    progress = (
        completed_c2
        / len(c2)
        * 100
    )

    print()
    print("=" * 70)
    print("C2 CURRENT PROGRESS")
    print("=" * 70)

    print()
    print(
        f"Completed : "
        f"{completed_c2}/"
        f"{len(c2)}"
    )

    print(
        f"Remaining : {remaining}"
    )

    print(
        f"Progress  : {progress:.2f}%"
    )


# =========================================================
# SHOW BATCH
# =========================================================

def show_batch(
    batch: pd.DataFrame,
):

    print()
    print("=" * 70)
    print("NEXT C2 BATCH")
    print("=" * 70)

    if batch.empty:

        print()
        print(
            "Tidak ada C2 yang tersisa."
        )

        return

    display = batch.copy()

    display.insert(
        0,
        "acquisition_row",
        display.index,
    )

    print()

    print(
        display[
            [
                "acquisition_row",
                "source_file_id",
                "label",
                "split",
                "repetition",
                "recording_id",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Batch size: {len(batch)}"
    )

    print()
    print("Label distribution:")

    print(
        batch[
            "label"
        ]
        .value_counts()
        .sort_index()
    )


# =========================================================
# RUN ONE RECORDING
# =========================================================

def run_recording(
    row_index: int,
):

    command = [
        sys.executable,
        str(
            PLAY_AND_RECORD
        ),

        "--row",
        str(
            row_index
        ),

        "--input-device",
        str(
            INPUT_DEVICE
        ),

        "--output-device",
        str(
            OUTPUT_DEVICE
        ),

        "--level",
        LEVEL_NOTE,

        "--agc",
        AGC_STATUS,
    ]

    result = subprocess.run(
        command,
        cwd=str(ROOT),
    )

    return result.returncode


# =========================================================
# POST RUN STATUS
# =========================================================

def post_run_status(
    c2: pd.DataFrame,
):

    existing = (
        load_existing_manifest()
    )

    completed = (
        get_completed_ids(
            existing
        )
    )

    completed_c2 = (
        c2[
            "recording_id"
        ]
        .isin(
            completed
        )
        .sum()
    )

    print()
    print("=" * 70)
    print("C2 POST-RUN STATUS")
    print("=" * 70)

    print()
    print(
        f"Completed C2 : "
        f"{completed_c2}/120"
    )

    print(
        f"Remaining C2 : "
        f"{120 - completed_c2}"
    )

    if not existing.empty:

        c2_manifest = (
            existing[
                existing[
                    "config_id"
                ].eq(
                    TARGET_CONFIG
                )
            ]
        )

        if not c2_manifest.empty:

            print()
            print(
                "C2 level distribution:"
            )

            print(
                c2_manifest[
                    "level"
                ]
                .value_counts(
                    dropna=False
                )
            )

    return completed_c2


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("CONTROLLED REAL REPLAY - C2 FINAL")
    print("=" * 70)

    print()
    print("FINAL C2 CONFIGURATION")

    print()
    print(
        f"Room          : {EXPECTED_ROOM}"
    )

    print(
        f"Playback      : {EXPECTED_PLAYBACK}"
    )

    print(
        f"Microphone    : {EXPECTED_MIC}"
    )

    print(
        f"Distance      : {EXPECTED_DISTANCE} m"
    )

    print(
        f"Angle         : {EXPECTED_ANGLE} deg"
    )

    print(
        f"Mic level     : {MIC_LEVEL}"
    )

    print(
        f"Speaker level : {SPEAKER_LEVEL}"
    )

    print(
        f"Input device  : {INPUT_DEVICE}"
    )

    print(
        f"Output device : {OUTPUT_DEVICE}"
    )

    print(
        f"Level note    : {LEVEL_NOTE}"
    )

    # =====================================================
    # FILE CHECK
    # =====================================================

    if not PLAY_AND_RECORD.exists():

        raise FileNotFoundError(
            PLAY_AND_RECORD
        )

    # =====================================================
    # LOAD PLAN
    # =====================================================

    _, c2 = (
        load_acquisition_plan()
    )

    c2 = (
        prepare_c2(
            c2
        )
    )

    validate_c2_plan(
        c2
    )

    # =====================================================
    # EXISTING
    # =====================================================

    existing = (
        load_existing_manifest()
    )

    validate_c1_complete(
        existing
    )

    validate_existing_c2(
        existing
    )

    completed = (
        get_completed_ids(
            existing
        )
    )

    # =====================================================
    # PROGRESS
    # =====================================================

    show_progress(
        c2,
        completed,
    )

    # =====================================================
    # NEXT BATCH
    # =====================================================

    _, batch = (
        select_next_batch(
            c2,
            completed,
        )
    )

    # =====================================================
    # C2 COMPLETE
    # =====================================================

    if batch.empty:

        print()
        print("=" * 70)
        print("C2 RECORDING COMPLETE")
        print("=" * 70)

        print()
        print(
            "C2 = 120/120"
        )

        print(
            "Lanjutkan ke final QC."
        )

        return

    # =====================================================
    # SHOW BATCH
    # =====================================================

    show_batch(
        batch
    )

    # =====================================================
    # SETUP CONFIRMATION
    # =====================================================

    print()
    print("=" * 70)
    print("PHYSICAL SETUP - C2")
    print("=" * 70)

    print()
    print(
        "Pastikan hardware benar-benar "
        "speaker_B dan mic_B."
    )

    print()
    print(
        f"Distance : "
        f"{EXPECTED_DISTANCE} m"
    )

    print(
        f"Angle    : "
        f"{EXPECTED_ANGLE}°"
    )

    print()
    print(
        f"Mic level     : "
        f"{MIC_LEVEL}"
    )

    print(
        f"Speaker level : "
        f"{SPEAKER_LEVEL}"
    )

    input(
        "\nJika physical setup C2 sudah benar, "
        "tekan ENTER untuk mulai..."
    )

    successful = []
    failed = []

    # =====================================================
    # RECORD LOOP
    # =====================================================

    for number, (
        row_index,
        row,
    ) in enumerate(
        batch.iterrows(),
        start=1,
    ):

        rid = (
            row[
                "recording_id"
            ]
        )

        print()
        print("=" * 70)

        print(
            f"C2 RECORDING "
            f"{number}/"
            f"{len(batch)}"
        )

        print("=" * 70)

        print()
        print(
            f"Acquisition row : "
            f"{row_index}"
        )

        print(
            f"Recording ID    : "
            f"{rid}"
        )

        print(
            f"Source          : "
            f"{row['source_file_id']}"
        )

        print(
            f"Label           : "
            f"{int(row['label'])}"
        )

        print(
            f"Split           : "
            f"{row['split']}"
        )

        print(
            f"Repetition      : "
            f"{int(row['repetition'])}"
        )

        return_code = (
            run_recording(
                row_index
            )
        )

        if return_code == 0:

            successful.append(
                rid
            )

            print()
            print(
                f"[PASS] {rid}"
            )

        else:

            failed.append(
                {
                    "row":
                        row_index,

                    "recording_id":
                        rid,
                }
            )

            print()
            print("=" * 70)
            print("C2 BATCH STOPPED")
            print("=" * 70)

            print()
            print(
                f"Recording gagal: "
                f"{rid}"
            )

            print()
            print(
                "Periksa error/QC sebelum "
                "melanjutkan."
            )

            break

    # =====================================================
    # SUMMARY
    # =====================================================

    print()
    print("=" * 70)
    print("C2 BATCH SUMMARY")
    print("=" * 70)

    print()
    print(
        f"Planned    : {len(batch)}"
    )

    print(
        f"Successful : {len(successful)}"
    )

    print(
        f"Failed     : {len(failed)}"
    )

    completed_c2 = (
        post_run_status(
            c2
        )
    )

    if failed:

        print()
        print(
            "Setelah masalah diperbaiki, "
            "jalankan file yang sama lagi."
        )

    elif completed_c2 < 120:

        print()
        print(
            "STOP recording."
        )

        print(
            "Jalankan test_real_replay.py "
            "untuk QC checkpoint."
        )

    else:

        print()
        print(
            "C2 FINAL ACQUISITION COMPLETE."
        )

        print(
            "Jalankan final QC C2."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()