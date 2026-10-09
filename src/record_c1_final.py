from pathlib import Path
import subprocess
import sys

import pandas as pd


# =========================================================
# PROJECT PATH
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

PLAY_AND_RECORD = (
    ROOT
    / "src"
    / "play_and_record.py"
)

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
# FINAL C1 CONFIGURATION
# =========================================================

TARGET_CONFIG = "C1"

EXPECTED_ROOM = "A"
EXPECTED_PLAYBACK = "speaker_A"
EXPECTED_MIC = "mic_A"

EXPECTED_DISTANCE = 0.5
EXPECTED_ANGLE = 0.0


# =========================================================
# FINAL AUDIO DEVICE / LEVEL CONFIGURATION
# =========================================================

INPUT_DEVICE = 1
OUTPUT_DEVICE = 3

MIC_LEVEL = 70
SPEAKER_LEVEL = 50

LEVEL_NOTE = "mic70_speaker50"

AGC_STATUS = "unknown"


# =========================================================
# BATCH CONFIGURATION
# =========================================================

BATCH_SIZE = 20

# Berdasarkan acquisition plan:
# C1 total = 120 recording
EXPECTED_C1_TOTAL = 120


# =========================================================
# BUILD RECORDING ID
# =========================================================

def build_recording_id(row):
    """
    Recording ID harus sama dengan format
    pada play_and_record.py.
    """

    return (
        f"{row['source_file_id']}"
        f"__{row['config_id']}"
        f"__rep{int(row['repetition'])}"
    )


# =========================================================
# LOAD ACQUISITION PLAN
# =========================================================

def load_acquisition_plan():
    """
    Membaca acquisition_plan.csv dan mengambil C1.
    """

    if not ACQUISITION_PLAN.exists():

        raise FileNotFoundError(
            "acquisition_plan.csv tidak ditemukan:\n"
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
            "acquisition_plan.csv kehilangan kolom:\n"
            f"{sorted(missing)}"
        )

    c1 = (
        plan[
            plan[
                "config_id"
            ].eq(
                TARGET_CONFIG
            )
        ]
        .copy()
    )

    if len(c1) != EXPECTED_C1_TOTAL:

        raise RuntimeError(
            f"Expected C1 = {EXPECTED_C1_TOTAL} recording, "
            f"tetapi acquisition plan memiliki {len(c1)}."
        )

    return plan, c1


# =========================================================
# VALIDATE C1 PLAN
# =========================================================

def validate_c1_plan(
    c1: pd.DataFrame,
):
    """
    Memastikan semua row C1 memiliki
    physical configuration yang benar.
    """

    print()
    print("=" * 70)
    print("VALIDATING C1 ACQUISITION PLAN")
    print("=" * 70)

    assert (
        c1[
            "config_id"
        ]
        == TARGET_CONFIG
    ).all(), (
        "Ada row C1 dengan config_id yang salah."
    )

    assert (
        c1[
            "room"
        ]
        .astype(str)
        == EXPECTED_ROOM
    ).all(), (
        "Room C1 tidak konsisten."
    )

    assert (
        c1[
            "playback"
        ]
        == EXPECTED_PLAYBACK
    ).all(), (
        "Playback C1 tidak konsisten."
    )

    assert (
        c1[
            "mic"
        ]
        == EXPECTED_MIC
    ).all(), (
        "Microphone C1 tidak konsisten."
    )

    assert (
        c1[
            "distance"
        ]
        .astype(float)
        .sub(
            EXPECTED_DISTANCE
        )
        .abs()
        .lt(1e-9)
        .all()
    ), (
        "Distance C1 tidak konsisten."
    )

    assert (
        c1[
            "angle"
        ]
        .astype(float)
        .sub(
            EXPECTED_ANGLE
        )
        .abs()
        .lt(1e-9)
        .all()
    ), (
        "Angle C1 tidak konsisten."
    )

    print()
    print("C1 acquisition plan: PASSED")

    print()
    print(
        f"Total C1 recording planned : "
        f"{len(c1)}"
    )

    print()
    print("Label distribution:")

    print(
        c1[
            "label"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print("Split distribution:")

    print(
        c1[
            "split"
        ]
        .value_counts()
        .sort_index()
    )


# =========================================================
# LOAD EXISTING MANIFEST
# =========================================================

def load_existing_manifest():
    """
    Membaca manifest real replay.

    Jika file belum ada atau kosong,
    kembalikan DataFrame kosong.
    """

    if not REAL_MANIFEST.exists():

        return pd.DataFrame()

    try:

        df = pd.read_csv(
            REAL_MANIFEST
        )

    except pd.errors.EmptyDataError:

        return pd.DataFrame()

    return df


# =========================================================
# VALIDATE EXISTING C1
# =========================================================

def validate_existing_c1(
    existing: pd.DataFrame,
):
    """
    Jangan izinkan pencampuran:

        mic70_speaker80
    dengan
        mic70_speaker50

    dalam C1 final.
    """

    if existing.empty:

        print()
        print(
            "Existing real replay manifest: EMPTY"
        )

        return

    if (
        "config_id"
        not in existing.columns
    ):

        raise RuntimeError(
            "Manifest existing tidak memiliki config_id."
        )

    c1_existing = (
        existing[
            existing[
                "config_id"
            ].eq(
                TARGET_CONFIG
            )
        ]
        .copy()
    )

    if c1_existing.empty:

        return

    print()
    print("=" * 70)
    print("VALIDATING EXISTING C1 DATA")
    print("=" * 70)

    print()
    print(
        f"Existing C1 recording : "
        f"{len(c1_existing)}"
    )

    if (
        "level"
        not in c1_existing.columns
    ):

        raise RuntimeError(
            "Manifest C1 existing tidak memiliki "
            "kolom level."
        )

    levels = (
        c1_existing[
            "level"
        ]
        .value_counts(
            dropna=False
        )
    )

    print()
    print("Existing level distribution:")

    print(
        levels
    )

    wrong_level = (
        c1_existing[
            "level"
        ]
        != LEVEL_NOTE
    )

    if wrong_level.any():

        bad = (
            c1_existing[
                wrong_level
            ]
        )

        print()
        print(
            "ERROR:"
        )

        print(
            "Masih ada C1 yang bukan "
            f"{LEVEL_NOTE}."
        )

        print()
        print(
            bad[
                [
                    "recording_id",
                    "acquisition_row",
                    "level",
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Jangan campurkan recording C1 "
            "dengan level lama. "
            "Bersihkan mic70_speaker80 terlebih dahulu."
        )

    if (
        c1_existing[
            "recording_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate recording_id ditemukan "
            "pada C1 existing."
        )

    print()
    print(
        "Existing C1 validation: PASSED"
    )


# =========================================================
# GET COMPLETED RECORDING IDS
# =========================================================

def get_completed_ids(
    existing: pd.DataFrame,
):
    """
    Recording yang sudah selesai akan dilewati.
    """

    if existing.empty:

        return set()

    if (
        "recording_id"
        not in existing.columns
    ):

        return set()

    return set(
        existing[
            "recording_id"
        ]
        .dropna()
        .astype(str)
    )


# =========================================================
# PREPARE C1 TABLE
# =========================================================

def prepare_c1(
    c1: pd.DataFrame,
):
    """
    Tambahkan recording_id pada setiap row C1.
    """

    c1 = (
        c1
        .copy()
    )

    c1[
        "recording_id"
    ] = c1.apply(
        build_recording_id,
        axis=1,
    )

    return c1


# =========================================================
# SELECT NEXT BATCH
# =========================================================

def select_next_batch(
    c1: pd.DataFrame,
    completed: set,
):
    """
    Mengambil recording C1 yang belum selesai.

    Maksimum BATCH_SIZE per run.
    """

    pending = (
        c1[
            ~c1[
                "recording_id"
            ]
            .isin(
                completed
            )
        ]
        .copy()
    )

    pending = (
        pending
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
# SHOW CURRENT PROGRESS
# =========================================================

def show_progress(
    c1: pd.DataFrame,
    completed: set,
):
    """
    Tampilkan progres C1 berdasarkan acquisition plan.
    """

    total = len(c1)

    completed_in_c1 = (
        c1[
            "recording_id"
        ]
        .isin(
            completed
        )
        .sum()
    )

    remaining = (
        total
        - completed_in_c1
    )

    percentage = (
        completed_in_c1
        / total
        * 100.0
    )

    print()
    print("=" * 70)
    print("C1 CURRENT PROGRESS")
    print("=" * 70)

    print()
    print(
        f"Completed : "
        f"{completed_in_c1}/{total}"
    )

    print(
        f"Remaining : "
        f"{remaining}"
    )

    print(
        f"Progress  : "
        f"{percentage:.2f}%"
    )


# =========================================================
# SHOW BATCH
# =========================================================

def show_batch(
    batch: pd.DataFrame,
):
    """
    Menampilkan batch sebelum recording.
    """

    print()
    print("=" * 70)
    print("NEXT C1 BATCH")
    print("=" * 70)

    if batch.empty:

        print()
        print(
            "Tidak ada recording tersisa."
        )

        return

    display = (
        batch
        .copy()
    )

    display.insert(
        0,
        "acquisition_row",
        display.index,
    )

    columns = [
        "acquisition_row",
        "source_file_id",
        "label",
        "split",
        "repetition",
        "recording_id",
    ]

    print()

    print(
        display[
            columns
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Batch size: "
        f"{len(batch)}"
    )

    print()
    print(
        "Label distribution:"
    )

    print(
        batch[
            "label"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Split distribution:"
    )

    print(
        batch[
            "split"
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
    """
    Memanggil play_and_record.py.
    """

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

    return (
        result.returncode
    )


# =========================================================
# POST-RUN STATUS
# =========================================================

def post_run_status(
    c1: pd.DataFrame,
):
    """
    Membaca manifest terbaru setelah batch.
    """

    existing = (
        load_existing_manifest()
    )

    completed = (
        get_completed_ids(
            existing
        )
    )

    completed_c1 = (
        c1[
            "recording_id"
        ]
        .isin(
            completed
        )
        .sum()
    )

    remaining = (
        len(c1)
        - completed_c1
    )

    print()
    print("=" * 70)
    print("C1 POST-RUN STATUS")
    print("=" * 70)

    print()
    print(
        f"Completed C1 : "
        f"{completed_c1}/"
        f"{len(c1)}"
    )

    print(
        f"Remaining C1 : "
        f"{remaining}"
    )

    # =====================================================
    # C1 MANIFEST SUMMARY
    # =====================================================

    if (
        not existing.empty
        and
        "config_id"
        in existing.columns
    ):

        c1_manifest = (
            existing[
                existing[
                    "config_id"
                ]
                .eq(
                    TARGET_CONFIG
                )
            ]
            .copy()
        )

        if not c1_manifest.empty:

            print()
            print(
                "Current C1 label distribution:"
            )

            print(
                c1_manifest[
                    "label"
                ]
                .value_counts()
                .sort_index()
            )

            print()
            print(
                "Current C1 split distribution:"
            )

            print(
                c1_manifest[
                    "split"
                ]
                .value_counts()
                .sort_index()
            )

            print()
            print(
                "Current level distribution:"
            )

            print(
                c1_manifest[
                    "level"
                ]
                .value_counts(
                    dropna=False
                )
            )

    return completed_c1


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("CONTROLLED REAL REPLAY - C1 FINAL")
    print("=" * 70)

    print()
    print(
        "FINAL LEVEL CONFIGURATION"
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

    print(
        f"Level note    : "
        f"{LEVEL_NOTE}"
    )

    # =====================================================
    # REQUIRED FILE CHECK
    # =====================================================

    if not PLAY_AND_RECORD.exists():

        raise FileNotFoundError(
            "play_and_record.py tidak ditemukan:\n"
            f"{PLAY_AND_RECORD}"
        )

    # =====================================================
    # LOAD PLAN
    # =====================================================

    plan, c1 = (
        load_acquisition_plan()
    )

    c1 = (
        prepare_c1(
            c1
        )
    )

    # =====================================================
    # VALIDATE PLAN
    # =====================================================

    validate_c1_plan(
        c1
    )

    # =====================================================
    # EXISTING MANIFEST
    # =====================================================

    existing = (
        load_existing_manifest()
    )

    validate_existing_c1(
        existing
    )

    completed = (
        get_completed_ids(
            existing
        )
    )

    # =====================================================
    # SHOW PROGRESS
    # =====================================================

    show_progress(
        c1,
        completed,
    )

    # =====================================================
    # SELECT NEXT BATCH
    # =====================================================

    pending, batch = (
        select_next_batch(
            c1,
            completed,
        )
    )

    # =====================================================
    # ALL COMPLETE
    # =====================================================

    if batch.empty:

        print()
        print("=" * 70)
        print("C1 RECORDING COMPLETE")
        print("=" * 70)

        print()
        print(
            f"All {EXPECTED_C1_TOTAL} "
            f"C1 recording sudah selesai."
        )

        print()
        print(
            "Jangan merekam lagi."
        )

        print(
            "Lanjutkan ke final QC C1."
        )

        return

    # =====================================================
    # SHOW BATCH
    # =====================================================

    show_batch(
        batch
    )

    # =====================================================
    # PHYSICAL SETUP
    # =====================================================

    print()
    print("=" * 70)
    print("FINAL C1 PHYSICAL SETUP")
    print("=" * 70)

    print()
    print(
        f"Config        : "
        f"{TARGET_CONFIG}"
    )

    print(
        f"Room          : "
        f"{EXPECTED_ROOM}"
    )

    print(
        f"Playback      : "
        f"{EXPECTED_PLAYBACK}"
    )

    print(
        f"Microphone    : "
        f"{EXPECTED_MIC}"
    )

    print(
        f"Distance      : "
        f"{EXPECTED_DISTANCE} m"
    )

    print(
        f"Angle         : "
        f"{EXPECTED_ANGLE} deg"
    )

    print(
        f"Mic level     : "
        f"{MIC_LEVEL}"
    )

    print(
        f"Speaker level : "
        f"{SPEAKER_LEVEL}"
    )

    print(
        f"Input device  : "
        f"{INPUT_DEVICE}"
    )

    print(
        f"Output device : "
        f"{OUTPUT_DEVICE}"
    )

    print(
        f"Level note    : "
        f"{LEVEL_NOTE}"
    )

    print(
        f"AGC           : "
        f"{AGC_STATUS}"
    )

    print()
    print(
        "PENTING:"
    )

    print(
        "Jangan ubah mic, speaker level, "
        "jarak, sudut, atau posisi selama C1."
    )

    # =====================================================
    # CONFIRM START
    # =====================================================

    input(
        "\nPastikan mic=70 dan speaker=50. "
        "Tekan ENTER untuk memulai batch..."
    )

    successful = []
    failed = []

    # =====================================================
    # RECORD LOOP
    # =====================================================

    for batch_number, (
        row_index,
        row,
    ) in enumerate(
        batch.iterrows(),
        start=1,
    ):

        recording_id = (
            row[
                "recording_id"
            ]
        )

        print()
        print("=" * 70)

        print(
            f"C1 RECORDING "
            f"{batch_number}/"
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
            f"{recording_id}"
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

        print()
        print(
            "FINAL SETTINGS:"
        )

        print(
            "Mic=70 | Speaker=50"
        )

        # =================================================
        # RUN PLAY + RECORD
        # =================================================

        return_code = (
            run_recording(
                row_index
            )
        )

        # =================================================
        # RESULT
        # =================================================

        if return_code == 0:

            successful.append(
                recording_id
            )

            print()
            print(
                f"[PASS] "
                f"{batch_number}/"
                f"{len(batch)} "
                f"{recording_id}"
            )

        else:

            failed.append(
                {
                    "row":
                        row_index,

                    "recording_id":
                        recording_id,

                    "return_code":
                        return_code,
                }
            )

            print()
            print("=" * 70)
            print("C1 BATCH STOPPED")
            print("=" * 70)

            print()
            print(
                "Recording gagal:"
            )

            print(
                recording_id
            )

            print()
            print(
                f"Acquisition row: "
                f"{row_index}"
            )

            print()
            print(
                "Jangan lanjutkan recording."
            )

            print(
                "Periksa QC/error terlebih dahulu."
            )

            break

    # =====================================================
    # BATCH SUMMARY
    # =====================================================

    print()
    print("=" * 70)
    print("C1 BATCH SUMMARY")
    print("=" * 70)

    print()
    print(
        f"Planned this run    : "
        f"{len(batch)}"
    )

    print(
        f"Successful this run : "
        f"{len(successful)}"
    )

    print(
        f"Failed this run     : "
        f"{len(failed)}"
    )

    # =====================================================
    # POST RUN
    # =====================================================

    completed_c1 = (
        post_run_status(
            c1
        )
    )

    # =====================================================
    # FINAL MESSAGE
    # =====================================================

    if failed:

        print()
        print(
            "C1 batch belum selesai."
        )

        print(
            "Setelah masalah diperbaiki, "
            "jalankan script ini lagi."
        )

        print(
            "Recording yang sudah valid "
            "akan otomatis dilewati."
        )

    elif (
        completed_c1
        < EXPECTED_C1_TOTAL
    ):

        print()
        print(
            "BATCH SELESAI."
        )

        print()
        print(
            "STOP recording dan lakukan QC."
        )

        print()
        print(
            "Setelah QC lolos, jalankan "
            "record_c1_final.py lagi "
            "untuk batch berikutnya."
        )

    else:

        print()
        print("=" * 70)
        print("C1 FINAL ACQUISITION COMPLETE")
        print("=" * 70)

        print()
        print(
            f"{EXPECTED_C1_TOTAL}/"
            f"{EXPECTED_C1_TOTAL} "
            f"recording C1 selesai."
        )

        print()
        print(
            "Lanjutkan ke final QC C1."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()