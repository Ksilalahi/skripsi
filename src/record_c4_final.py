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
# FINAL C4 CONFIGURATION
# =========================================================

TARGET_CONFIG = "C4"

EXPECTED_ROOM = "B"
EXPECTED_PLAYBACK = "speaker_B"
EXPECTED_MIC = "mic_unseen"

EXPECTED_DISTANCE = 1.0
EXPECTED_ANGLE = 0.0


# =========================================================
# DEVICE / LEVEL
# =========================================================

INPUT_DEVICE = 1
OUTPUT_DEVICE = 3

MIC_LEVEL = 70
SPEAKER_LEVEL = 50

LEVEL_NOTE = "mic70_speaker50"

AGC_STATUS = "unknown"


# =========================================================
# BATCH
# =========================================================

BATCH_SIZE = 20
EXPECTED_C4_TOTAL = 120


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
# ACQUISITION PLAN
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

    c4 = (
        plan[
            plan[
                "config_id"
            ].eq(
                TARGET_CONFIG
            )
        ]
        .copy()
    )

    if len(c4) != EXPECTED_C4_TOTAL:

        raise RuntimeError(
            f"C4 seharusnya memiliki "
            f"{EXPECTED_C4_TOTAL} recording, "
            f"tetapi ditemukan {len(c4)}."
        )

    return plan, c4


# =========================================================
# VALIDATE C4 PLAN
# =========================================================

def validate_c4_plan(
    c4: pd.DataFrame,
):

    print()
    print("=" * 70)
    print("VALIDATING C4 ACQUISITION PLAN")
    print("=" * 70)

    assert (
        c4[
            "config_id"
        ]
        .eq(
            TARGET_CONFIG
        )
        .all()
    )

    assert (
        c4[
            "room"
        ]
        .astype(str)
        .eq(
            EXPECTED_ROOM
        )
        .all()
    ), (
        "Room C4 tidak sesuai."
    )

    assert (
        c4[
            "playback"
        ]
        .eq(
            EXPECTED_PLAYBACK
        )
        .all()
    ), (
        "Playback C4 tidak sesuai."
    )

    assert (
        c4[
            "mic"
        ]
        .eq(
            EXPECTED_MIC
        )
        .all()
    ), (
        "Microphone C4 tidak sesuai."
    )

    assert (
        c4[
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
        "Distance C4 tidak sesuai."
    )

    assert (
        c4[
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
        "Angle C4 tidak sesuai."
    )

    print()
    print(
        "C4 acquisition plan: PASSED"
    )

    print()
    print(
        f"Total C4 planned: "
        f"{len(c4)}"
    )

    print()
    print(
        "Label distribution:"
    )

    print(
        c4[
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
        c4[
            "split"
        ]
        .value_counts()
        .sort_index()
    )


# =========================================================
# PREPARE C4
# =========================================================

def prepare_c4(
    c4: pd.DataFrame,
):

    c4 = c4.copy()

    c4[
        "recording_id"
    ] = c4.apply(
        build_recording_id,
        axis=1,
    )

    return c4


# =========================================================
# LOAD MANIFEST
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
# VALIDATE PREVIOUS CONFIGS
# =========================================================

def validate_previous_configs(
    existing: pd.DataFrame,
):

    if existing.empty:

        raise RuntimeError(
            "Manifest kosong."
        )

    counts = {
        config:
            int(
                existing[
                    "config_id"
                ]
                .eq(config)
                .sum()
            )
        for config
        in [
            "C1",
            "C2",
            "C3",
        ]
    }

    for config in [
        "C1",
        "C2",
        "C3",
    ]:

        if counts[config] != 120:

            raise RuntimeError(
                f"{config} belum complete: "
                f"{counts[config]}/120"
            )

    print()
    print(
        "Prerequisite:"
    )

    print(
        "C1 = 120/120 PASSED"
    )

    print(
        "C2 = 120/120 PASSED"
    )

    print(
        "C3 = 120/120 PASSED"
    )


# =========================================================
# VALIDATE EXISTING C4
# =========================================================

def validate_existing_c4(
    existing: pd.DataFrame,
):

    if existing.empty:

        return

    c4 = (
        existing[
            existing[
                "config_id"
            ].eq(
                TARGET_CONFIG
            )
        ]
        .copy()
    )

    if c4.empty:

        print()
        print(
            "Existing C4 recording: 0"
        )

        return

    print()
    print("=" * 70)
    print("VALIDATING EXISTING C4")
    print("=" * 70)

    print()
    print(
        f"Existing C4 recording: "
        f"{len(c4)}"
    )

    if (
        c4[
            "recording_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate recording_id pada C4."
        )

    if not (
        c4[
            "level"
        ]
        .astype(str)
        .eq(
            LEVEL_NOTE
        )
        .all()
    ):

        raise RuntimeError(
            "Level C4 tidak konsisten."
        )

    if not (
        c4[
            "input_device_id"
        ]
        .astype(int)
        .eq(
            INPUT_DEVICE
        )
        .all()
    ):

        raise RuntimeError(
            "Input device C4 berubah."
        )

    if not (
        c4[
            "output_device_id"
        ]
        .astype(int)
        .eq(
            OUTPUT_DEVICE
        )
        .all()
    ):

        raise RuntimeError(
            "Output device C4 berubah."
        )

    print()
    print(
        "Existing C4 validation: PASSED"
    )


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
# NEXT BATCH
# =========================================================

def select_next_batch(
    c4: pd.DataFrame,
    completed: set,
):

    pending = (
        c4[
            ~c4[
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
# PROGRESS
# =========================================================

def show_progress(
    c4: pd.DataFrame,
    completed: set,
):

    completed_c4 = int(
        c4[
            "recording_id"
        ]
        .isin(
            completed
        )
        .sum()
    )

    remaining = (
        EXPECTED_C4_TOTAL
        - completed_c4
    )

    progress = (
        completed_c4
        / EXPECTED_C4_TOTAL
        * 100
    )

    print()
    print("=" * 70)
    print("C4 CURRENT PROGRESS")
    print("=" * 70)

    print()
    print(
        f"Completed : "
        f"{completed_c4}/120"
    )

    print(
        f"Remaining : "
        f"{remaining}"
    )

    print(
        f"Progress  : "
        f"{progress:.2f}%"
    )


# =========================================================
# SHOW BATCH
# =========================================================

def show_batch(
    batch: pd.DataFrame,
):

    print()
    print("=" * 70)
    print("NEXT C4 BATCH")
    print("=" * 70)

    if batch.empty:

        print()
        print(
            "Tidak ada recording C4 tersisa."
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
        f"Batch size: "
        f"{len(batch)}"
    )

    print()
    print(
        "Batch label distribution:"
    )

    print(
        batch[
            "label"
        ]
        .value_counts()
        .sort_index()
    )


# =========================================================
# RUN ONE
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

    return (
        result.returncode
    )


# =========================================================
# POST RUN
# =========================================================

def post_run_status(
    c4: pd.DataFrame,
):

    existing = (
        load_existing_manifest()
    )

    completed = (
        get_completed_ids(
            existing
        )
    )

    completed_c4 = int(
        c4[
            "recording_id"
        ]
        .isin(
            completed
        )
        .sum()
    )

    print()
    print("=" * 70)
    print("C4 POST-RUN STATUS")
    print("=" * 70)

    print()
    print(
        f"Completed C4 : "
        f"{completed_c4}/120"
    )

    print(
        f"Remaining C4 : "
        f"{120 - completed_c4}"
    )

    return completed_c4


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("CONTROLLED REAL REPLAY - C4 FINAL")
    print("=" * 70)

    print()
    print("FINAL C4 SETTINGS")

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
        f"Input device  : "
        f"{INPUT_DEVICE}"
    )

    print(
        f"Output device : "
        f"{OUTPUT_DEVICE}"
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
        f"Level note    : "
        f"{LEVEL_NOTE}"
    )

    # =====================================================
    # CHECK SCRIPT
    # =====================================================

    if not PLAY_AND_RECORD.exists():

        raise FileNotFoundError(
            PLAY_AND_RECORD
        )

    # =====================================================
    # LOAD PLAN
    # =====================================================

    _, c4 = (
        load_acquisition_plan()
    )

    c4 = (
        prepare_c4(
            c4
        )
    )

    validate_c4_plan(
        c4
    )

    # =====================================================
    # MANIFEST
    # =====================================================

    existing = (
        load_existing_manifest()
    )

    validate_previous_configs(
        existing
    )

    validate_existing_c4(
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
        c4,
        completed,
    )

    # =====================================================
    # NEXT BATCH
    # =====================================================

    _, batch = (
        select_next_batch(
            c4,
            completed,
        )
    )

    if batch.empty:

        print()
        print("=" * 70)
        print("C4 RECORDING COMPLETE")
        print("=" * 70)

        print()
        print(
            "C4 = 120/120"
        )

        print()
        print(
            "CONTROLLED REAL REPLAY "
            "= 480/480"
        )

        print()
        print(
            "Jalankan final QC:"
        )

        print(
            "python src\\test_real_replay.py"
        )

        return

    # =====================================================
    # SHOW
    # =====================================================

    show_batch(
        batch
    )

    # =====================================================
    # PHYSICAL SETUP
    # =====================================================

    print()
    print("=" * 70)
    print("C4 PHYSICAL SETUP")
    print("=" * 70)

    print()
    print(
        "Room       = B"
    )

    print(
        "Playback   = speaker_B"
    )

    print(
        "Mic        = mic_unseen"
    )

    print(
        "Distance   = 1.0 m"
    )

    print(
        "Angle      = 0 deg"
    )

    print(
        "Mic level  = 70"
    )

    print(
        "Speaker    = 50"
    )

    input(
        "\nJika setup C4 sudah benar, "
        "tekan ENTER untuk mulai..."
    )

    successful = []
    failed = []

    # =====================================================
    # RECORD
    # =====================================================

    for number, (
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
            f"C4 RECORDING "
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

        return_code = (
            run_recording(
                row_index
            )
        )

        if return_code == 0:

            successful.append(
                recording_id
            )

            print()
            print(
                f"[PASS] "
                f"{recording_id}"
            )

        else:

            failed.append(
                recording_id
            )

            print()
            print("=" * 70)
            print("C4 BATCH STOPPED")
            print("=" * 70)

            print()
            print(
                f"Recording gagal:"
            )

            print(
                recording_id
            )

            break

    # =====================================================
    # SUMMARY
    # =====================================================

    print()
    print("=" * 70)
    print("C4 BATCH SUMMARY")
    print("=" * 70)

    print()
    print(
        f"Planned    : "
        f"{len(batch)}"
    )

    print(
        f"Successful : "
        f"{len(successful)}"
    )

    print(
        f"Failed     : "
        f"{len(failed)}"
    )

    completed_c4 = (
        post_run_status(
            c4
        )
    )

    if failed:

        print()
        print(
            "Setelah error diperiksa, "
            "jalankan file ini lagi."
        )

    elif completed_c4 < 120:

        print()
        print(
            "STOP recording."
        )

        print()
        print(
            "Jalankan QC:"
        )

        print(
            "python src\\test_real_replay.py"
        )

    else:

        print()
        print("=" * 70)
        print(
            "C4 FINAL ACQUISITION COMPLETE"
        )
        print("=" * 70)

        print()
        print(
            "C4 = 120/120"
        )

        print(
            "TOTAL = 480/480"
        )

        print()
        print(
            "Jalankan FINAL QC:"
        )

        print(
            "python src\\test_real_replay.py"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()