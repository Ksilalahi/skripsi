from pathlib import Path
import subprocess
import sys

import pandas as pd


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
# FINAL C3 CONFIGURATION
# =========================================================

TARGET_CONFIG = "C3"

EXPECTED_ROOM = "B"
EXPECTED_PLAYBACK = "speaker_A"
EXPECTED_MIC = "mic_unseen"

EXPECTED_DISTANCE = 0.5
EXPECTED_ANGLE = 45.0


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
EXPECTED_C3_TOTAL = 120


def build_recording_id(row):

    return (
        f"{row['source_file_id']}"
        f"__{row['config_id']}"
        f"__rep{int(row['repetition'])}"
    )


def load_acquisition_plan():

    if not ACQUISITION_PLAN.exists():
        raise FileNotFoundError(
            ACQUISITION_PLAN
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
            f"Missing columns: {sorted(missing)}"
        )

    c3 = (
        plan[
            plan["config_id"].eq(
                TARGET_CONFIG
            )
        ]
        .copy()
    )

    if len(c3) != EXPECTED_C3_TOTAL:

        raise RuntimeError(
            f"Expected C3 = {EXPECTED_C3_TOTAL}, "
            f"actual = {len(c3)}"
        )

    return plan, c3


def validate_c3_plan(
    c3: pd.DataFrame,
):

    print()
    print("=" * 70)
    print("VALIDATING C3 ACQUISITION PLAN")
    print("=" * 70)

    assert (
        c3["config_id"]
        .eq(TARGET_CONFIG)
        .all()
    )

    assert (
        c3["room"]
        .astype(str)
        .eq(EXPECTED_ROOM)
        .all()
    ), "Room C3 tidak sesuai."

    assert (
        c3["playback"]
        .eq(EXPECTED_PLAYBACK)
        .all()
    ), "Playback C3 tidak sesuai."

    assert (
        c3["mic"]
        .eq(EXPECTED_MIC)
        .all()
    ), "Microphone C3 tidak sesuai."

    assert (
        c3["distance"]
        .astype(float)
        .sub(EXPECTED_DISTANCE)
        .abs()
        .lt(1e-9)
        .all()
    ), "Distance C3 tidak sesuai."

    assert (
        c3["angle"]
        .astype(float)
        .sub(EXPECTED_ANGLE)
        .abs()
        .lt(1e-9)
        .all()
    ), "Angle C3 tidak sesuai."

    print()
    print("C3 acquisition plan: PASSED")

    print()
    print(
        f"Total C3 planned: {len(c3)}"
    )

    print()
    print("Label distribution:")

    print(
        c3["label"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Split distribution:")

    print(
        c3["split"]
        .value_counts()
        .sort_index()
    )


def prepare_c3(
    c3: pd.DataFrame,
):

    c3 = c3.copy()

    c3["recording_id"] = (
        c3.apply(
            build_recording_id,
            axis=1,
        )
    )

    return c3


def load_existing_manifest():

    if not REAL_MANIFEST.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            REAL_MANIFEST
        )

    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def validate_previous_configs(
    existing: pd.DataFrame,
):

    if existing.empty:
        raise RuntimeError(
            "Manifest kosong."
        )

    c1_count = int(
        existing[
            "config_id"
        ]
        .eq("C1")
        .sum()
    )

    c2_count = int(
        existing[
            "config_id"
        ]
        .eq("C2")
        .sum()
    )

    if c1_count != 120:

        raise RuntimeError(
            f"C1 belum complete: {c1_count}/120"
        )

    if c2_count != 120:

        raise RuntimeError(
            f"C2 belum complete: {c2_count}/120"
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


def validate_existing_c3(
    existing: pd.DataFrame,
):

    if existing.empty:
        return

    c3 = (
        existing[
            existing[
                "config_id"
            ].eq(TARGET_CONFIG)
        ]
        .copy()
    )

    if c3.empty:

        print()
        print(
            "Existing C3 recording: 0"
        )

        return

    print()
    print("=" * 70)
    print("VALIDATING EXISTING C3")
    print("=" * 70)

    print()
    print(
        f"Existing C3 recording: "
        f"{len(c3)}"
    )

    if (
        c3["recording_id"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate recording_id pada C3."
        )

    if not (
        c3["level"]
        .astype(str)
        .eq(LEVEL_NOTE)
        .all()
    ):

        raise RuntimeError(
            "Level C3 tidak konsisten."
        )

    if not (
        c3["input_device_id"]
        .astype(int)
        .eq(INPUT_DEVICE)
        .all()
    ):

        raise RuntimeError(
            "Input device C3 berubah."
        )

    if not (
        c3["output_device_id"]
        .astype(int)
        .eq(OUTPUT_DEVICE)
        .all()
    ):

        raise RuntimeError(
            "Output device C3 berubah."
        )

    print()
    print(
        "Existing C3 validation: PASSED"
    )


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


def select_next_batch(
    c3: pd.DataFrame,
    completed: set,
):

    pending = (
        c3[
            ~c3[
                "recording_id"
            ]
            .isin(completed)
        ]
        .copy()
        .sort_index()
    )

    batch = (
        pending
        .head(BATCH_SIZE)
        .copy()
    )

    return pending, batch


def show_progress(
    c3: pd.DataFrame,
    completed: set,
):

    completed_c3 = int(
        c3[
            "recording_id"
        ]
        .isin(completed)
        .sum()
    )

    remaining = (
        EXPECTED_C3_TOTAL
        - completed_c3
    )

    progress = (
        completed_c3
        / EXPECTED_C3_TOTAL
        * 100
    )

    print()
    print("=" * 70)
    print("C3 CURRENT PROGRESS")
    print("=" * 70)

    print()
    print(
        f"Completed : {completed_c3}/120"
    )

    print(
        f"Remaining : {remaining}"
    )

    print(
        f"Progress  : {progress:.2f}%"
    )


def show_batch(
    batch: pd.DataFrame,
):

    print()
    print("=" * 70)
    print("NEXT C3 BATCH")
    print("=" * 70)

    if batch.empty:

        print()
        print(
            "Tidak ada recording C3 tersisa."
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
        .to_string(index=False)
    )

    print()
    print(
        f"Batch size: {len(batch)}"
    )


def run_recording(
    row_index: int,
):

    command = [
        sys.executable,
        str(PLAY_AND_RECORD),

        "--row",
        str(row_index),

        "--input-device",
        str(INPUT_DEVICE),

        "--output-device",
        str(OUTPUT_DEVICE),

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


def post_run_status(
    c3: pd.DataFrame,
):

    existing = (
        load_existing_manifest()
    )

    completed = (
        get_completed_ids(existing)
    )

    completed_c3 = int(
        c3[
            "recording_id"
        ]
        .isin(completed)
        .sum()
    )

    print()
    print("=" * 70)
    print("C3 POST-RUN STATUS")
    print("=" * 70)

    print()
    print(
        f"Completed C3 : "
        f"{completed_c3}/120"
    )

    print(
        f"Remaining C3 : "
        f"{120 - completed_c3}"
    )

    return completed_c3


def main():

    print("=" * 70)
    print("CONTROLLED REAL REPLAY - C3 FINAL")
    print("=" * 70)

    print()
    print("FINAL C3 SETTINGS")

    print()
    print(
        f"Config        : {TARGET_CONFIG}"
    )

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
        f"Input device  : {INPUT_DEVICE}"
    )

    print(
        f"Output device : {OUTPUT_DEVICE}"
    )

    print(
        f"Mic level     : {MIC_LEVEL}"
    )

    print(
        f"Speaker level : {SPEAKER_LEVEL}"
    )

    print(
        f"Level note    : {LEVEL_NOTE}"
    )

    if not PLAY_AND_RECORD.exists():

        raise FileNotFoundError(
            PLAY_AND_RECORD
        )

    _, c3 = (
        load_acquisition_plan()
    )

    c3 = (
        prepare_c3(c3)
    )

    validate_c3_plan(c3)

    existing = (
        load_existing_manifest()
    )

    validate_previous_configs(
        existing
    )

    validate_existing_c3(
        existing
    )

    completed = (
        get_completed_ids(existing)
    )

    show_progress(
        c3,
        completed,
    )

    _, batch = (
        select_next_batch(
            c3,
            completed,
        )
    )

    if batch.empty:

        print()
        print("=" * 70)
        print("C3 RECORDING COMPLETE")
        print("=" * 70)

        print()
        print(
            "C3 = 120/120"
        )

        print()
        print(
            "Jalankan:"
        )

        print(
            "python src\\test_real_replay.py"
        )

        return

    show_batch(batch)

    print()
    print("=" * 70)
    print("C3 PHYSICAL SETUP")
    print("=" * 70)

    print()
    print(
        "Room       = B"
    )

    print(
        "Playback   = speaker_A"
    )

    print(
        "Mic        = mic_unseen"
    )

    print(
        "Distance   = 0.5 m"
    )

    print(
        "Angle      = 45 deg"
    )

    print(
        "Mic level  = 70"
    )

    print(
        "Speaker    = 50"
    )

    input(
        "\nJika setup C3 sudah benar, "
        "tekan ENTER untuk mulai..."
    )

    successful = []
    failed = []

    for number, (
        row_index,
        row,
    ) in enumerate(
        batch.iterrows(),
        start=1,
    ):

        recording_id = (
            row["recording_id"]
        )

        print()
        print("=" * 70)

        print(
            f"C3 RECORDING "
            f"{number}/{len(batch)}"
        )

        print("=" * 70)

        print()
        print(
            f"Acquisition row : {row_index}"
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
                f"[PASS] {recording_id}"
            )

        else:

            failed.append(
                recording_id
            )

            print()
            print("=" * 70)
            print("C3 BATCH STOPPED")
            print("=" * 70)

            print()
            print(
                f"Recording gagal:"
            )

            print(
                recording_id
            )

            break

    print()
    print("=" * 70)
    print("C3 BATCH SUMMARY")
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

    completed_c3 = (
        post_run_status(c3)
    )

    if failed:

        print()
        print(
            "Setelah error diperiksa, "
            "jalankan file ini lagi."
        )

    elif completed_c3 < 120:

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
        print(
            "C3 FINAL ACQUISITION COMPLETE."
        )

        print()
        print(
            "Jalankan final QC."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()