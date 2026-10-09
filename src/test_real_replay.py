from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


# =========================================================
# PROJECT PATH
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

REAL_MANIFEST = (
    ROOT
    / "manifests"
    / "manifest_real_replay.csv"
)

ACQUISITION_PLAN = (
    ROOT
    / "manifests"
    / "acquisition_plan.csv"
)


# =========================================================
# CONTROLLED REAL REPLAY DESIGN
# =========================================================

BATCH_SIZE = 20

EXPECTED_RECORDINGS_PER_CONFIG = 120

TOTAL_EXPECTED_RECORDINGS = 480

KNOWN_CONFIGS = [
    "C1",
    "C2",
    "C3",
    "C4",
]


# =========================================================
# FINAL LEVEL CONFIGURATION
# =========================================================

EXPECTED_LEVEL_BY_CONFIG = {
    "C1": "mic70_speaker50",
    "C2": "mic70_speaker50",
    "C3": "mic70_speaker50",
    "C4": "mic70_speaker50",
}


# =========================================================
# FINAL AUDIO DEVICE CONFIGURATION
# =========================================================

EXPECTED_INPUT_DEVICE_BY_CONFIG = {
    "C1": 1,
    "C2": 1,
    "C3": 1,
    "C4": 1,
}

EXPECTED_OUTPUT_DEVICE_BY_CONFIG = {
    "C1": 3,
    "C2": 3,
    "C3": 3,
    "C4": 3,
}


# =========================================================
# AUDIO FORMAT
# =========================================================

MASTER_SR = 48000
MASTER_SUBTYPE = "PCM_24"

MODEL_SR = 16000
MODEL_SUBTYPE = "PCM_16"

EXPECTED_CHANNELS = 1


# =========================================================
# DURATION CONFIGURATION
# =========================================================

PRE_ROLL_SECONDS = 0.5
POST_ROLL_SECONDS = 0.5

EXPECTED_EXTRA_DURATION = (
    PRE_ROLL_SECONDS
    + POST_ROLL_SECONDS
)

DURATION_TOLERANCE = 0.02


# =========================================================
# QC THRESHOLDS
# =========================================================

SILENCE_RMS_THRESHOLD = 1e-6

CLIP_THRESHOLD = 0.999

MANIFEST_PEAK_TOLERANCE = 1e-3
MANIFEST_RMS_TOLERANCE = 1e-3


# =========================================================
# REQUIRED MANIFEST COLUMNS
# =========================================================

REQUIRED_COLUMNS = {
    "recording_id",
    "acquisition_row",
    "source_file_id",
    "source_file_path",
    "master_file_path",
    "model_file_path",
    "speaker_id",
    "attack_id",
    "label",
    "dataset",
    "split",
    "config_id",
    "room",
    "playback_device",
    "microphone",
    "distance_m",
    "angle_deg",
    "repetition",
    "input_device_id",
    "input_device_name",
    "output_device_id",
    "output_device_name",
    "level",
    "agc",
    "master_sample_rate",
    "master_bit_depth",
    "master_subtype",
    "model_sample_rate",
    "model_bit_depth",
    "model_subtype",
    "recording_duration_s",
    "peak",
    "rms",
    "finite",
    "silent",
    "clipped",
}


# =========================================================
# REQUIRED ACQUISITION PLAN COLUMNS
# =========================================================

PLAN_REQUIRED_COLUMNS = {
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


# =========================================================
# BOOLEAN NORMALIZER
# =========================================================

def normalize_bool(value):
    """
    Mengubah berbagai representasi boolean
    menjadi True / False.
    """

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    text = str(value).strip().lower()

    if text in {
        "true",
        "1",
        "yes",
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
    }:
        return False

    raise ValueError(
        f"Invalid boolean value: {value}"
    )


# =========================================================
# AUDIO STATISTICS
# =========================================================

def compute_audio_stats(
    path: Path,
):
    """
    Membaca WAV aktual dan menghitung:
        sample rate
        channels
        subtype
        duration
        peak
        RMS
        finite
        silent
        clipped
    """

    info = sf.info(
        str(path)
    )

    audio, _ = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )

    if audio.size == 0:

        raise ValueError(
            f"Audio kosong: {path}"
        )

    finite = bool(
        np.isfinite(audio).all()
    )

    if finite:

        peak = float(
            np.max(
                np.abs(audio)
            )
        )

        rms = float(
            np.sqrt(
                np.mean(
                    np.square(audio)
                )
            )
        )

    else:

        peak = float("nan")
        rms = float("nan")

    duration = (
        info.frames
        / info.samplerate
    )

    silent = (
        finite
        and
        rms < SILENCE_RMS_THRESHOLD
    )

    clipped = (
        finite
        and
        peak >= CLIP_THRESHOLD
    )

    return {
        "sample_rate":
            int(
                info.samplerate
            ),

        "channels":
            int(
                info.channels
            ),

        "subtype":
            str(
                info.subtype
            ),

        "duration_s":
            float(
                duration
            ),

        "peak":
            peak,

        "rms":
            rms,

        "finite":
            finite,

        "silent":
            silent,

        "clipped":
            clipped,
    }


# =========================================================
# SOURCE DURATION
# =========================================================

def get_source_duration(
    path: Path,
):

    info = sf.info(
        str(path)
    )

    return (
        info.frames
        / info.samplerate
    )


# =========================================================
# RECORDING ID FROM PLAN
# =========================================================

def build_recording_id_from_plan(
    row,
):

    return (
        f"{row['source_file_id']}"
        f"__{row['config_id']}"
        f"__rep{int(row['repetition'])}"
    )


# =========================================================
# LOAD MANIFEST
# =========================================================

def load_manifest():

    if not REAL_MANIFEST.exists():

        raise FileNotFoundError(
            "manifest_real_replay.csv "
            "tidak ditemukan:\n"
            f"{REAL_MANIFEST}"
        )

    try:

        manifest = pd.read_csv(
            REAL_MANIFEST
        )

    except pd.errors.EmptyDataError:

        raise RuntimeError(
            "manifest_real_replay.csv kosong."
        )

    if manifest.empty:

        raise RuntimeError(
            "Belum ada real replay recording "
            "untuk di-QC."
        )

    missing = (
        REQUIRED_COLUMNS
        - set(
            manifest.columns
        )
    )

    if missing:

        raise ValueError(
            "Manifest kehilangan kolom:\n"
            f"{sorted(missing)}"
        )

    return manifest


# =========================================================
# LOAD ACQUISITION PLAN
# =========================================================

def load_acquisition_plan():

    if not ACQUISITION_PLAN.exists():

        raise FileNotFoundError(
            "acquisition_plan.csv tidak ditemukan:\n"
            f"{ACQUISITION_PLAN}"
        )

    plan = pd.read_csv(
        ACQUISITION_PLAN
    )

    missing = (
        PLAN_REQUIRED_COLUMNS
        - set(
            plan.columns
        )
    )

    if missing:

        raise ValueError(
            "Acquisition plan kehilangan kolom:\n"
            f"{sorted(missing)}"
        )

    plan = (
        plan
        .copy()
    )

    plan[
        "recording_id"
    ] = plan.apply(
        build_recording_id_from_plan,
        axis=1,
    )

    return plan


# =========================================================
# MANIFEST STRUCTURE VALIDATION
# =========================================================

def validate_manifest_structure(
    manifest: pd.DataFrame,
):

    print()
    print("=" * 70)
    print("MANIFEST STRUCTURE CHECK")
    print("=" * 70)

    total = len(
        manifest
    )

    print()
    print(
        f"Manifest rows          : "
        f"{total}"
    )

    print(
        f"Unique recording IDs   : "
        f"{manifest['recording_id'].nunique()}"
    )

    print(
        f"Unique acquisition rows: "
        f"{manifest['acquisition_row'].nunique()}"
    )

    # =====================================================
    # TOTAL LIMIT
    # =====================================================

    assert (
        total
        <= TOTAL_EXPECTED_RECORDINGS
    ), (
        f"Manifest melebihi "
        f"{TOTAL_EXPECTED_RECORDINGS} recording."
    )

    # =====================================================
    # RECORDING ID UNIQUE
    # =====================================================

    assert (
        manifest[
            "recording_id"
        ]
        .nunique()
        == total
    ), (
        "Duplicate recording_id ditemukan."
    )

    # =====================================================
    # ACQUISITION ROW UNIQUE
    # =====================================================

    assert (
        manifest[
            "acquisition_row"
        ]
        .nunique()
        == total
    ), (
        "Duplicate acquisition_row ditemukan."
    )

    # =====================================================
    # VALID CONFIG
    # =====================================================

    invalid_configs = (
        set(
            manifest[
                "config_id"
            ]
            .astype(str)
            .unique()
        )
        - set(
            KNOWN_CONFIGS
        )
    )

    assert not invalid_configs, (
        "Config tidak dikenal ditemukan: "
        f"{sorted(invalid_configs)}"
    )

    print()
    print(
        "Manifest structure: PASSED"
    )


# =========================================================
# ACQUISITION PLAN CONSISTENCY
# =========================================================

def validate_against_plan(
    manifest: pd.DataFrame,
    plan: pd.DataFrame,
):

    print()
    print("=" * 70)
    print("ACQUISITION PLAN CONSISTENCY")
    print("=" * 70)

    plan_lookup = (
        plan
        .set_index(
            "recording_id",
            drop=False,
        )
    )

    mismatches = []

    for _, row in (
        manifest.iterrows()
    ):

        rid = str(
            row[
                "recording_id"
            ]
        )

        if rid not in plan_lookup.index:

            mismatches.append(
                (
                    rid,
                    "recording_id tidak ada "
                    "di acquisition plan",
                )
            )

            continue

        expected = (
            plan_lookup.loc[
                rid
            ]
        )

        if isinstance(
            expected,
            pd.DataFrame,
        ):

            mismatches.append(
                (
                    rid,
                    "recording_id duplicate "
                    "di acquisition plan",
                )
            )

            continue

        checks = {
            "source_file_id":
                str(
                    row[
                        "source_file_id"
                    ]
                )
                ==
                str(
                    expected[
                        "source_file_id"
                    ]
                ),

            "label":
                int(
                    row[
                        "label"
                    ]
                )
                ==
                int(
                    expected[
                        "label"
                    ]
                ),

            "split":
                str(
                    row[
                        "split"
                    ]
                )
                ==
                str(
                    expected[
                        "split"
                    ]
                ),

            "config_id":
                str(
                    row[
                        "config_id"
                    ]
                )
                ==
                str(
                    expected[
                        "config_id"
                    ]
                ),

            "room":
                str(
                    row[
                        "room"
                    ]
                )
                ==
                str(
                    expected[
                        "room"
                    ]
                ),

            "playback_device":
                str(
                    row[
                        "playback_device"
                    ]
                )
                ==
                str(
                    expected[
                        "playback"
                    ]
                ),

            "microphone":
                str(
                    row[
                        "microphone"
                    ]
                )
                ==
                str(
                    expected[
                        "mic"
                    ]
                ),

            "distance_m":
                np.isclose(
                    float(
                        row[
                            "distance_m"
                        ]
                    ),
                    float(
                        expected[
                            "distance"
                        ]
                    ),
                ),

            "angle_deg":
                np.isclose(
                    float(
                        row[
                            "angle_deg"
                        ]
                    ),
                    float(
                        expected[
                            "angle"
                        ]
                    ),
                ),

            "repetition":
                int(
                    row[
                        "repetition"
                    ]
                )
                ==
                int(
                    expected[
                        "repetition"
                    ]
                ),
        }

        failed_fields = [
            key
            for key, ok
            in checks.items()
            if not ok
        ]

        if failed_fields:

            mismatches.append(
                (
                    rid,
                    "Mismatch: "
                    + ", ".join(
                        failed_fields
                    ),
                )
            )

    if mismatches:

        print()
        print(
            "Acquisition-plan mismatches:"
        )

        for rid, reason in (
            mismatches[:20]
        ):

            print(
                f"[FAIL] "
                f"{rid}: "
                f"{reason}"
            )

    assert not mismatches, (
        f"Ditemukan {len(mismatches)} "
        "recording yang tidak konsisten "
        "dengan acquisition_plan.csv."
    )

    print()
    print(
        "Acquisition plan consistency: PASSED"
    )


# =========================================================
# CHECKPOINT VALIDATION
# =========================================================

def validate_checkpoints(
    manifest: pd.DataFrame,
):

    print()
    print("=" * 70)
    print("CHECKPOINT STATUS")
    print("=" * 70)

    for config in KNOWN_CONFIGS:

        config_df = (
            manifest[
                manifest[
                    "config_id"
                ].eq(
                    config
                )
            ]
            .copy()
        )

        count = len(
            config_df
        )

        print()
        print(
            f"{config}: "
            f"{count}/"
            f"{EXPECTED_RECORDINGS_PER_CONFIG}"
        )

        if count == 0:

            print(
                f"{config} status: NOT STARTED"
            )

            continue

        assert (
            count
            <= EXPECTED_RECORDINGS_PER_CONFIG
        ), (
            f"{config} melebihi "
            f"{EXPECTED_RECORDINGS_PER_CONFIG} "
            "recording."
        )

        # QC dilakukan setiap 20 recording.
        assert (
            count % BATCH_SIZE
            == 0
            or
            count
            == EXPECTED_RECORDINGS_PER_CONFIG
        ), (
            f"{config} memiliki "
            f"{count} recording. "
            f"Checkpoint QC seharusnya "
            f"setiap {BATCH_SIZE} recording."
        )

        if (
            count
            == EXPECTED_RECORDINGS_PER_CONFIG
        ):

            print(
                f"{config} status: COMPLETE"
            )

        else:

            print(
                f"{config} status: CHECKPOINT"
            )


# =========================================================
# LEVEL CONSISTENCY
# =========================================================

def validate_levels(
    manifest: pd.DataFrame,
):

    print()
    print("=" * 70)
    print("LEVEL CONSISTENCY")
    print("=" * 70)

    for config in KNOWN_CONFIGS:

        config_df = (
            manifest[
                manifest[
                    "config_id"
                ].eq(
                    config
                )
            ]
            .copy()
        )

        if config_df.empty:

            continue

        expected_level = (
            EXPECTED_LEVEL_BY_CONFIG[
                config
            ]
        )

        print()
        print(
            f"{config} level distribution:"
        )

        print(
            config_df[
                "level"
            ]
            .value_counts(
                dropna=False
            )
        )

        assert (
            config_df[
                "level"
            ]
            .astype(str)
            .eq(
                expected_level
            )
            .all()
        ), (
            f"{config} harus menggunakan "
            f"level={expected_level}."
        )

    print()
    print(
        "Level consistency: PASSED"
    )


# =========================================================
# DEVICE CONSISTENCY
# =========================================================
def validate_devices(
    manifest: pd.DataFrame,
):

    print()
    print("=" * 70)
    print("AUDIO DEVICE CONSISTENCY")
    print("=" * 70)

    # =====================================================
    # EXPECTED PHYSICAL DEVICE NAMES
    # =====================================================

    EXPECTED_INPUT_NAME = (
        "Microphone Array (Senary Audio)"
    )

    EXPECTED_OUTPUT_NAME = (
        "Speakers (Senary Audio)"
    )

    for config in KNOWN_CONFIGS:

        config_df = (
            manifest[
                manifest[
                    "config_id"
                ].eq(
                    config
                )
            ]
            .copy()
        )

        if config_df.empty:
            continue

        # =================================================
        # UNIQUE DEVICE IDS
        # =================================================

        input_ids = sorted(
            config_df[
                "input_device_id"
            ]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        output_ids = sorted(
            config_df[
                "output_device_id"
            ]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        # =================================================
        # UNIQUE DEVICE NAMES
        # =================================================

        input_names = (
            config_df[
                "input_device_name"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        output_names = (
            config_df[
                "output_device_name"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        # =================================================
        # PRINT
        # =================================================

        print()
        print(
            f"{config}:"
        )

        print(
            f"Input device IDs   : "
            f"{input_ids}"
        )

        print(
            f"Input device names : "
            f"{input_names}"
        )

        print(
            f"Output device IDs  : "
            f"{output_ids}"
        )

        print(
            f"Output device names: "
            f"{output_names}"
        )

        # =================================================
        # HARD CHECK:
        # one physical device name per config
        # =================================================

        assert (
            len(input_names)
            == 1
        ), (
            f"{config}: lebih dari satu "
            "input device name digunakan."
        )

        assert (
            len(output_names)
            == 1
        ), (
            f"{config}: lebih dari satu "
            "output device name digunakan."
        )

        # =================================================
        # HARD CHECK:
        # physical device must match expected device
        # =================================================

        assert (
            input_names[0]
            == EXPECTED_INPUT_NAME
        ), (
            f"{config}: input device berbeda.\n"
            f"Expected: {EXPECTED_INPUT_NAME}\n"
            f"Actual  : {input_names[0]}"
        )

        assert (
            output_names[0]
            == EXPECTED_OUTPUT_NAME
        ), (
            f"{config}: output device berbeda.\n"
            f"Expected: {EXPECTED_OUTPUT_NAME}\n"
            f"Actual  : {output_names[0]}"
        )

    print()
    print(
        "Audio device consistency: PASSED"
    )


# =========================================================
# FILE LEVEL QC
# =========================================================

def perform_file_qc(
    manifest: pd.DataFrame,
):

    print()
    print("=" * 70)
    print("FILE-LEVEL AUDIO QC")
    print("=" * 70)

    qc_rows = []
    failures = []

    manifest_reset = (
        manifest
        .reset_index(
            drop=True
        )
    )

    total = len(
        manifest_reset
    )

    for index, row in (
        manifest_reset.iterrows()
    ):

        rid = str(
            row[
                "recording_id"
            ]
        )

        try:

            # =============================================
            # PATH
            # =============================================

            source_path = (
                ROOT
                / str(
                    row[
                        "source_file_path"
                    ]
                )
            )

            master_path = (
                ROOT
                / str(
                    row[
                        "master_file_path"
                    ]
                )
            )

            model_path = (
                ROOT
                / str(
                    row[
                        "model_file_path"
                    ]
                )
            )

            # =============================================
            # EXISTENCE
            # =============================================

            if not source_path.exists():

                raise FileNotFoundError(
                    f"Source missing: "
                    f"{source_path}"
                )

            if not master_path.exists():

                raise FileNotFoundError(
                    f"Master missing: "
                    f"{master_path}"
                )

            if not model_path.exists():

                raise FileNotFoundError(
                    f"Model copy missing: "
                    f"{model_path}"
                )

            # =============================================
            # SOURCE DURATION
            # =============================================

            source_duration = (
                get_source_duration(
                    source_path
                )
            )

            # =============================================
            # MASTER
            # =============================================

            master = (
                compute_audio_stats(
                    master_path
                )
            )

            # =============================================
            # MODEL
            # =============================================

            model = (
                compute_audio_stats(
                    model_path
                )
            )

            # =============================================
            # EXPECTED DURATION
            # =============================================

            expected_duration = (
                source_duration
                + EXPECTED_EXTRA_DURATION
            )

            master_duration_diff = abs(
                master[
                    "duration_s"
                ]
                - expected_duration
            )

            model_duration_diff = abs(
                model[
                    "duration_s"
                ]
                - expected_duration
            )

            master_model_duration_diff = abs(
                master[
                    "duration_s"
                ]
                - model[
                    "duration_s"
                ]
            )

            manifest_duration_diff = abs(
                master[
                    "duration_s"
                ]
                - float(
                    row[
                        "recording_duration_s"
                    ]
                )
            )

            duration_ok = (
                master_duration_diff
                <= DURATION_TOLERANCE
                and
                model_duration_diff
                <= DURATION_TOLERANCE
                and
                master_model_duration_diff
                <= DURATION_TOLERANCE
            )

            manifest_duration_ok = (
                manifest_duration_diff
                <= DURATION_TOLERANCE
            )

            # =============================================
            # MASTER FORMAT
            # =============================================

            master_format_ok = (
                master[
                    "sample_rate"
                ]
                == MASTER_SR
                and
                master[
                    "subtype"
                ]
                == MASTER_SUBTYPE
                and
                master[
                    "channels"
                ]
                == EXPECTED_CHANNELS
            )

            # =============================================
            # MODEL FORMAT
            # =============================================

            model_format_ok = (
                model[
                    "sample_rate"
                ]
                == MODEL_SR
                and
                model[
                    "subtype"
                ]
                == MODEL_SUBTYPE
                and
                model[
                    "channels"
                ]
                == EXPECTED_CHANNELS
            )

            # =============================================
            # MANIFEST FORMAT
            # =============================================

            manifest_format_ok = (
                int(
                    row[
                        "master_sample_rate"
                    ]
                )
                == MASTER_SR
                and
                int(
                    row[
                        "master_bit_depth"
                    ]
                )
                == 24
                and
                str(
                    row[
                        "master_subtype"
                    ]
                )
                == MASTER_SUBTYPE
                and
                int(
                    row[
                        "model_sample_rate"
                    ]
                )
                == MODEL_SR
                and
                int(
                    row[
                        "model_bit_depth"
                    ]
                )
                == 16
                and
                str(
                    row[
                        "model_subtype"
                    ]
                )
                == MODEL_SUBTYPE
            )

            # =============================================
            # PEAK / RMS VS MANIFEST
            # =============================================

            peak_diff = abs(
                master[
                    "peak"
                ]
                - float(
                    row[
                        "peak"
                    ]
                )
            )

            rms_diff = abs(
                master[
                    "rms"
                ]
                - float(
                    row[
                        "rms"
                    ]
                )
            )

            manifest_stats_ok = (
                peak_diff
                <= MANIFEST_PEAK_TOLERANCE
                and
                rms_diff
                <= MANIFEST_RMS_TOLERANCE
            )

            # =============================================
            # QC FLAGS VS MANIFEST
            # =============================================

            manifest_qc_flags_ok = (
                normalize_bool(
                    row[
                        "finite"
                    ]
                )
                == master[
                    "finite"
                ]
                and
                normalize_bool(
                    row[
                        "silent"
                    ]
                )
                == master[
                    "silent"
                ]
                and
                normalize_bool(
                    row[
                        "clipped"
                    ]
                )
                == master[
                    "clipped"
                ]
            )

            # =============================================
            # STORE
            # =============================================

            qc_rows.append(
                {
                    "recording_id":
                        rid,

                    "acquisition_row":
                        int(
                            row[
                                "acquisition_row"
                            ]
                        ),

                    "source_file_id":
                        row[
                            "source_file_id"
                        ],

                    "config_id":
                        row[
                            "config_id"
                        ],

                    "label":
                        int(
                            row[
                                "label"
                            ]
                        ),

                    "split":
                        row[
                            "split"
                        ],

                    "repetition":
                        int(
                            row[
                                "repetition"
                            ]
                        ),

                    "level":
                        row[
                            "level"
                        ],

                    "source_duration_s":
                        source_duration,

                    "expected_duration_s":
                        expected_duration,

                    "master_duration_s":
                        master[
                            "duration_s"
                        ],

                    "model_duration_s":
                        model[
                            "duration_s"
                        ],

                    "master_sr":
                        master[
                            "sample_rate"
                        ],

                    "master_subtype":
                        master[
                            "subtype"
                        ],

                    "master_channels":
                        master[
                            "channels"
                        ],

                    "model_sr":
                        model[
                            "sample_rate"
                        ],

                    "model_subtype":
                        model[
                            "subtype"
                        ],

                    "model_channels":
                        model[
                            "channels"
                        ],

                    "master_peak":
                        master[
                            "peak"
                        ],

                    "master_rms":
                        master[
                            "rms"
                        ],

                    "finite":
                        master[
                            "finite"
                        ],

                    "silent":
                        master[
                            "silent"
                        ],

                    "clipped":
                        master[
                            "clipped"
                        ],

                    "master_format_ok":
                        master_format_ok,

                    "model_format_ok":
                        model_format_ok,

                    "manifest_format_ok":
                        manifest_format_ok,

                    "duration_ok":
                        duration_ok,

                    "manifest_duration_ok":
                        manifest_duration_ok,

                    "manifest_stats_ok":
                        manifest_stats_ok,

                    "manifest_qc_flags_ok":
                        manifest_qc_flags_ok,
                }
            )

        except Exception as exc:

            failures.append(
                {
                    "recording_id":
                        rid,

                    "error":
                        str(
                            exc
                        ),
                }
            )

            print()
            print(
                f"[ERROR] "
                f"{rid}"
            )

            print(
                str(
                    exc
                )
            )

        # =============================================
        # PROGRESS
        # =============================================

        checked = (
            index + 1
        )

        if (
            checked % 20
            == 0
            or
            checked
            == total
        ):

            print(
                f"Checked "
                f"{checked}/"
                f"{total}"
            )

    qc = pd.DataFrame(
        qc_rows
    )

    return qc, failures


# =========================================================
# QC SUMMARY
# =========================================================

def print_qc_summary(
    manifest: pd.DataFrame,
    qc: pd.DataFrame,
    failures: list,
):

    print()
    print("=" * 70)
    print("REAL REPLAY QC SUMMARY")
    print("=" * 70)

    print()
    print(
        f"Manifest rows      : "
        f"{len(manifest)}"
    )

    print(
        f"Checked recordings : "
        f"{len(qc)}"
    )

    print(
        f"File-level errors  : "
        f"{len(failures)}"
    )

    print()
    print(
        "Config distribution:"
    )

    print(
        manifest[
            "config_id"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Label distribution:"
    )

    print(
        manifest[
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
        manifest[
            "split"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Config x label:"
    )

    print(
        pd.crosstab(
            manifest[
                "config_id"
            ],
            manifest[
                "label"
            ],
        )
    )

    print()
    print(
        "Config x split:"
    )

    print(
        pd.crosstab(
            manifest[
                "config_id"
            ],
            manifest[
                "split"
            ],
        )
    )

    print()
    print(
        "Repetition distribution:"
    )

    print(
        manifest[
            "repetition"
        ]
        .value_counts()
        .sort_index()
    )

    if qc.empty:

        return

    print()
    print(
        "Master format:"
    )

    print(
        qc[
            [
                "master_sr",
                "master_subtype",
                "master_channels",
            ]
        ]
        .value_counts()
    )

    print()
    print(
        "Model format:"
    )

    print(
        qc[
            [
                "model_sr",
                "model_subtype",
                "model_channels",
            ]
        ]
        .value_counts()
    )

    print()
    print(
        "Master format OK:"
    )

    print(
        qc[
            "master_format_ok"
        ]
        .value_counts()
    )

    print()
    print(
        "Model format OK:"
    )

    print(
        qc[
            "model_format_ok"
        ]
        .value_counts()
    )

    print()
    print(
        "Manifest format metadata OK:"
    )

    print(
        qc[
            "manifest_format_ok"
        ]
        .value_counts()
    )

    print()
    print(
        "Duration OK:"
    )

    print(
        qc[
            "duration_ok"
        ]
        .value_counts()
    )

    print()
    print(
        "Manifest duration match:"
    )

    print(
        qc[
            "manifest_duration_ok"
        ]
        .value_counts()
    )

    print()
    print(
        "Finite:"
    )

    print(
        qc[
            "finite"
        ]
        .value_counts()
    )

    print()
    print(
        "Silent:"
    )

    print(
        qc[
            "silent"
        ]
        .value_counts()
    )

    print()
    print(
        "Clipped:"
    )

    print(
        qc[
            "clipped"
        ]
        .value_counts()
    )

    print()
    print(
        "Manifest peak/RMS match:"
    )

    print(
        qc[
            "manifest_stats_ok"
        ]
        .value_counts()
    )

    print()
    print(
        "Manifest QC flags match:"
    )

    print(
        qc[
            "manifest_qc_flags_ok"
        ]
        .value_counts()
    )

    print()
    print(
        "Peak statistics:"
    )

    print(
        qc[
            "master_peak"
        ]
        .describe()
    )

    print()
    print(
        "RMS statistics:"
    )

    print(
        qc[
            "master_rms"
        ]
        .describe()
    )

    print()
    print(
        "Peak statistics by config:"
    )

    print(
        qc
        .groupby(
            "config_id"
        )[
            "master_peak"
        ]
        .agg(
            [
                "count",
                "min",
                "mean",
                "max",
            ]
        )
    )

    print()
    print(
        "RMS statistics by config:"
    )

    print(
        qc
        .groupby(
            "config_id"
        )[
            "master_rms"
        ]
        .agg(
            [
                "count",
                "min",
                "mean",
                "max",
            ]
        )
    )


# =========================================================
# HARD AUDIO VALIDATION
# =========================================================

def validate_audio_qc(
    manifest: pd.DataFrame,
    qc: pd.DataFrame,
    failures: list,
):

    print()
    print("=" * 70)
    print("HARD AUDIO VALIDATION")
    print("=" * 70)

    assert (
        len(
            failures
        )
        == 0
    ), (
        f"Ada {len(failures)} "
        "recording gagal dibaca/diverifikasi."
    )

    assert (
        len(
            qc
        )
        == len(
            manifest
        )
    ), (
        "Jumlah hasil QC tidak sama "
        "dengan manifest."
    )

    assert (
        qc[
            "master_format_ok"
        ]
        .all()
    ), (
        "Ada master yang bukan "
        "48 kHz / PCM_24 / mono."
    )

    assert (
        qc[
            "model_format_ok"
        ]
        .all()
    ), (
        "Ada model copy yang bukan "
        "16 kHz / PCM_16 / mono."
    )

    assert (
        qc[
            "manifest_format_ok"
        ]
        .all()
    ), (
        "Metadata format manifest "
        "tidak sesuai file aktual."
    )

    assert (
        qc[
            "duration_ok"
        ]
        .all()
    ), (
        "Ada source/master/model "
        "dengan durasi tidak konsisten."
    )

    assert (
        qc[
            "manifest_duration_ok"
        ]
        .all()
    ), (
        "recording_duration_s "
        "tidak cocok dengan master WAV."
    )

    assert (
        qc[
            "finite"
        ]
        .all()
    ), (
        "Ada waveform mengandung NaN/Inf."
    )

    assert (
        ~qc[
            "silent"
        ]
    ).all(), (
        "Ada recording silent."
    )

    assert (
        ~qc[
            "clipped"
        ]
    ).all(), (
        "Ada recording clipping."
    )

    assert (
        qc[
            "manifest_stats_ok"
        ]
        .all()
    ), (
        "Peak/RMS manifest tidak cocok "
        "dengan WAV aktual."
    )

    assert (
        qc[
            "manifest_qc_flags_ok"
        ]
        .all()
    ), (
        "finite/silent/clipped manifest "
        "tidak cocok dengan WAV aktual."
    )

    print()
    print(
        "Hard audio validation: PASSED"
    )


# =========================================================
# COMPLETED CONFIG VALIDATION
# =========================================================

def validate_completed_configs(
    manifest: pd.DataFrame,
    plan: pd.DataFrame,
):

    print()
    print("=" * 70)
    print("COMPLETED CONFIG VALIDATION")
    print("=" * 70)

    for config in KNOWN_CONFIGS:

        actual = (
            manifest[
                manifest[
                    "config_id"
                ].eq(
                    config
                )
            ]
            .copy()
        )

        count = len(
            actual
        )

        if (
            count
            != EXPECTED_RECORDINGS_PER_CONFIG
        ):

            print()
            print(
                f"{config}: "
                f"{count}/120 "
                f"(belum final)"
            )

            continue

        expected = (
            plan[
                plan[
                    "config_id"
                ].eq(
                    config
                )
            ]
            .copy()
        )

        assert (
            len(
                expected
            )
            == EXPECTED_RECORDINGS_PER_CONFIG
        ), (
            f"Acquisition plan "
            f"{config} "
            "tidak berjumlah 120."
        )

        # =================================================
        # RECORDING IDS
        # =================================================

        actual_ids = set(
            actual[
                "recording_id"
            ]
            .astype(str)
        )

        expected_ids = set(
            expected[
                "recording_id"
            ]
            .astype(str)
        )

        assert (
            actual_ids
            == expected_ids
        ), (
            f"{config} sudah 120 "
            "tetapi recording IDs tidak sama "
            "dengan acquisition plan."
        )

        # =================================================
        # LABEL DISTRIBUTION
        # =================================================

        actual_labels = (
            actual[
                "label"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        expected_labels = (
            expected[
                "label"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        assert (
            actual_labels
            == expected_labels
        ), (
            f"{config}: "
            "label distribution "
            "tidak sesuai acquisition plan."
        )

        # =================================================
        # SPLIT DISTRIBUTION
        # =================================================

        actual_splits = (
            actual[
                "split"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        expected_splits = (
            expected[
                "split"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        assert (
            actual_splits
            == expected_splits
        ), (
            f"{config}: "
            "split distribution "
            "tidak sesuai acquisition plan."
        )

        # =================================================
        # REPETITION DISTRIBUTION
        # =================================================

        actual_repetitions = (
            actual[
                "repetition"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        expected_repetitions = (
            expected[
                "repetition"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        assert (
            actual_repetitions
            == expected_repetitions
        ), (
            f"{config}: "
            "repetition distribution "
            "tidak sesuai acquisition plan."
        )

        print()
        print(
            f"{config}: "
            "FINAL 120/120 PASSED"
        )

        print(
            f"Label distribution: "
            f"{actual_labels}"
        )

        print(
            f"Split distribution: "
            f"{actual_splits}"
        )

        print(
            f"Repetition distribution: "
            f"{actual_repetitions}"
        )


# =========================================================
# OVERALL STATUS
# =========================================================

def print_overall_status(
    manifest: pd.DataFrame,
):

    print()
    print("=" * 70)
    print("OVERALL REAL REPLAY STATUS")
    print("=" * 70)

    grand_total = 0

    for config in KNOWN_CONFIGS:

        count = int(
            manifest[
                "config_id"
            ]
            .eq(
                config
            )
            .sum()
        )

        grand_total += count

        if (
            count
            == EXPECTED_RECORDINGS_PER_CONFIG
        ):

            status = "COMPLETE"

        elif (
            count
            == 0
        ):

            status = "NOT STARTED"

        else:

            status = "IN PROGRESS"

        print(
            f"{config}: "
            f"{count:3d}/"
            f"{EXPECTED_RECORDINGS_PER_CONFIG} "
            f"[{status}]"
        )

    print()
    print(
        f"TOTAL: "
        f"{grand_total}/"
        f"{TOTAL_EXPECTED_RECORDINGS}"
    )

    percentage = (
        grand_total
        / TOTAL_EXPECTED_RECORDINGS
        * 100.0
    )

    print(
        f"PROGRESS: "
        f"{percentage:.2f}%"
    )


# =========================================================
# FINAL DATASET VALIDATION
# =========================================================

def validate_final_dataset(
    manifest: pd.DataFrame,
):

    if (
        len(
            manifest
        )
        != TOTAL_EXPECTED_RECORDINGS
    ):

        return False

    print()
    print("=" * 70)
    print("FINAL 480 RECORDING VALIDATION")
    print("=" * 70)

    # =====================================================
    # EACH CONFIG = 120
    # =====================================================

    for config in KNOWN_CONFIGS:

        count = int(
            manifest[
                "config_id"
            ]
            .eq(
                config
            )
            .sum()
        )

        assert (
            count
            == EXPECTED_RECORDINGS_PER_CONFIG
        ), (
            f"{config} bukan 120 recording."
        )

    # =====================================================
    # TOTAL UNIQUE
    # =====================================================

    assert (
        manifest[
            "recording_id"
        ]
        .nunique()
        == TOTAL_EXPECTED_RECORDINGS
    ), (
        "Final dataset memiliki duplicate "
        "recording_id."
    )

    assert (
        manifest[
            "acquisition_row"
        ]
        .nunique()
        == TOTAL_EXPECTED_RECORDINGS
    ), (
        "Final dataset memiliki duplicate "
        "acquisition_row."
    )

    print()
    print(
        "FINAL CONTROLLED REAL REPLAY "
        "DATASET: PASSED"
    )

    return True


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("CONTROLLED REAL REPLAY - FINAL QC")
    print("=" * 70)

    # =====================================================
    # LOAD
    # =====================================================

    manifest = (
        load_manifest()
    )

    plan = (
        load_acquisition_plan()
    )

    # =====================================================
    # STRUCTURE
    # =====================================================

    validate_manifest_structure(
        manifest
    )

    # =====================================================
    # ACQUISITION PLAN
    # =====================================================

    validate_against_plan(
        manifest,
        plan,
    )

    # =====================================================
    # CHECKPOINT
    # =====================================================

    validate_checkpoints(
        manifest
    )

    # =====================================================
    # LEVEL
    # =====================================================

    validate_levels(
        manifest
    )

    # =====================================================
    # DEVICES
    # =====================================================

    validate_devices(
        manifest
    )

    # =====================================================
    # FILE QC
    # =====================================================

    qc, failures = (
        perform_file_qc(
            manifest
        )
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    print_qc_summary(
        manifest,
        qc,
        failures,
    )

    # =====================================================
    # HARD VALIDATION
    # =====================================================

    validate_audio_qc(
        manifest,
        qc,
        failures,
    )

    # =====================================================
    # CONFIG VALIDATION
    # =====================================================

    validate_completed_configs(
        manifest,
        plan,
    )

    # =====================================================
    # STATUS
    # =====================================================

    print_overall_status(
        manifest
    )

    # =====================================================
    # FINAL DATASET CHECK
    # =====================================================

    final_complete = (
        validate_final_dataset(
            manifest
        )
    )

    # =====================================================
    # SUCCESS
    # =====================================================

    print()
    print("=" * 70)
    print("VALIDATION PASSED")
    print("=" * 70)

    print()
    print(
        f"{len(manifest)} recording "
        "berhasil melewati QC."
    )

    print()
    print(
        "Tidak ditemukan:"
    )

    print(
        "- missing file"
    )

    print(
        "- duplicate recording"
    )

    print(
        "- format error"
    )

    print(
        "- duration mismatch"
    )

    print(
        "- silence"
    )

    print(
        "- clipping"
    )

    print(
        "- manifest mismatch"
    )

    if final_complete:

        print()
        print("=" * 70)
        print(
            "CONTROLLED REAL REPLAY COMPLETE"
        )
        print("=" * 70)

        print()
        print(
            "C1 = 120/120"
        )

        print(
            "C2 = 120/120"
        )

        print(
            "C3 = 120/120"
        )

        print(
            "C4 = 120/120"
        )

        print()
        print(
            "TOTAL = 480/480"
        )

        print()
        print(
            "Tahap controlled real replay "
            "selesai."
        )

    else:

        print()
        print(
            "Checkpoint ini boleh dilanjutkan "
            "ke batch recording berikutnya."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()