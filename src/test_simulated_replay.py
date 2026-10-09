from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


# =========================================================
# PATH
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

SIM_MANIFEST = (
    ROOT
    / "manifests"
    / "manifest_replay_sim.csv"
)


# =========================================================
# CONFIGURATION
# =========================================================

EXPECTED_SR = 16000

MAX_DURATION_DIFF_S = 0.05

SILENCE_RMS_THRESHOLD = 1e-6

CLIP_THRESHOLD = 0.999

N_EXAMPLES = 10

RANDOM_SEED = 2026


# =========================================================
# AUDIO LOADER
# =========================================================

def load_audio(path: Path):
    """
    Load audio using SoundFile.

    Returns:
        audio: np.ndarray [time, channels]
        sample_rate: int
    """

    audio, sr = sf.read(
        str(path),
        always_2d=True,
        dtype="float32",
    )

    return audio, sr


# =========================================================
# AUDIO METRICS
# =========================================================

def compute_audio_stats(audio: np.ndarray):
    """
    Compute basic waveform statistics.
    """

    if audio.size == 0:
        raise ValueError(
            "Audio array kosong."
        )

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

    finite = bool(
        np.isfinite(audio).all()
    )

    silent = (
        rms < SILENCE_RMS_THRESHOLD
    )

    clipped = (
        peak >= CLIP_THRESHOLD
    )

    return {
        "peak": peak,
        "rms": rms,
        "finite": finite,
        "silent": silent,
        "clipped": clipped,
    }


# =========================================================
# DURATION
# =========================================================

def get_duration(
    audio: np.ndarray,
    sample_rate: int,
):
    return (
        len(audio)
        / sample_rate
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("SIMULATED REPLAY QC")
    print("=" * 70)

    # =====================================================
    # CHECK MANIFEST
    # =====================================================

    if not SIM_MANIFEST.exists():

        raise FileNotFoundError(
            f"Manifest simulated replay "
            f"tidak ditemukan:\n"
            f"{SIM_MANIFEST}"
        )

    manifest = pd.read_csv(
        SIM_MANIFEST
    )

    print()
    print("Manifest rows:")
    print(
        len(manifest)
    )

    # =====================================================
    # REQUIRED COLUMNS
    # =====================================================

    required_columns = {
        "file_id",
        "source_file_id",
        "file_path",
        "source_file_path",
        "label",
        "split",
        "condition",
        "rir_id",
        "rir_room",
        "rir_split",
        "snr_db",
        "sample_rate",
        "peak",
        "rms",
        "silent",
        "clipped",
    }

    missing_columns = (
        required_columns
        - set(
            manifest.columns
        )
    )

    if missing_columns:

        raise ValueError(
            "manifest_replay_sim.csv "
            "kehilangan kolom: "
            f"{sorted(missing_columns)}"
        )

    # =====================================================
    # STORAGE
    # =====================================================

    rows = []

    failed = []

    # =====================================================
    # CHECK EACH OUTPUT
    # =====================================================

    for index, row in (
        manifest
        .reset_index(drop=True)
        .iterrows()
    ):

        replay_path = (
            ROOT
            / row["file_path"]
        )

        source_path = (
            ROOT
            / row["source_file_path"]
        )

        try:

            # =============================================
            # FILE EXISTENCE
            # =============================================

            if not replay_path.exists():

                raise FileNotFoundError(
                    f"Replay file tidak ditemukan: "
                    f"{replay_path}"
                )

            if not source_path.exists():

                raise FileNotFoundError(
                    f"Source file tidak ditemukan: "
                    f"{source_path}"
                )

            # =============================================
            # LOAD REPLAY
            # =============================================

            replay_audio, replay_sr = (
                load_audio(
                    replay_path
                )
            )

            # =============================================
            # LOAD SOURCE
            # =============================================

            source_audio, source_sr = (
                load_audio(
                    source_path
                )
            )

            # =============================================
            # AUDIO STATS
            # =============================================

            replay_stats = (
                compute_audio_stats(
                    replay_audio
                )
            )

            source_stats = (
                compute_audio_stats(
                    source_audio
                )
            )

            # =============================================
            # DURATION
            # =============================================

            replay_duration = (
                get_duration(
                    replay_audio,
                    replay_sr,
                )
            )

            source_duration = (
                get_duration(
                    source_audio,
                    source_sr,
                )
            )

            duration_diff = abs(
                replay_duration
                - source_duration
            )

            # =============================================
            # VALIDATION FLAGS
            # =============================================

            sample_rate_ok = (
                replay_sr
                == EXPECTED_SR
            )

            duration_ok = (
                duration_diff
                <= MAX_DURATION_DIFF_S
            )

            manifest_peak_diff = abs(
                replay_stats["peak"]
                - float(
                    row["peak"]
                )
            )

            manifest_rms_diff = abs(
                replay_stats["rms"]
                - float(
                    row["rms"]
                )
            )

            # Small tolerance because PCM16 writing
            # slightly changes floating-point values.
            manifest_stats_ok = (
                manifest_peak_diff < 1e-3
                and
                manifest_rms_diff < 1e-3
            )

            # =============================================
            # STORE
            # =============================================

            rows.append(
                {
                    "file_id":
                        row["file_id"],

                    "source_file_id":
                        row[
                            "source_file_id"
                        ],

                    "split":
                        row["split"],

                    "label":
                        int(
                            row["label"]
                        ),

                    "snr_db":
                        row["snr_db"],

                    "rir_id":
                        row["rir_id"],

                    "rir_room":
                        row["rir_room"],

                    "rir_split":
                        row["rir_split"],

                    "replay_sr":
                        replay_sr,

                    "source_sr":
                        source_sr,

                    "replay_duration_s":
                        round(
                            replay_duration,
                            6,
                        ),

                    "source_duration_s":
                        round(
                            source_duration,
                            6,
                        ),

                    "duration_diff_s":
                        round(
                            duration_diff,
                            6,
                        ),

                    "replay_peak":
                        round(
                            replay_stats[
                                "peak"
                            ],
                            8,
                        ),

                    "replay_rms":
                        round(
                            replay_stats[
                                "rms"
                            ],
                            8,
                        ),

                    "source_peak":
                        round(
                            source_stats[
                                "peak"
                            ],
                            8,
                        ),

                    "source_rms":
                        round(
                            source_stats[
                                "rms"
                            ],
                            8,
                        ),

                    "finite":
                        replay_stats[
                            "finite"
                        ],

                    "silent":
                        replay_stats[
                            "silent"
                        ],

                    "clipped":
                        replay_stats[
                            "clipped"
                        ],

                    "sample_rate_ok":
                        sample_rate_ok,

                    "duration_ok":
                        duration_ok,

                    "manifest_stats_ok":
                        manifest_stats_ok,
                }
            )

        except Exception as e:

            failed.append(
                {
                    "file_id":
                        row[
                            "file_id"
                        ],

                    "error":
                        str(e),
                }
            )

            print()
            print(
                f"[ERROR] "
                f"{row['file_id']}: "
                f"{e}"
            )

        # =============================================
        # PROGRESS
        # =============================================

        if (
            (index + 1) % 25 == 0
            or
            (index + 1)
            == len(manifest)
        ):

            print(
                f"Checked "
                f"{index + 1}/"
                f"{len(manifest)}"
            )

    # =====================================================
    # QC DATAFRAME
    # =====================================================

    qc = pd.DataFrame(
        rows
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    print()
    print("=" * 70)
    print("QC SUMMARY")
    print("=" * 70)

    print()
    print("Checked files:")
    print(
        len(qc)
    )

    print()
    print("Failed files:")
    print(
        len(failed)
    )

    if not qc.empty:

        print()
        print("Sample rate:")
        print(
            qc[
                "replay_sr"
            ]
            .value_counts()
            .sort_index()
        )

        print()
        print("Sample rate OK:")
        print(
            qc[
                "sample_rate_ok"
            ]
            .value_counts()
        )

        print()
        print("Finite waveform:")
        print(
            qc[
                "finite"
            ]
            .value_counts()
        )

        print()
        print("Silent:")
        print(
            qc[
                "silent"
            ]
            .value_counts()
        )

        print()
        print("Clipped:")
        print(
            qc[
                "clipped"
            ]
            .value_counts()
        )

        print()
        print("Duration check:")
        print(
            qc[
                "duration_ok"
            ]
            .value_counts()
        )

        print()
        print("Manifest statistics match:")
        print(
            qc[
                "manifest_stats_ok"
            ]
            .value_counts()
        )

        print()
        print("Replay duration statistics:")
        print(
            qc[
                "replay_duration_s"
            ]
            .describe()
        )

        print()
        print("Duration difference statistics:")
        print(
            qc[
                "duration_diff_s"
            ]
            .describe()
        )

        print()
        print("Replay peak statistics:")
        print(
            qc[
                "replay_peak"
            ]
            .describe()
        )

        print()
        print("Replay RMS statistics:")
        print(
            qc[
                "replay_rms"
            ]
            .describe()
        )

        print()
        print("Source RMS statistics:")
        print(
            qc[
                "source_rms"
            ]
            .describe()
        )

        print()
        print("SNR distribution:")
        print(
            qc[
                "snr_db"
            ]
            .value_counts()
            .sort_index()
        )

        print()
        print("Split x RIR split:")
        print(
            pd.crosstab(
                qc[
                    "split"
                ],
                qc[
                    "rir_split"
                ],
            )
        )

        print()
        print("Label distribution:")
        print(
            qc[
                "label"
            ]
            .value_counts()
            .sort_index()
        )

    # =====================================================
    # RANDOM EXAMPLES
    # =====================================================

    if len(qc) > 0:

        n_examples = min(
            N_EXAMPLES,
            len(qc),
        )

        examples = qc.sample(
            n=n_examples,
            random_state=RANDOM_SEED,
        )

        print()
        print("=" * 70)
        print("RANDOM QC EXAMPLES")
        print("=" * 70)

        columns = [
            "file_id",
            "split",
            "label",
            "snr_db",
            "rir_room",
            "replay_peak",
            "replay_rms",
            "source_rms",
            "duration_diff_s",
        ]

        print(
            examples[
                columns
            ]
            .to_string(
                index=False
            )
        )

    # =====================================================
    # HARD VALIDATION
    # =====================================================

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    # No failed files
    assert len(failed) == 0, (
        f"Ada {len(failed)} "
        f"file gagal diperiksa."
    )

    # Same number as manifest
    assert len(qc) == len(manifest), (
        "Jumlah QC tidak sama dengan "
        "jumlah manifest."
    )

    # All replay must be 16 kHz
    assert (
        qc[
            "sample_rate_ok"
        ]
        .all()
    ), (
        "Ada replay yang bukan 16 kHz."
    )

    # No NaN / Inf
    assert (
        qc[
            "finite"
        ]
        .all()
    ), (
        "Ada waveform mengandung NaN/Inf."
    )

    # No silent output
    assert (
        ~qc[
            "silent"
        ]
    ).all(), (
        "Ada replay silent."
    )

    # No clipping
    assert (
        ~qc[
            "clipped"
        ]
    ).all(), (
        "Ada replay clipping."
    )

    # Duration must remain approximately identical
    assert (
        qc[
            "duration_ok"
        ]
        .all()
    ), (
        "Ada perbedaan durasi source/replay "
        "yang terlalu besar."
    )

    # Saved manifest values should match actual output
    assert (
        qc[
            "manifest_stats_ok"
        ]
        .all()
    ), (
        "Peak/RMS aktual tidak cocok dengan manifest."
    )

    # Train / validation must use seen
    train_val = qc[
        qc["split"].isin(
            [
                "train",
                "validation",
            ]
        )
    ]

    assert (
        train_val[
            "rir_split"
        ]
        == "seen"
    ).all(), (
        "Train/validation memakai unseen RIR."
    )

    # Test must use unseen
    test = qc[
        qc["split"]
        == "test"
    ]

    assert (
        test[
            "rir_split"
        ]
        == "unseen"
    ).all(), (
        "Test memakai seen RIR."
    )

    # =====================================================
    # FINAL
    # =====================================================

    print()
    print("VALIDATION PASSED")

    print()
    print(
        "Simulated replay pilot "
        "lolos QC otomatis."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()