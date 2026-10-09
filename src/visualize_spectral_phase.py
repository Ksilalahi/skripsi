from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_MANIFEST = (
    ROOT
    / "manifests"
    / "manifest_spectral_phase.csv"
)

TRAIN_STATS = (
    ROOT
    / "data"
    / "features"
    / "spectral_phase"
    / "train_stats.npz"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "spectral_phase_visualization"
)


# ============================================================
# CONDITIONS
# ============================================================

CONDITIONS = [
    "clean",
    "simulated",
    "real",
]


# ============================================================
# LOAD TRAINING STATS
# ============================================================

def load_training_stats():

    if not TRAIN_STATS.exists():

        raise FileNotFoundError(
            f"Training statistics tidak ditemukan:\n"
            f"{TRAIN_STATS}"
        )

    with np.load(
        TRAIN_STATS
    ) as data:

        mean = (
            data[
                "mean"
            ]
            .astype(
                np.float32
            )
        )

        std = (
            data[
                "std"
            ]
            .astype(
                np.float32
            )
        )

    if mean.shape != (
        2,
        80,
        1,
    ):

        raise RuntimeError(
            f"Mean shape invalid: "
            f"{mean.shape}"
        )

    if std.shape != (
        2,
        80,
        1,
    ):

        raise RuntimeError(
            f"Std shape invalid: "
            f"{std.shape}"
        )

    if not np.isfinite(
        mean
    ).all():

        raise RuntimeError(
            "Mean contains NaN/Inf."
        )

    if not np.isfinite(
        std
    ).all():

        raise RuntimeError(
            "Std contains NaN/Inf."
        )

    if np.any(
        std <= 0
    ):

        raise RuntimeError(
            "Std contains <= 0."
        )

    return mean, std


# ============================================================
# LOAD FEATURE MANIFEST
# ============================================================

def load_manifest():

    if not FEATURE_MANIFEST.exists():

        raise FileNotFoundError(
            f"Feature manifest tidak ditemukan:\n"
            f"{FEATURE_MANIFEST}"
        )

    df = pd.read_csv(
        FEATURE_MANIFEST,
        low_memory=False,
    )

    if df.empty:

        raise RuntimeError(
            "Feature manifest kosong."
        )

    return df


# ============================================================
# SELECT REPRESENTATIVE SAMPLE
# ============================================================

def select_sample(
    df,
    condition,
):
    """
    Memilih satu sample representatif.

    Priority:
        validation
        test
        train

    Tujuannya hanya visualisasi,
    bukan training.
    """

    condition_df = (
        df[
            df[
                "condition"
            ]
            .astype(str)
            .eq(
                condition
            )
        ]
        .copy()
    )

    if condition_df.empty:

        raise RuntimeError(
            f"Tidak ada condition="
            f"{condition}"
        )

    for split in [
        "validation",
        "test",
        "train",
    ]:

        candidate = (
            condition_df[
                condition_df[
                    "split"
                ]
                .astype(str)
                .str.lower()
                .eq(
                    split
                )
            ]
        )

        if not candidate.empty:

            return candidate.iloc[
                0
            ]

    return condition_df.iloc[
        0
    ]


# ============================================================
# LOAD ONE FEATURE
# ============================================================

def load_feature(
    row,
):

    path = (
        ROOT
        / str(
            row[
                "feature_path"
            ]
        )
    )

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    with np.load(
        path
    ) as data:

        feature = (
            data[
                "feature"
            ]
            .astype(
                np.float32
            )
        )

    if feature.ndim != 3:

        raise RuntimeError(
            f"Invalid feature shape: "
            f"{feature.shape}"
        )

    if feature.shape[0] != 2:

        raise RuntimeError(
            f"Expected 2 channels, "
            f"got {feature.shape}"
        )

    if feature.shape[1] != 80:

        raise RuntimeError(
            f"Expected 80 Mel bins, "
            f"got {feature.shape}"
        )

    if not np.isfinite(
        feature
    ).all():

        raise RuntimeError(
            "Feature contains NaN/Inf."
        )

    return feature


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_feature(
    feature,
    mean,
    std,
):

    normalized = (
        feature
        - mean
    ) / std

    if not np.isfinite(
        normalized
    ).all():

        raise RuntimeError(
            "Normalized feature "
            "contains NaN/Inf."
        )

    return normalized


# ============================================================
# PLOT ONE CHANNEL
# ============================================================

def save_plot(
    data,
    condition,
    channel_name,
    source_id,
    split,
    output_path,
):

    plt.figure(
        figsize=(
            12,
            5,
        )
    )

    plt.imshow(
        data,
        origin="lower",
        aspect="auto",
    )

    plt.colorbar()

    plt.xlabel(
        "Time frame"
    )

    plt.ylabel(
        "Mel bin"
    )

    plt.title(
        f"{condition.upper()} | "
        f"{channel_name}\n"
        f"ID={source_id} | "
        f"split={split}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "SPECTRAL-PHASE VISUALIZATION"
    )
    print("=" * 75)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD
    # ========================================================

    df = load_manifest()

    mean, std = (
        load_training_stats()
    )

    summary_rows = []

    # ========================================================
    # CONDITIONS
    # ========================================================

    for condition in CONDITIONS:

        print()
        print("=" * 75)
        print(
            f"CONDITION: "
            f"{condition.upper()}"
        )
        print("=" * 75)

        row = select_sample(
            df,
            condition,
        )

        feature = load_feature(
            row
        )

        normalized = normalize_feature(
            feature,
            mean,
            std,
        )

        source_id = (
            row[
                "recording_id"
            ]
            if (
                "recording_id"
                in row.index
                and
                pd.notna(
                    row[
                        "recording_id"
                    ]
                )
                and
                str(
                    row[
                        "recording_id"
                    ]
                ).strip()
                != ""
            )
            else row[
                "source_file_id"
            ]
        )

        split = row[
            "split"
        ]

        print()
        print(
            f"Source ID : "
            f"{source_id}"
        )

        print(
            f"Split     : "
            f"{split}"
        )

        print(
            f"Shape     : "
            f"{feature.shape}"
        )

        print(
            f"Finite    : "
            f"{np.isfinite(feature).all()}"
        )

        # ====================================================
        # LOG-MEL
        # ====================================================

        logmel_path = (
            OUTPUT_DIR
            / (
                f"{condition}"
                "_logmel.png"
            )
        )

        save_plot(
            data=normalized[
                0
            ],

            condition=condition,

            channel_name=(
                "Normalized Log-Mel"
            ),

            source_id=source_id,

            split=split,

            output_path=logmel_path,
        )

        # ====================================================
        # PHASE
        # ====================================================

        phase_path = (
            OUTPUT_DIR
            / (
                f"{condition}"
                "_phase_derived.png"
            )
        )

        save_plot(
            data=normalized[
                1
            ],

            condition=condition,

            channel_name=(
                "Normalized Phase-derived"
            ),

            source_id=source_id,

            split=split,

            output_path=phase_path,
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        summary_rows.append(
            {
                "condition":
                    condition,

                "source_id":
                    source_id,

                "split":
                    split,

                "frames":
                    int(
                        feature.shape[2]
                    ),

                "logmel_min":
                    float(
                        normalized[
                            0
                        ].min()
                    ),

                "logmel_max":
                    float(
                        normalized[
                            0
                        ].max()
                    ),

                "phase_min":
                    float(
                        normalized[
                            1
                        ].min()
                    ),

                "phase_max":
                    float(
                        normalized[
                            1
                        ].max()
                    ),

                "finite":
                    bool(
                        np.isfinite(
                            normalized
                        ).all()
                    ),
            }
        )

        print()
        print(
            f"Saved Log-Mel:"
        )

        print(
            logmel_path
        )

        print()
        print(
            f"Saved Phase-derived:"
        )

        print(
            phase_path
        )

    # ========================================================
    # SUMMARY CSV
    # ========================================================

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        OUTPUT_DIR
        / "visualization_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 75)
    print(
        "VISUALIZATION COMPLETE"
    )
    print("=" * 75)

    print()
    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Generated:"
    )

    print(
        "- clean_logmel.png"
    )

    print(
        "- clean_phase_derived.png"
    )

    print(
        "- simulated_logmel.png"
    )

    print(
        "- simulated_phase_derived.png"
    )

    print(
        "- real_logmel.png"
    )

    print(
        "- real_phase_derived.png"
    )

    print(
        "- visualization_summary.csv"
    )

    print()
    print(
        "SPECTRAL-PHASE VISUALIZATION: PASSED"
    )


if __name__ == "__main__":

    main()