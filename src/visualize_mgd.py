from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MGD_MANIFEST = (
    ROOT
    / "manifests"
    / "manifest_mgd.csv"
)

MGD_STATS = (
    ROOT
    / "data"
    / "features"
    / "mgd"
    / "train_stats.npz"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "mgd_visualization"
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
# LOAD STATS
# ============================================================

def load_stats():

    if not MGD_STATS.exists():

        raise FileNotFoundError(
            MGD_STATS
        )

    with np.load(
        MGD_STATS
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
        80,
        1,
    ):

        raise RuntimeError(
            f"Invalid mean shape: "
            f"{mean.shape}"
        )

    if std.shape != (
        80,
        1,
    ):

        raise RuntimeError(
            f"Invalid std shape: "
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
# LOAD MANIFEST
# ============================================================

def load_manifest():

    if not MGD_MANIFEST.exists():

        raise FileNotFoundError(
            MGD_MANIFEST
        )

    df = pd.read_csv(
        MGD_MANIFEST,
        low_memory=False,
    )

    if df.empty:

        raise RuntimeError(
            "MGD manifest kosong."
        )

    return df


# ============================================================
# SELECT SAMPLE
# ============================================================

def select_sample(
    df,
    condition,
):
    """
    Prioritas:
        validation
        test
        train
    """

    sub = (
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

    if sub.empty:

        raise RuntimeError(
            f"No condition={condition}"
        )

    for split in [
        "validation",
        "test",
        "train",
    ]:

        candidate = (
            sub[
                sub[
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

    return sub.iloc[
        0
    ]


# ============================================================
# LOAD MGD
# ============================================================

def load_mgd(
    row,
):

    path = (
        ROOT
        / str(
            row[
                "mgd_path"
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

        mgd = (
            data[
                "mgd"
            ]
            .astype(
                np.float32
            )
        )

    if mgd.ndim != 2:

        raise RuntimeError(
            f"Invalid MGD shape: "
            f"{mgd.shape}"
        )

    if mgd.shape[0] != 80:

        raise RuntimeError(
            f"Expected 80 Mel bins, "
            f"got {mgd.shape}"
        )

    if not np.isfinite(
        mgd
    ).all():

        raise RuntimeError(
            "MGD contains NaN/Inf."
        )

    return mgd


# ============================================================
# NORMALIZE
# ============================================================

def normalize_mgd(
    mgd,
    mean,
    std,
):

    normalized = (
        mgd
        - mean
    ) / std

    if not np.isfinite(
        normalized
    ).all():

        raise RuntimeError(
            "Normalized MGD "
            "contains NaN/Inf."
        )

    return normalized


# ============================================================
# SAVE PLOT
# ============================================================

def save_plot(
    mgd,
    condition,
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
        mgd,
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
        f"Normalized MGD\n"
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
        "MGD VISUALIZATION"
    )

    print("=" * 75)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_manifest()

    mean, std = (
        load_stats()
    )

    summary_rows = []

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

        mgd = load_mgd(
            row
        )

        normalized = (
            normalize_mgd(
                mgd,
                mean,
                std,
            )
        )

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
        ):

            source_id = (
                row[
                    "recording_id"
                ]
            )

        else:

            source_id = (
                row[
                    "source_file_id"
                ]
            )

        split = (
            row[
                "split"
            ]
        )

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
            f"{mgd.shape}"
        )

        print(
            f"Finite    : "
            f"{np.isfinite(mgd).all()}"
        )

        print(
            f"Norm min  : "
            f"{normalized.min():.6f}"
        )

        print(
            f"Norm max  : "
            f"{normalized.max():.6f}"
        )

        # ====================================================
        # SAVE IMAGE
        # ====================================================

        output_path = (
            OUTPUT_DIR
            / f"{condition}_mgd.png"
        )

        save_plot(
            mgd=normalized,

            condition=condition,

            source_id=source_id,

            split=split,

            output_path=output_path,
        )

        print()

        print(
            "Saved:"
        )

        print(
            output_path
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
                        mgd.shape[
                            1
                        ]
                    ),

                "raw_min":
                    float(
                        mgd.min()
                    ),

                "raw_max":
                    float(
                        mgd.max()
                    ),

                "norm_min":
                    float(
                        normalized.min()
                    ),

                "norm_max":
                    float(
                        normalized.max()
                    ),

                "norm_mean":
                    float(
                        normalized.mean()
                    ),

                "norm_std":
                    float(
                        normalized.std()
                    ),

                "finite":
                    bool(
                        np.isfinite(
                            normalized
                        ).all()
                    ),
            }
        )

    # ========================================================
    # SUMMARY CSV
    # ========================================================

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        OUTPUT_DIR
        / "mgd_visualization_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # DONE
    # ========================================================

    print()

    print("=" * 75)

    print(
        "MGD VISUALIZATION COMPLETE"
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
        "Generated:"
    )

    print(
        "- clean_mgd.png"
    )

    print(
        "- simulated_mgd.png"
    )

    print(
        "- real_mgd.png"
    )

    print(
        "- mgd_visualization_summary.csv"
    )

    print()

    print(
        "MGD VISUALIZATION: PASSED"
    )


if __name__ == "__main__":

    main()