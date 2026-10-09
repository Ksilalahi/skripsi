from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# INPUT MANIFESTS
# ============================================================

SPECTRAL_MANIFEST = (
    ROOT
    / "manifests"
    / "manifest_spectral_phase.csv"
)

MGD_MANIFEST = (
    ROOT
    / "manifests"
    / "manifest_mgd.csv"
)

SIM_MANIFEST = (
    ROOT
    / "manifests"
    / "manifest_replay_sim.csv"
)


# ============================================================
# TRAINING STATISTICS
# ============================================================

SPECTRAL_STATS = (
    ROOT
    / "data"
    / "features"
    / "spectral_phase"
    / "train_stats.npz"
)

MGD_STATS = (
    ROOT
    / "data"
    / "features"
    / "mgd"
    / "train_stats.npz"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_ROOT = (
    ROOT
    / "data"
    / "features"
    / "three_channel"
)

OUTPUT_MANIFEST = (
    ROOT
    / "manifests"
    / "manifest_three_channel.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_CHANNELS = 3
EXPECTED_MELS = 80


# ============================================================
# SAFE OPTIONAL VALUE
# ============================================================

def optional_value(
    row,
    column,
    default="",
):

    if column not in row.index:

        return default

    value = row[column]

    if pd.isna(
        value
    ):

        return default

    return value


# ============================================================
# LOAD NORMALIZATION STATS
# ============================================================

def load_stats():

    # ========================================================
    # SPECTRAL-PHASE STATS
    # ========================================================

    if not SPECTRAL_STATS.exists():

        raise FileNotFoundError(
            f"Spectral stats not found:\n"
            f"{SPECTRAL_STATS}"
        )

    with np.load(
        SPECTRAL_STATS
    ) as data:

        if (
            "mean"
            not in data.files
            or
            "std"
            not in data.files
        ):

            raise RuntimeError(
                "spectral train_stats.npz "
                "does not contain mean/std."
            )

        spectral_mean = (
            data[
                "mean"
            ]
            .astype(
                np.float32
            )
        )

        spectral_std = (
            data[
                "std"
            ]
            .astype(
                np.float32
            )
        )

    # ========================================================
    # SPECTRAL SHAPE QC
    # ========================================================

    if spectral_mean.shape != (
        2,
        EXPECTED_MELS,
        1,
    ):

        raise RuntimeError(
            "Invalid spectral mean shape: "
            f"{spectral_mean.shape}"
        )

    if spectral_std.shape != (
        2,
        EXPECTED_MELS,
        1,
    ):

        raise RuntimeError(
            "Invalid spectral std shape: "
            f"{spectral_std.shape}"
        )

    # ========================================================
    # MGD STATS
    # ========================================================

    if not MGD_STATS.exists():

        raise FileNotFoundError(
            f"MGD stats not found:\n"
            f"{MGD_STATS}"
        )

    with np.load(
        MGD_STATS
    ) as data:

        if (
            "mean"
            not in data.files
            or
            "std"
            not in data.files
        ):

            raise RuntimeError(
                "MGD train_stats.npz "
                "does not contain mean/std."
            )

        mgd_mean = (
            data[
                "mean"
            ]
            .astype(
                np.float32
            )
        )

        mgd_std = (
            data[
                "std"
            ]
            .astype(
                np.float32
            )
        )

    # ========================================================
    # MGD SHAPE QC
    # ========================================================

    if mgd_mean.shape != (
        EXPECTED_MELS,
        1,
    ):

        raise RuntimeError(
            "Invalid MGD mean shape: "
            f"{mgd_mean.shape}"
        )

    if mgd_std.shape != (
        EXPECTED_MELS,
        1,
    ):

        raise RuntimeError(
            "Invalid MGD std shape: "
            f"{mgd_std.shape}"
        )

    # ========================================================
    # FINITE QC
    # ========================================================

    arrays = [
        spectral_mean,
        spectral_std,
        mgd_mean,
        mgd_std,
    ]

    for array in arrays:

        if not np.isfinite(
            array
        ).all():

            raise RuntimeError(
                "Normalization statistics "
                "contain NaN/Inf."
            )

    if np.any(
        spectral_std <= 0
    ):

        raise RuntimeError(
            "Spectral std contains <= 0."
        )

    if np.any(
        mgd_std <= 0
    ):

        raise RuntimeError(
            "MGD std contains <= 0."
        )

    return (
        spectral_mean,
        spectral_std,
        mgd_mean,
        mgd_std,
    )


# ============================================================
# LOAD FEATURE MANIFESTS
# ============================================================

def load_manifests():

    if not SPECTRAL_MANIFEST.exists():

        raise FileNotFoundError(
            SPECTRAL_MANIFEST
        )

    if not MGD_MANIFEST.exists():

        raise FileNotFoundError(
            MGD_MANIFEST
        )

    spectral_df = pd.read_csv(
        SPECTRAL_MANIFEST,
        low_memory=False,
    )

    mgd_df = pd.read_csv(
        MGD_MANIFEST,
        low_memory=False,
    )

    if spectral_df.empty:

        raise RuntimeError(
            "Spectral manifest empty."
        )

    if mgd_df.empty:

        raise RuntimeError(
            "MGD manifest empty."
        )

    return (
        spectral_df,
        mgd_df,
    )


# ============================================================
# LOAD SIMULATION METADATA
# ============================================================

def load_simulation_metadata():
    """
    Mengambil metadata domain/RIR dari
    manifest_replay_sim.csv.

    Penting:
        Metadata ini harus dipertahankan hingga
        manifest final agar protokol:

            train      -> seen RIR
            validation -> seen RIR
            test       -> unseen RIR

        dapat diverifikasi.
    """

    if not SIM_MANIFEST.exists():

        raise FileNotFoundError(
            f"Simulation manifest not found:\n"
            f"{SIM_MANIFEST}"
        )

    sim_df = pd.read_csv(
        SIM_MANIFEST,
        low_memory=False,
    )

    if sim_df.empty:

        raise RuntimeError(
            "Simulation manifest empty."
        )

    required_columns = {
        "source_file_id",
        "rir_id",
        "rir_room",
        "rir_split",
        "snr_db",
        "split",
    }

    missing = (
        required_columns
        - set(
            sim_df.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Simulation manifest missing "
            "required metadata:\n"
            f"{sorted(missing)}"
        )

    # ========================================================
    # BUILD SAME FEATURE ID USED BY EXTRACTION
    # ========================================================
    #
    # Observed format:
    #
    # LA_T_9431362
    # __air_phone_lecture2_hfrp_1_1
    # __20
    #
    # ========================================================

    sim_df[
        "feature_id"
    ] = (
        sim_df[
            "source_file_id"
        ]
        .astype(str)
        .str.strip()
        +
        "__"
        +
        sim_df[
            "rir_id"
        ]
        .astype(str)
        .str.strip()
        +
        "__"
        +
        sim_df[
            "snr_db"
        ]
        .astype(str)
        .str.strip()
    )

    sim_df[
        "global_feature_id"
    ] = (
        "simulated__"
        +
        sim_df[
            "feature_id"
        ]
    )

    # ========================================================
    # UNIQUE ID QC
    # ========================================================

    if sim_df[
        "global_feature_id"
    ].duplicated().any():

        duplicate_df = (
            sim_df[
                sim_df[
                    "global_feature_id"
                ]
                .duplicated(
                    keep=False
                )
            ]
        )

        print()

        print(
            duplicate_df[
                [
                    "global_feature_id",
                    "source_file_id",
                    "rir_id",
                    "snr_db",
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Duplicate simulated "
            "global_feature_id."
        )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = (
        sim_df[
            [
                "global_feature_id",
                "rir_id",
                "rir_room",
                "rir_split",
                "snr_db",
            ]
        ]
        .copy()
    )

    return metadata


# ============================================================
# MERGE SPECTRAL AND MGD
# ============================================================

def merge_manifests(
    spectral_df,
    mgd_df,
):

    spectral_df = (
        spectral_df
        .copy()
    )

    mgd_df = (
        mgd_df
        .copy()
    )

    # ========================================================
    # REQUIRED IDS
    # ========================================================

    if (
        "global_feature_id"
        not in spectral_df.columns
    ):

        raise RuntimeError(
            "Spectral manifest does not "
            "contain global_feature_id."
        )

    if (
        "global_mgd_id"
        not in mgd_df.columns
    ):

        raise RuntimeError(
            "MGD manifest does not "
            "contain global_mgd_id."
        )

    # ========================================================
    # COMMON MATCH KEY
    # ========================================================

    spectral_df[
        "match_id"
    ] = (
        spectral_df[
            "global_feature_id"
        ]
        .astype(str)
    )

    mgd_df[
        "match_id"
    ] = (
        mgd_df[
            "global_mgd_id"
        ]
        .astype(str)
    )

    # ========================================================
    # DUPLICATE QC
    # ========================================================

    if spectral_df[
        "match_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate spectral match_id."
        )

    if mgd_df[
        "match_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate MGD match_id."
        )

    # ========================================================
    # MGD COLUMNS
    # ========================================================

    required_mgd_columns = {
        "match_id",
        "mgd_path",
        "n_frames",
    }

    missing = (
        required_mgd_columns
        - set(
            mgd_df.columns
        )
    )

    if missing:

        raise RuntimeError(
            "MGD manifest missing columns: "
            f"{sorted(missing)}"
        )

    # ========================================================
    # INNER MERGE
    # ========================================================

    merged = spectral_df.merge(
        mgd_df[
            [
                "match_id",
                "mgd_path",
                "n_frames",
            ]
        ]
        .rename(
            columns={
                "n_frames":
                    "mgd_n_frames"
            }
        ),

        on="match_id",

        how="inner",

        validate="one_to_one",
    )

    # ========================================================
    # MATCH COUNT QC
    # ========================================================

    if len(
        merged
    ) != len(
        spectral_df
    ):

        raise RuntimeError(
            "Some spectral features have no "
            "matching MGD feature. "
            f"Spectral={len(spectral_df)}, "
            f"matched={len(merged)}"
        )

    if len(
        merged
    ) != len(
        mgd_df
    ):

        raise RuntimeError(
            "Some MGD features have no "
            "matching spectral feature. "
            f"MGD={len(mgd_df)}, "
            f"matched={len(merged)}"
        )

    return merged


# ============================================================
# ATTACH SIMULATION METADATA
# ============================================================

def attach_simulation_metadata(
    dataframe,
):
    """
    Menambahkan:
        rir_id
        rir_room
        rir_split
        snr_db

    Hanya simulated rows yang memiliki
    metadata tersebut.

    Clean dan real tetap NaN.
    """

    metadata = (
        load_simulation_metadata()
    )

    df = (
        dataframe
        .copy()
    )

    # ========================================================
    # REMOVE OLD COLUMNS TO AVOID _x / _y
    # ========================================================

    for column in [
        "rir_id",
        "rir_room",
        "rir_split",
        "snr_db",
    ]:

        if column in df.columns:

            df = df.drop(
                columns=[
                    column
                ]
            )

    # ========================================================
    # MERGE
    # ========================================================

    df = df.merge(
        metadata,

        on="global_feature_id",

        how="left",

        validate="one_to_one",
    )

    # ========================================================
    # SIMULATED QC
    # ========================================================

    simulated_mask = (
        df[
            "condition"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(
            "simulated"
        )
    )

    simulated = (
        df[
            simulated_mask
        ]
        .copy()
    )

    if len(
        simulated
    ) != 150:

        raise RuntimeError(
            "Expected 150 simulated rows, "
            f"got {len(simulated)}."
        )

    # Every simulated sample must have RIR metadata.

    required_metadata = [
        "rir_id",
        "rir_room",
        "rir_split",
        "snr_db",
    ]

    for column in required_metadata:

        if simulated[
            column
        ].isna().any():

            bad_count = int(
                simulated[
                    column
                ]
                .isna()
                .sum()
            )

            raise RuntimeError(
                f"{bad_count} simulated rows "
                f"missing {column}."
            )

    # ========================================================
    # ALLOWED RIR SPLIT
    # ========================================================

    rir_split_values = set(
        simulated[
            "rir_split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )

    if rir_split_values != {
        "seen",
        "unseen",
    }:

        raise RuntimeError(
            "Unexpected rir_split values: "
            f"{sorted(rir_split_values)}"
        )

    return df


# ============================================================
# PROTOCOL QC
# ============================================================

def validate_simulation_protocol(
    df,
):
    """
    Hard QC berdasarkan protocol simulasi saat ini:

        train:
            60 simulated seen

        validation:
            45 simulated seen

        test:
            45 simulated unseen
    """

    simulated = (
        df[
            df[
                "condition"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq(
                "simulated"
            )
        ]
        .copy()
    )

    if len(
        simulated
    ) != 150:

        raise RuntimeError(
            "Simulation protocol QC expected "
            f"150 rows, got {len(simulated)}."
        )

    simulated[
        "_split"
    ] = (
        simulated[
            "split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    simulated[
        "_rir_split"
    ] = (
        simulated[
            "rir_split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    table = (
        simulated
        .groupby(
            [
                "_split",
                "_rir_split",
            ]
        )
        .size()
        .to_dict()
    )

    expected = {
        (
            "train",
            "seen",
        ):
            60,

        (
            "validation",
            "seen",
        ):
            45,

        (
            "test",
            "unseen",
        ):
            45,
    }

    # ========================================================
    # EXPECTED PAIRS
    # ========================================================

    for key, expected_count in (
        expected.items()
    ):

        actual = int(
            table.get(
                key,
                0
            )
        )

        if actual != expected_count:

            raise RuntimeError(
                f"Protocol mismatch {key}: "
                f"expected {expected_count}, "
                f"got {actual}."
            )

    # ========================================================
    # NO EXTRA PAIRS
    # ========================================================

    unexpected = {
        key:
            value

        for key, value in table.items()

        if (
            value > 0
            and
            key not in expected
        )
    }

    if unexpected:

        raise RuntimeError(
            "Unexpected split x RIR combinations: "
            f"{unexpected}"
        )

    return simulated


# ============================================================
# LOAD SPECTRAL FEATURE
# ============================================================

def load_spectral(
    path,
):

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    with np.load(
        path
    ) as data:

        if (
            "feature"
            not in data.files
        ):

            raise RuntimeError(
                f"{path} does not contain "
                "key 'feature'."
            )

        feature = (
            data[
                "feature"
            ]
            .astype(
                np.float32,
                copy=False,
            )
        )

    if feature.ndim != 3:

        raise RuntimeError(
            "Invalid spectral shape: "
            f"{feature.shape}"
        )

    if feature.shape[
        0
    ] != 2:

        raise RuntimeError(
            "Spectral feature must contain "
            "2 channels."
        )

    if feature.shape[
        1
    ] != EXPECTED_MELS:

        raise RuntimeError(
            "Spectral Mel bins != 80."
        )

    if feature.shape[
        2
    ] <= 0:

        raise RuntimeError(
            "Spectral feature has 0 frames."
        )

    if not np.isfinite(
        feature
    ).all():

        raise RuntimeError(
            "Spectral feature contains "
            "NaN/Inf."
        )

    return feature


# ============================================================
# LOAD MGD FEATURE
# ============================================================

def load_mgd(
    path,
):

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    with np.load(
        path
    ) as data:

        if (
            "mgd"
            not in data.files
        ):

            raise RuntimeError(
                f"{path} does not contain "
                "key 'mgd'."
            )

        mgd = (
            data[
                "mgd"
            ]
            .astype(
                np.float32,
                copy=False,
            )
        )

    if mgd.ndim != 2:

        raise RuntimeError(
            f"Invalid MGD shape: "
            f"{mgd.shape}"
        )

    if mgd.shape[
        0
    ] != EXPECTED_MELS:

        raise RuntimeError(
            "MGD Mel bins != 80."
        )

    if mgd.shape[
        1
    ] <= 0:

        raise RuntimeError(
            "MGD feature has 0 frames."
        )

    if not np.isfinite(
        mgd
    ).all():

        raise RuntimeError(
            "MGD contains NaN/Inf."
        )

    return mgd


# ============================================================
# NORMALIZE AND STACK
# ============================================================

def build_three_channel(
    spectral,
    mgd,
    spectral_mean,
    spectral_std,
    mgd_mean,
    mgd_std,
):

    # ========================================================
    # TIME SYNCHRONIZATION
    # ========================================================

    spectral_frames = int(
        spectral.shape[
            2
        ]
    )

    mgd_frames = int(
        mgd.shape[
            1
        ]
    )

    if (
        spectral_frames
        != mgd_frames
    ):

        raise RuntimeError(
            "Frame mismatch: "
            f"spectral={spectral_frames}, "
            f"mgd={mgd_frames}"
        )

    # ========================================================
    # NORMALIZE SPECTRAL-PHASE
    # ========================================================

    spectral_norm = (
        spectral
        - spectral_mean
    ) / spectral_std

    # ========================================================
    # NORMALIZE MGD
    # ========================================================

    mgd_norm = (
        mgd
        - mgd_mean
    ) / mgd_std

    # ========================================================
    # STACK
    # ========================================================
    #
    # channel 0 = Log-Mel
    # channel 1 = phase-derived
    # channel 2 = MGD
    #
    # ========================================================

    three_channel = np.concatenate(
        [
            spectral_norm,

            mgd_norm[
                None,
                ...,
            ],
        ],

        axis=0,
    )

    # ========================================================
    # FINAL QC
    # ========================================================

    if three_channel.ndim != 3:

        raise RuntimeError(
            "Three-channel feature is "
            "not 3-dimensional."
        )

    if three_channel.shape[
        0
    ] != EXPECTED_CHANNELS:

        raise RuntimeError(
            "Three-channel feature does "
            "not have 3 channels."
        )

    if three_channel.shape[
        1
    ] != EXPECTED_MELS:

        raise RuntimeError(
            "Three-channel Mel bins != 80."
        )

    if three_channel.shape[
        2
    ] <= 0:

        raise RuntimeError(
            "Three-channel feature has "
            "0 frames."
        )

    if not np.isfinite(
        three_channel
    ).all():

        raise RuntimeError(
            "Three-channel feature contains "
            "NaN/Inf."
        )

    return (
        three_channel
        .astype(
            np.float32,
            copy=False,
        )
    )


# ============================================================
# BUILD OUTPUT MANIFEST ROW
# ============================================================

def build_output_row(
    row,
    output_relative,
    feature,
):

    return {
        "global_feature_id":
            row[
                "global_feature_id"
            ],

        "feature_id":
            row[
                "feature_id"
            ],

        "condition":
            row[
                "condition"
            ],

        "source_file_id":
            optional_value(
                row,
                "source_file_id",
            ),

        "recording_id":
            optional_value(
                row,
                "recording_id",
            ),

        "speaker_id":
            optional_value(
                row,
                "speaker_id",
            ),

        "attack_id":
            optional_value(
                row,
                "attack_id",
            ),

        "label":
            optional_value(
                row,
                "label",
            ),

        "dataset":
            optional_value(
                row,
                "dataset",
            ),

        "split":
            optional_value(
                row,
                "split",
            ),

        "config_id":
            optional_value(
                row,
                "config_id",
            ),

        "room":
            optional_value(
                row,
                "room",
            ),

        "playback_device":
            optional_value(
                row,
                "playback_device",
            ),

        "microphone":
            optional_value(
                row,
                "microphone",
            ),

        # ====================================================
        # SIMULATION / RIR METADATA
        # ====================================================

        "rir_id":
            optional_value(
                row,
                "rir_id",
            ),

        "rir_room":
            optional_value(
                row,
                "rir_room",
            ),

        "rir_split":
            optional_value(
                row,
                "rir_split",
            ),

        "snr_db":
            optional_value(
                row,
                "snr_db",
            ),

        # ====================================================
        # FINAL FEATURE
        # ====================================================

        "feature_path":
            output_relative,

        "n_channels":
            int(
                feature.shape[
                    0
                ]
            ),

        "n_mels":
            int(
                feature.shape[
                    1
                ]
            ),

        "n_frames":
            int(
                feature.shape[
                    2
                ]
            ),

        "finite":
            bool(
                np.isfinite(
                    feature
                ).all()
            ),
    }


# ============================================================
# FINAL MANIFEST QC
# ============================================================

def validate_final_manifest(
    output_df,
):

    print()

    print("=" * 75)

    print(
        "FINAL THREE-CHANNEL MANIFEST QC"
    )

    print("=" * 75)

    if output_df.empty:

        raise RuntimeError(
            "Output manifest empty."
        )

    # ========================================================
    # EXPECTED TOTAL
    # ========================================================

    if len(
        output_df
    ) != 120814:

        raise RuntimeError(
            "Expected 120814 final features, "
            f"got {len(output_df)}."
        )

    # ========================================================
    # DUPLICATES
    # ========================================================

    if output_df[
        "global_feature_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate final "
            "global_feature_id."
        )

    # ========================================================
    # CHANNEL QC
    # ========================================================

    if not (
        output_df[
            "n_channels"
        ]
        .astype(int)
        .eq(
            EXPECTED_CHANNELS
        )
        .all()
    ):

        raise RuntimeError(
            "Some feature channel != 3."
        )

    # ========================================================
    # MEL QC
    # ========================================================

    if not (
        output_df[
            "n_mels"
        ]
        .astype(int)
        .eq(
            EXPECTED_MELS
        )
        .all()
    ):

        raise RuntimeError(
            "Some feature Mel bins != 80."
        )

    # ========================================================
    # FINITE QC
    # ========================================================

    finite_values = (
        output_df[
            "finite"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if not finite_values.eq(
        "true"
    ).all():

        raise RuntimeError(
            "Some final feature is not finite."
        )

    # ========================================================
    # SIMULATION PROTOCOL
    # ========================================================

    simulated = (
        validate_simulation_protocol(
            output_df
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print(
        f"Total final features : "
        f"{len(output_df)}"
    )

    print()

    print(
        "Condition distribution:"
    )

    print(
        output_df[
            "condition"
        ]
        .value_counts()
        .sort_index()
    )

    print()

    print(
        "Split distribution:"
    )

    print(
        output_df[
            "split"
        ]
        .value_counts()
    )

    print()

    print(
        "Simulated rir_split:"
    )

    print(
        simulated[
            "rir_split"
        ]
        .value_counts()
    )

    print()

    print(
        "Simulated split x rir_split:"
    )

    print(
        pd.crosstab(
            simulated[
                "split"
            ],
            simulated[
                "rir_split"
            ],
        )
    )

    print()

    print(
        "Channels check      : PASSED"
    )

    print(
        "Mel-bin check       : PASSED"
    )

    print(
        "Finite check        : PASSED"
    )

    print(
        "Duplicate IDs       : 0"
    )

    print(
        "RIR metadata check  : PASSED"
    )

    print(
        "Seen/unseen protocol: PASSED"
    )


# ============================================================
# METADATA-ONLY PATCH
# ============================================================

def metadata_only_patch():
    """
    Memperbaiki manifest_three_channel.csv
    TANPA membangun ulang 120814 .npz.

    Feature binary tidak berubah.

    Yang ditambahkan:
        rir_id
        rir_room
        rir_split
        snr_db
    """

    print("=" * 75)

    print(
        "THREE-CHANNEL METADATA-ONLY PATCH"
    )

    print("=" * 75)

    print()

    print(
        "Feature .npz files : UNCHANGED"
    )

    print(
        "Only manifest metadata will be updated."
    )

    # ========================================================
    # LOAD CURRENT MANIFEST
    # ========================================================

    if not OUTPUT_MANIFEST.exists():

        raise FileNotFoundError(
            "Current three-channel manifest "
            "not found:\n"
            f"{OUTPUT_MANIFEST}"
        )

    df = pd.read_csv(
        OUTPUT_MANIFEST,
        low_memory=False,
    )

    if df.empty:

        raise RuntimeError(
            "Current three-channel manifest "
            "is empty."
        )

    print()

    print(
        f"Current rows : {len(df)}"
    )

    # ========================================================
    # REMOVE EXISTING RIR COLUMNS
    # ========================================================

    for column in [
        "rir_id",
        "rir_room",
        "rir_split",
        "snr_db",
    ]:

        if column in df.columns:

            df = df.drop(
                columns=[
                    column
                ]
            )

    # ========================================================
    # LOAD SIM METADATA
    # ========================================================

    metadata = (
        load_simulation_metadata()
    )

    # ========================================================
    # PATCH
    # ========================================================

    patched = df.merge(
        metadata,

        on="global_feature_id",

        how="left",

        validate="one_to_one",
    )

    # ========================================================
    # BASIC ROW COUNT
    # ========================================================

    if len(
        patched
    ) != len(
        df
    ):

        raise RuntimeError(
            "Row count changed during "
            "metadata patch."
        )

    # ========================================================
    # SIMULATION QC
    # ========================================================

    simulated = (
        patched[
            patched[
                "condition"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq(
                "simulated"
            )
        ]
        .copy()
    )

    if len(
        simulated
    ) != 150:

        raise RuntimeError(
            "Expected 150 simulated rows, "
            f"got {len(simulated)}."
        )

    for column in [
        "rir_id",
        "rir_room",
        "rir_split",
        "snr_db",
    ]:

        missing = int(
            simulated[
                column
            ]
            .isna()
            .sum()
        )

        if missing > 0:

            raise RuntimeError(
                f"{missing} simulated rows "
                f"missing {column}."
            )

    # ========================================================
    # PROTOCOL QC
    # ========================================================

    validate_simulation_protocol(
        patched
    )

    # ========================================================
    # SAVE
    # ========================================================

    patched.to_csv(
        OUTPUT_MANIFEST,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print(
        "Simulated RIR split:"
    )

    print(
        simulated[
            "rir_split"
        ]
        .value_counts()
    )

    print()

    print(
        "split x rir_split:"
    )

    print(
        pd.crosstab(
            simulated[
                "split"
            ],
            simulated[
                "rir_split"
            ],
        )
    )

    print()

    print(
        "Updated manifest:"
    )

    print(
        OUTPUT_MANIFEST
    )

    print()

    print(
        "Three-channel .npz files : UNCHANGED"
    )

    print()

    print("=" * 75)

    print(
        "THREE-CHANNEL METADATA PATCH: PASSED"
    )

    print("=" * 75)


# ============================================================
# FULL BUILD
# ============================================================

def build_features():

    print("=" * 75)

    print(
        "BUILD THREE-CHANNEL FEATURES"
    )

    print("=" * 75)

    print()

    print(
        "Channel 0 : Log-Mel"
    )

    print(
        "Channel 1 : Phase-derived"
    )

    print(
        "Channel 2 : MGD"
    )

    print()

    print(
        "Simulation metadata:"
    )

    print(
        "rir_id, rir_room, rir_split, snr_db"
    )

    # ========================================================
    # LOAD STATS
    # ========================================================

    (
        spectral_mean,
        spectral_std,
        mgd_mean,
        mgd_std,
    ) = load_stats()

    # ========================================================
    # LOAD MANIFESTS
    # ========================================================

    (
        spectral_df,
        mgd_df,
    ) = load_manifests()

    # ========================================================
    # FEATURE MERGE
    # ========================================================

    merged = merge_manifests(
        spectral_df,
        mgd_df,
    )

    # ========================================================
    # RIR METADATA
    # ========================================================

    merged = (
        attach_simulation_metadata(
            merged
        )
    )

    # ========================================================
    # PROTOCOL QC BEFORE PROCESSING
    # ========================================================

    validate_simulation_protocol(
        merged
    )

    print()

    print(
        f"Spectral rows : "
        f"{len(spectral_df)}"
    )

    print(
        f"MGD rows      : "
        f"{len(mgd_df)}"
    )

    print(
        f"Matched rows  : "
        f"{len(merged)}"
    )

    print()

    simulated = (
        merged[
            merged[
                "condition"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq(
                "simulated"
            )
        ]
    )

    print(
        "Simulated RIR split:"
    )

    print(
        simulated[
            "rir_split"
        ]
        .value_counts()
    )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_rows = []

    errors = []

    # ========================================================
    # PROCESS
    # ========================================================

    for _, row in tqdm(
        merged.iterrows(),

        total=len(
            merged
        ),

        desc="three-channel",
    ):

        try:

            # =================================================
            # INPUT PATHS
            # =================================================

            spectral_path = (
                ROOT
                / str(
                    row[
                        "feature_path"
                    ]
                )
            )

            mgd_path = (
                ROOT
                / str(
                    row[
                        "mgd_path"
                    ]
                )
            )

            # =================================================
            # LOAD RAW FEATURES
            # =================================================

            spectral = (
                load_spectral(
                    spectral_path
                )
            )

            mgd = (
                load_mgd(
                    mgd_path
                )
            )

            # =================================================
            # NORMALIZE + STACK
            # =================================================

            feature = (
                build_three_channel(
                    spectral=
                        spectral,

                    mgd=
                        mgd,

                    spectral_mean=
                        spectral_mean,

                    spectral_std=
                        spectral_std,

                    mgd_mean=
                        mgd_mean,

                    mgd_std=
                        mgd_std,
                )
            )

            # =================================================
            # OUTPUT CONDITION
            # =================================================

            condition = (
                str(
                    row[
                        "condition"
                    ]
                )
                .strip()
                .lower()
            )

            condition_dir = (
                OUTPUT_ROOT
                / condition
            )

            condition_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # =================================================
            # FEATURE ID
            # =================================================

            feature_id = str(
                row[
                    "feature_id"
                ]
            )

            output_path = (
                condition_dir
                / f"{feature_id}.npz"
            )

            # =================================================
            # SAVE
            # =================================================

            np.savez_compressed(
                output_path,

                feature=
                    feature,
            )

            output_relative = str(
                output_path
                .relative_to(
                    ROOT
                )
            )

            # =================================================
            # MANIFEST ROW
            # =================================================

            output_rows.append(
                build_output_row(
                    row=row,

                    output_relative=
                        output_relative,

                    feature=
                        feature,
                )
            )

        except Exception as exc:

            errors.append(
                {
                    "match_id":
                        optional_value(
                            row,
                            "match_id",
                        ),

                    "global_feature_id":
                        optional_value(
                            row,
                            "global_feature_id",
                        ),

                    "error":
                        str(
                            exc
                        ),
                }
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print("=" * 75)

    print(
        "THREE-CHANNEL BUILD SUMMARY"
    )

    print("=" * 75)

    print()

    print(
        f"Successful : "
        f"{len(output_rows)}"
    )

    print(
        f"Errors     : "
        f"{len(errors)}"
    )

    # ========================================================
    # ERROR CHECK
    # ========================================================

    if errors:

        print()

        print(
            "First errors:"
        )

        for item in (
            errors[:10]
        ):

            print(
                item
            )

        raise RuntimeError(
            f"{len(errors)} "
            "three-channel integration "
            "error(s)."
        )

    # ========================================================
    # OUTPUT DATAFRAME
    # ========================================================

    output_df = pd.DataFrame(
        output_rows
    )

    # ========================================================
    # FINAL QC
    # ========================================================

    validate_final_manifest(
        output_df
    )

    # ========================================================
    # SAVE MANIFEST
    # ========================================================

    OUTPUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        OUTPUT_MANIFEST,
        index=False,
    )

    print()

    print(
        "Manifest:"
    )

    print(
        OUTPUT_MANIFEST
    )

    print()

    print("=" * 75)

    print(
        "THREE-CHANNEL INTEGRATION PASSED"
    )

    print("=" * 75)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build normalized three-channel "
            "Log-Mel + Phase + MGD features "
            "and preserve simulation RIR metadata."
        )
    )

    parser.add_argument(
        "--metadata-only",

        action="store_true",

        help=(
            "Patch RIR metadata into the existing "
            "manifest_three_channel.csv without "
            "rebuilding .npz feature files."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # METADATA-ONLY
    # ========================================================

    if args.metadata_only:

        metadata_only_patch()

        return

    # ========================================================
    # FULL BUILD
    # ========================================================

    build_features()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()