from pathlib import Path
from math import gcd
import argparse
import re

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from scipy.signal import resample_poly
from tqdm import tqdm

from spectral_phase import SpectralPhase


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_DIR = (
    ROOT
    / "manifests"
)

SOURCE_MANIFEST = (
    MANIFEST_DIR
    / "source_manifest.csv"
)

SIM_MANIFEST = (
    MANIFEST_DIR
    / "manifest_replay_sim.csv"
)

REAL_MANIFEST = (
    MANIFEST_DIR
    / "manifest_real_replay.csv"
)


# ============================================================
# MGD OUTPUT
# ============================================================

MGD_ROOT = (
    ROOT
    / "data"
    / "features"
    / "mgd"
)

MGD_RAW_ROOT = (
    MGD_ROOT
    / "raw"
)

MGD_MANIFEST = (
    MANIFEST_DIR
    / "manifest_mgd.csv"
)

MGD_PILOT_MANIFEST = (
    MANIFEST_DIR
    / "manifest_mgd_pilot.csv"
)

MGD_PILOT_QC = (
    ROOT
    / "results"
    / "mgd"
    / "mgd_pilot_qc.csv"
)

MGD_TRAIN_STATS = (
    MGD_ROOT
    / "train_stats.npz"
)


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

TARGET_SR = 16000

N_FFT = 512

HOP_LENGTH = 160

N_MELS = 80

ALPHA = 0.4

GAMMA = 0.9

EPS = 1e-12


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MANIFEST CONFIG
# ============================================================

MANIFEST_CONFIG = {

    "clean": {

        "path":
            SOURCE_MANIFEST,

        "audio_columns": [
            "file_path",
            "audio_path",
            "source_file_path",
        ],
    },

    "simulated": {

        "path":
            SIM_MANIFEST,

        "audio_columns": [
            "model_file_path",
            "replay_file_path",
            "output_file_path",
            "file_path",
            "audio_path",
        ],
    },

    "real": {

        "path":
            REAL_MANIFEST,

        "audio_columns": [
            "model_file_path",
            "master_file_path",
        ],
    },
}


# ============================================================
# SAFE NAME
# ============================================================

def safe_name(
    value,
):

    value = str(
        value
    )

    value = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        value,
    )

    return value


# ============================================================
# OPTIONAL VALUE
# ============================================================

def optional_value(
    row,
    column,
    default="",
):

    if column not in row.index:

        return default

    value = row[
        column
    ]

    if pd.isna(
        value
    ):

        return default

    return value


# ============================================================
# RESOLVE AUDIO PATH
# ============================================================

def resolve_audio_path(
    value,
):

    path = Path(
        str(value)
    )

    if path.is_absolute():

        return path

    return (
        ROOT
        / path
    ).resolve()


# ============================================================
# FIND AUDIO COLUMN
# ============================================================

def find_audio_column(
    dataframe,
    candidates,
):

    for column in candidates:

        if column in dataframe.columns:

            return column

    raise RuntimeError(
        "\nTidak ditemukan kolom audio.\n"
        f"Candidates : {candidates}\n"
        f"Available  : {list(dataframe.columns)}"
    )


# ============================================================
# LOAD AUDIO READ-ONLY
# ============================================================

def load_audio_read_only(
    path,
    target_sr=16000,
):
    """
    Membaca audio ke RAM.

    Tidak pernah menulis ulang audio asli.

    Jika sample rate berbeda:
        resampling dilakukan hanya di RAM.
    """

    audio, sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )

    if audio.size == 0:

        raise RuntimeError(
            f"Audio kosong: {path}"
        )

    if not np.isfinite(
        audio
    ).all():

        raise RuntimeError(
            f"Audio contains NaN/Inf: "
            f"{path}"
        )

    # ========================================================
    # MONO IN RAM
    # ========================================================

    audio = audio.mean(
        axis=1
    )

    # ========================================================
    # RESAMPLE IN RAM
    # ========================================================

    if int(sr) != int(
        target_sr
    ):

        common = gcd(
            int(sr),
            int(target_sr),
        )

        up = (
            int(target_sr)
            // common
        )

        down = (
            int(sr)
            // common
        )

        audio = resample_poly(
            audio,
            up,
            down,
        )

        audio = audio.astype(
            np.float32,
            copy=False,
        )

        sr = target_sr

    if not np.isfinite(
        audio
    ).all():

        raise RuntimeError(
            "Audio becomes NaN/Inf "
            f"after preprocessing: {path}"
        )

    return (
        audio.astype(
            np.float32,
            copy=False,
        ),
        int(sr),
    )


# ============================================================
# FEATURE ID
# ============================================================

def build_feature_id(
    row,
    condition,
    row_index,
):
    """
    Mengikuti pola ID Tahap 4 agar
    clean / simulated / real tetap dapat
    ditelusuri ke audio asal.
    """

    # ========================================================
    # REAL REPLAY
    # ========================================================

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

        return safe_name(
            row[
                "recording_id"
            ]
        )

    # ========================================================
    # SOURCE ID
    # ========================================================

    if (
        "source_file_id"
        in row.index
        and
        pd.notna(
            row[
                "source_file_id"
            ]
        )
    ):

        source_id = safe_name(
            row[
                "source_file_id"
            ]
        )

        # ====================================================
        # SIMULATED
        # ====================================================

        if condition == "simulated":

            extra_parts = []

            possible_fields = [
                "rir_id",
                "rir_file",
                "snr_db",
                "simulation_id",
                "replay_id",
            ]

            for field in possible_fields:

                if (
                    field
                    in row.index
                    and
                    pd.notna(
                        row[
                            field
                        ]
                    )
                ):

                    extra_parts.append(
                        safe_name(
                            row[
                                field
                            ]
                        )
                    )

            if extra_parts:

                return (
                    source_id
                    + "__"
                    + "__".join(
                        extra_parts
                    )
                )

            return (
                f"{source_id}"
                f"__simrow{row_index}"
            )

        return source_id

    # ========================================================
    # FALLBACK
    # ========================================================

    return (
        f"{condition}"
        f"__row_{row_index:07d}"
    )


# ============================================================
# VALIDATE MGD
# ============================================================

def validate_mgd(
    mgd,
    feature_id,
):
    """
    Expected:
        [80,T]
    """

    if mgd.ndim != 2:

        raise RuntimeError(
            f"{feature_id}: "
            f"MGD must be 2D [80,T], "
            f"got {mgd.shape}"
        )

    if (
        mgd.shape[0]
        != N_MELS
    ):

        raise RuntimeError(
            f"{feature_id}: "
            f"expected {N_MELS} Mel bins, "
            f"got {mgd.shape}"
        )

    if (
        mgd.shape[1]
        <= 0
    ):

        raise RuntimeError(
            f"{feature_id}: "
            "MGD contains 0 frames."
        )

    if not np.isfinite(
        mgd
    ).all():

        raise RuntimeError(
            f"{feature_id}: "
            "MGD contains NaN/Inf."
        )

    if np.any(
        mgd < 0
    ):

        raise RuntimeError(
            f"{feature_id}: "
            "absolute MGD contains "
            "negative values."
        )


# ============================================================
# SAVE MGD
# ============================================================

def save_mgd(
    mgd,
    destination,
):

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        destination,

        mgd=mgd.astype(
            np.float32,
            copy=False,
        ),
    )


# ============================================================
# MGD DISTRIBUTION STATISTICS
# ============================================================

def compute_mgd_statistics(
    mgd,
):
    """
    Statistik diagnostik per file.

    Metric tambahan:
        max_to_p99_9
        top_0_1pct_energy_ratio

    adalah engineering sanity check,
    bukan threshold ilmiah dari panduan.
    """

    values = (
        mgd
        .astype(
            np.float64,
            copy=False,
        )
        .reshape(
            -1
        )
    )

    p50 = float(
        np.percentile(
            values,
            50.0,
        )
    )

    p95 = float(
        np.percentile(
            values,
            95.0,
        )
    )

    p99 = float(
        np.percentile(
            values,
            99.0,
        )
    )

    p999 = float(
        np.percentile(
            values,
            99.9,
        )
    )

    maximum = float(
        values.max()
    )

    mean = float(
        values.mean()
    )

    std = float(
        values.std()
    )

    max_to_p999 = (
        maximum
        /
        max(
            p999,
            EPS,
        )
    )

    squared = (
        values
        * values
    )

    total_energy = float(
        squared.sum()
    )

    top_mask = (
        values
        >= p999
    )

    top_energy = float(
        squared[
            top_mask
        ].sum()
    )

    if total_energy > 0:

        top_0_1pct_energy_ratio = (
            top_energy
            /
            total_energy
        )

    else:

        top_0_1pct_energy_ratio = 0.0

    return {
        "min":
            float(
                values.min()
            ),

        "max":
            maximum,

        "mean":
            mean,

        "std":
            std,

        "p50":
            p50,

        "p95":
            p95,

        "p99":
            p99,

        "p99_9":
            p999,

        "max_to_p99_9":
            float(
                max_to_p999
            ),

        "top_0_1pct_energy_ratio":
            float(
                top_0_1pct_energy_ratio
            ),
    }


# ============================================================
# PROCESS CONDITION
# ============================================================

def process_condition(
    condition,
    extractor,
    limit=None,
    overwrite=False,
):

    config = (
        MANIFEST_CONFIG[
            condition
        ]
    )

    manifest_path = (
        config[
            "path"
        ]
    )

    # ========================================================
    # CHECK MANIFEST
    # ========================================================

    if not manifest_path.exists():

        raise FileNotFoundError(
            f"Manifest tidak ditemukan:\n"
            f"{manifest_path}"
        )

    df = pd.read_csv(
        manifest_path,
        low_memory=False,
    )

    if df.empty:

        raise RuntimeError(
            f"{manifest_path.name} kosong."
        )

    audio_column = (
        find_audio_column(
            df,
            config[
                "audio_columns"
            ],
        )
    )

    # ========================================================
    # PILOT LIMIT
    # ========================================================

    if limit is not None:

        df = df.head(
            limit
        )

    # ========================================================
    # INFO
    # ========================================================

    print()
    print("=" * 75)

    print(
        f"PROCESSING MGD CONDITION: "
        f"{condition.upper()}"
    )

    print("=" * 75)

    print()

    print(
        f"Manifest      : "
        f"{manifest_path.name}"
    )

    print(
        f"Manifest rows : "
        f"{len(df)}"
    )

    print(
        f"Audio column  : "
        f"{audio_column}"
    )

    print(
        f"Target SR     : "
        f"{TARGET_SR}"
    )

    print(
        f"Alpha         : "
        f"{ALPHA}"
    )

    print(
        f"Gamma         : "
        f"{GAMMA}"
    )

    print(
        f"Device        : "
        f"{DEVICE}"
    )

    if limit is not None:

        print(
            f"Pilot limit   : "
            f"{limit}"
        )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    condition_dir = (
        MGD_RAW_ROOT
        / condition
    )

    condition_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    successful_rows = []

    qc_rows = []

    errors = []

    # ========================================================
    # LOOP
    # ========================================================

    for row_index, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"mgd-{condition}",
    ):

        try:

            # =================================================
            # AUDIO
            # =================================================

            audio_path = (
                resolve_audio_path(
                    row[
                        audio_column
                    ]
                )
            )

            if not audio_path.exists():

                raise FileNotFoundError(
                    f"Audio missing: "
                    f"{audio_path}"
                )

            # =================================================
            # ID
            # =================================================

            feature_id = (
                build_feature_id(
                    row=row,
                    condition=condition,
                    row_index=row_index,
                )
            )

            output_path = (
                condition_dir
                / f"{feature_id}.npz"
            )

            # =================================================
            # LOAD EXISTING
            # =================================================

            if (
                output_path.exists()
                and
                not overwrite
            ):

                with np.load(
                    output_path
                ) as data:

                    if "mgd" not in data.files:

                        raise RuntimeError(
                            "Existing NPZ does not "
                            "contain key 'mgd'."
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

                validate_mgd(
                    mgd,
                    feature_id,
                )

            # =================================================
            # COMPUTE
            # =================================================

            else:

                waveform, sr = (
                    load_audio_read_only(
                        audio_path,
                        target_sr=TARGET_SR,
                    )
                )

                if sr != TARGET_SR:

                    raise RuntimeError(
                        f"Unexpected SR: {sr}"
                    )

                wav_tensor = (
                    torch
                    .from_numpy(
                        waveform
                    )
                    .float()
                    .unsqueeze(
                        0
                    )
                    .unsqueeze(
                        0
                    )
                    .to(
                        DEVICE
                    )
                )

                # =============================================
                # MGD
                # =============================================

                with torch.no_grad():

                    mgd_tensor = (
                        extractor
                        .modified_group_delay(
                            wav_tensor,

                            alpha=ALPHA,

                            gamma=GAMMA,
                        )
                    )

                # Expected:
                # [1,80,T]

                if (
                    mgd_tensor.ndim
                    != 3
                ):

                    raise RuntimeError(
                        "MGD extractor output "
                        "is not 3D."
                    )

                mgd = (
                    mgd_tensor[
                        0
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float32,
                        copy=False,
                    )
                )

                validate_mgd(
                    mgd,
                    feature_id,
                )

                save_mgd(
                    mgd,
                    output_path,
                )

            # =================================================
            # QC STATS
            # =================================================

            stats = (
                compute_mgd_statistics(
                    mgd
                )
            )

            # =================================================
            # RELATIVE PATH
            # =================================================

            try:

                audio_relative = str(
                    audio_path.relative_to(
                        ROOT
                    )
                )

            except ValueError:

                audio_relative = str(
                    audio_path
                )

            output_relative = str(
                output_path.relative_to(
                    ROOT
                )
            )

            # =================================================
            # MANIFEST ROW
            # =================================================

            successful_rows.append(
                {
                    "global_mgd_id":
                        (
                            f"{condition}"
                            f"__{feature_id}"
                        ),

                    "feature_id":
                        feature_id,

                    "condition":
                        condition,

                    "source_manifest":
                        manifest_path.name,

                    "source_row_index":
                        int(
                            row_index
                        ),

                    "audio_path":
                        audio_relative,

                    "mgd_path":
                        output_relative,

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

                    "sample_rate":
                        TARGET_SR,

                    "n_fft":
                        N_FFT,

                    "hop_length":
                        HOP_LENGTH,

                    "n_mels":
                        N_MELS,

                    "alpha":
                        ALPHA,

                    "gamma":
                        GAMMA,

                    "n_frames":
                        int(
                            mgd.shape[
                                1
                            ]
                        ),

                    "finite":
                        bool(
                            np.isfinite(
                                mgd
                            ).all()
                        ),
                }
            )

            # =================================================
            # QC ROW
            # =================================================

            qc_rows.append(
                {
                    "condition":
                        condition,

                    "feature_id":
                        feature_id,

                    "split":
                        optional_value(
                            row,
                            "split",
                        ),

                    **stats,
                }
            )

        except Exception as exc:

            errors.append(
                {
                    "row_index":
                        int(
                            row_index
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
        f"{condition.upper()} "
        "MGD EXTRACTION SUMMARY"
    )

    print("=" * 75)

    print()

    print(
        f"Successful : "
        f"{len(successful_rows)}"
    )

    print(
        f"Errors     : "
        f"{len(errors)}"
    )

    if errors:

        print()

        print(
            "First errors:"
        )

        for item in (
            errors[:10]
        ):

            print(
                f"[row "
                f"{item['row_index']}] "
                f"{item['error']}"
            )

        raise RuntimeError(
            f"{condition}: "
            f"{len(errors)} MGD "
            "extraction error(s)."
        )

    return (
        pd.DataFrame(
            successful_rows
        ),

        pd.DataFrame(
            qc_rows
        ),
    )


# ============================================================
# DISTRIBUTION QC SUMMARY
# ============================================================

def summarize_qc(
    qc_df,
):

    print()

    print("=" * 75)

    print(
        "MGD DISTRIBUTION QC"
    )

    print("=" * 75)

    if qc_df.empty:

        raise RuntimeError(
            "QC dataframe kosong."
        )

    numeric_columns = [
        "min",
        "max",
        "mean",
        "std",
        "p50",
        "p95",
        "p99",
        "p99_9",
        "max_to_p99_9",
        "top_0_1pct_energy_ratio",
    ]

    if not np.isfinite(
        qc_df[
            numeric_columns
        ]
        .to_numpy(
            dtype=np.float64
        )
    ).all():

        raise RuntimeError(
            "QC statistics contain NaN/Inf."
        )

    for condition in (
        qc_df[
            "condition"
        ]
        .drop_duplicates()
        .tolist()
    ):

        sub = (
            qc_df[
                qc_df[
                    "condition"
                ]
                == condition
            ]
        )

        print()

        print(
            f"Condition: "
            f"{condition.upper()}"
        )

        print(
            f"Files     : "
            f"{len(sub)}"
        )

        print(
            "Median of file means : "
            f"{sub['mean'].median():.6f}"
        )

        print(
            "Median of file stds  : "
            f"{sub['std'].median():.6f}"
        )

        print(
            "Median p50           : "
            f"{sub['p50'].median():.6f}"
        )

        print(
            "Median p95           : "
            f"{sub['p95'].median():.6f}"
        )

        print(
            "Median p99           : "
            f"{sub['p99'].median():.6f}"
        )

        print(
            "Median p99.9         : "
            f"{sub['p99_9'].median():.6f}"
        )

        print(
            "Largest max          : "
            f"{sub['max'].max():.6f}"
        )

        print(
            "Largest max/p99.9    : "
            f"{sub['max_to_p99_9'].max():.6f}"
        )

        print(
            "Largest top0.1% "
            "energy ratio         : "
            f"{sub['top_0_1pct_energy_ratio'].max():.6f}"
        )

    print()

    print(
        "Finite QC statistics : PASSED"
    )

    print(
        "NaN/Inf              : NONE"
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "max/p99.9 and top-0.1% energy "
        "ratio are diagnostic metrics, "
        "not literature-derived acceptance "
        "thresholds."
    )


# ============================================================
# SAVE FINAL MANIFEST
# ============================================================

def save_final_manifest(
    new_df,
    conditions,
):

    if MGD_MANIFEST.exists():

        old_df = pd.read_csv(
            MGD_MANIFEST,
            low_memory=False,
        )

        old_df = (
            old_df[
                ~old_df[
                    "condition"
                ].isin(
                    conditions
                )
            ]
            .copy()
        )

        combined = pd.concat(
            [
                old_df,
                new_df,
            ],
            ignore_index=True,
        )

    else:

        combined = (
            new_df.copy()
        )

    if (
        combined[
            "global_mgd_id"
        ]
        .duplicated()
        .any()
    ):

        duplicates = (
            combined[
                combined[
                    "global_mgd_id"
                ]
                .duplicated(
                    keep=False
                )
            ]
        )

        print()

        print(
            duplicates[
                [
                    "global_mgd_id",
                    "condition",
                    "audio_path",
                ]
            ].to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Duplicate global_mgd_id."
        )

    MGD_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        MGD_MANIFEST,
        index=False,
    )

    return combined


# ============================================================
# MANIFEST QC
# ============================================================

def validate_manifest(
    df,
):

    print()

    print("=" * 75)

    print(
        "MGD MANIFEST QC"
    )

    print("=" * 75)

    if df.empty:

        raise RuntimeError(
            "MGD manifest kosong."
        )

    if (
        df[
            "global_mgd_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate MGD ID."
        )

    if not (
        df[
            "n_mels"
        ]
        .astype(int)
        .eq(
            N_MELS
        )
        .all()
    ):

        raise RuntimeError(
            "MGD Mel-bin QC failed."
        )

    finite_values = (
        df[
            "finite"
        ]
        .astype(str)
        .str.lower()
    )

    if not finite_values.eq(
        "true"
    ).all():

        raise RuntimeError(
            "MGD finite QC failed."
        )

    missing = []

    for value in (
        df[
            "mgd_path"
        ]
    ):

        path = (
            ROOT
            / str(value)
        )

        if not path.exists():

            missing.append(
                path
            )

    if missing:

        raise RuntimeError(
            f"{len(missing)} "
            "MGD files missing."
        )

    print()

    print(
        f"Total MGD features : "
        f"{len(df)}"
    )

    print()

    print(
        "Condition distribution:"
    )

    print(
        df[
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
        df[
            "split"
        ]
        .value_counts(
            dropna=False
        )
    )

    print()

    print(
        "Mel-bin check  : PASSED"
    )

    print(
        "Finite check   : PASSED"
    )

    print(
        "Missing files  : 0"
    )

    print(
        "Duplicate IDs  : 0"
    )


# ============================================================
# TRAINING-ONLY MGD NORMALIZATION STATISTICS
# ============================================================

def compute_training_statistics():
    """
    Menghitung mean/std hanya dari split=train.

    Input per file:
        [80,T]

    Output:
        mean [80,1]
        std  [80,1]

    Validation dan test tidak digunakan.
    """

    print()

    print("=" * 75)

    print(
        "MGD TRAINING-ONLY "
        "NORMALIZATION STATISTICS"
    )

    print("=" * 75)

    # ========================================================
    # MANIFEST CHECK
    # ========================================================

    if not MGD_MANIFEST.exists():

        raise FileNotFoundError(
            "manifest_mgd.csv tidak ditemukan:\n"
            f"{MGD_MANIFEST}"
        )

    df = pd.read_csv(
        MGD_MANIFEST,
        low_memory=False,
    )

    if df.empty:

        raise RuntimeError(
            "manifest_mgd.csv kosong."
        )

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required_columns = {
        "mgd_path",
        "split",
        "condition",
        "n_mels",
    }

    missing_columns = (
        required_columns
        - set(
            df.columns
        )
    )

    if missing_columns:

        raise RuntimeError(
            "MGD manifest kehilangan "
            "kolom:\n"
            f"{sorted(missing_columns)}"
        )

    # ========================================================
    # FILTER TRAIN
    # ========================================================

    normalized_split = (
        df[
            "split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    train_df = (
        df[
            normalized_split.eq(
                "train"
            )
        ]
        .copy()
    )

    if train_df.empty:

        raise RuntimeError(
            "Tidak ditemukan "
            "split=train."
        )

    # ========================================================
    # INFO
    # ========================================================

    print()

    print(
        f"All MGD features : "
        f"{len(df)}"
    )

    print(
        f"Training files   : "
        f"{len(train_df)}"
    )

    print()

    print(
        "Training condition distribution:"
    )

    print(
        train_df[
            "condition"
        ]
        .value_counts()
        .sort_index()
    )

    print()

    print(
        "Validation/test are excluded "
        "from normalization statistics."
    )

    # ========================================================
    # RUNNING ACCUMULATORS
    # ========================================================

    running_sum = np.zeros(
        (
            N_MELS,
        ),
        dtype=np.float64,
    )

    running_squared_sum = np.zeros(
        (
            N_MELS,
        ),
        dtype=np.float64,
    )

    total_frames = 0

    processed_files = 0

    errors = []

    # ========================================================
    # LOOP TRAIN FILES
    # ========================================================

    for _, row in tqdm(
        train_df.iterrows(),
        total=len(train_df),
        desc="mgd train statistics",
    ):

        try:

            mgd_path = (
                ROOT
                / str(
                    row[
                        "mgd_path"
                    ]
                )
            )

            if not mgd_path.exists():

                raise FileNotFoundError(
                    f"Missing MGD: "
                    f"{mgd_path}"
                )

            with np.load(
                mgd_path
            ) as data:

                if (
                    "mgd"
                    not in data.files
                ):

                    raise RuntimeError(
                        "NPZ does not contain "
                        "key 'mgd'."
                    )

                mgd = (
                    data[
                        "mgd"
                    ]
                    .astype(
                        np.float64,
                        copy=False,
                    )
                )

            # =================================================
            # SHAPE
            # =================================================

            if mgd.ndim != 2:

                raise RuntimeError(
                    f"Invalid MGD shape: "
                    f"{mgd.shape}"
                )

            if (
                mgd.shape[0]
                != N_MELS
            ):

                raise RuntimeError(
                    f"Expected {N_MELS} Mel bins, "
                    f"got {mgd.shape}"
                )

            if (
                mgd.shape[1]
                <= 0
            ):

                raise RuntimeError(
                    "MGD has zero frames."
                )

            # =================================================
            # FINITE
            # =================================================

            if not np.isfinite(
                mgd
            ).all():

                raise RuntimeError(
                    "MGD contains NaN/Inf."
                )

            # =================================================
            # ACCUMULATE OVER TIME
            # =================================================

            running_sum += (
                mgd.sum(
                    axis=1
                )
            )

            running_squared_sum += (
                np.square(
                    mgd
                ).sum(
                    axis=1
                )
            )

            total_frames += int(
                mgd.shape[
                    1
                ]
            )

            processed_files += 1

        except Exception as exc:

            errors.append(
                {
                    "mgd_path":
                        str(
                            row[
                                "mgd_path"
                            ]
                        ),

                    "error":
                        str(
                            exc
                        ),
                }
            )

    # ========================================================
    # ERROR CHECK
    # ========================================================

    if errors:

        print()

        print(
            "MGD statistics errors:"
        )

        for item in (
            errors[:10]
        ):

            print(
                f"{item['mgd_path']} "
                f"-> "
                f"{item['error']}"
            )

        raise RuntimeError(
            f"{len(errors)} MGD feature "
            "gagal saat menghitung statistics."
        )

    if (
        processed_files
        != len(
            train_df
        )
    ):

        raise RuntimeError(
            "Jumlah processed training "
            "files tidak sesuai."
        )

    if total_frames <= 0:

        raise RuntimeError(
            "total_frames <= 0."
        )

    # ========================================================
    # MEAN
    # ========================================================

    mean = (
        running_sum
        / float(
            total_frames
        )
    )

    # ========================================================
    # VARIANCE
    # ========================================================

    variance = (
        running_squared_sum
        / float(
            total_frames
        )
        - np.square(
            mean
        )
    )

    variance = np.maximum(
        variance,
        EPS,
    )

    # ========================================================
    # STD
    # ========================================================

    std = np.sqrt(
        variance
    )

    # ========================================================
    # [80] -> [80,1]
    # ========================================================

    mean = (
        mean[
            :,
            None
        ]
        .astype(
            np.float32
        )
    )

    std = (
        std[
            :,
            None
        ]
        .astype(
            np.float32
        )
    )

    expected_shape = (
        N_MELS,
        1,
    )

    # ========================================================
    # HARD QC
    # ========================================================

    if (
        mean.shape
        != expected_shape
    ):

        raise RuntimeError(
            f"Mean shape invalid: "
            f"{mean.shape}"
        )

    if (
        std.shape
        != expected_shape
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
            "Ada std <= 0."
        )

    # ========================================================
    # CONDITION COUNTS
    # ========================================================

    clean_count = int(
        (
            train_df[
                "condition"
            ]
            .astype(str)
            .eq(
                "clean"
            )
        ).sum()
    )

    simulated_count = int(
        (
            train_df[
                "condition"
            ]
            .astype(str)
            .eq(
                "simulated"
            )
        ).sum()
    )

    real_count = int(
        (
            train_df[
                "condition"
            ]
            .astype(str)
            .eq(
                "real"
            )
        ).sum()
    )

    # ========================================================
    # SAVE STATS
    # ========================================================

    MGD_TRAIN_STATS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        MGD_TRAIN_STATS,

        mean=mean,

        std=std,

        total_frames=np.array(
            total_frames,
            dtype=np.int64,
        ),

        training_files=np.array(
            len(
                train_df
            ),
            dtype=np.int64,
        ),

        clean_training_files=np.array(
            clean_count,
            dtype=np.int64,
        ),

        simulated_training_files=np.array(
            simulated_count,
            dtype=np.int64,
        ),

        real_training_files=np.array(
            real_count,
            dtype=np.int64,
        ),

        sample_rate=np.array(
            TARGET_SR,
            dtype=np.int64,
        ),

        n_fft=np.array(
            N_FFT,
            dtype=np.int64,
        ),

        hop_length=np.array(
            HOP_LENGTH,
            dtype=np.int64,
        ),

        n_mels=np.array(
            N_MELS,
            dtype=np.int64,
        ),

        alpha=np.array(
            ALPHA,
            dtype=np.float32,
        ),

        gamma=np.array(
            GAMMA,
            dtype=np.float32,
        ),
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print("=" * 75)

    print(
        "MGD TRAINING STATISTICS COMPLETE"
    )

    print("=" * 75)

    print()

    print(
        f"Training files : "
        f"{len(train_df)}"
    )

    print(
        f"Processed      : "
        f"{processed_files}"
    )

    print(
        f"Total frames   : "
        f"{total_frames}"
    )

    print()

    print(
        "Training condition counts:"
    )

    print(
        f"Clean      : "
        f"{clean_count}"
    )

    print(
        f"Simulated  : "
        f"{simulated_count}"
    )

    print(
        f"Real       : "
        f"{real_count}"
    )

    print()

    print(
        f"Mean shape : "
        f"{mean.shape}"
    )

    print(
        f"Std shape  : "
        f"{std.shape}"
    )

    print()

    print(
        "MGD mean range:"
    )

    print(
        f"{float(mean.min()):.6f} "
        f"to "
        f"{float(mean.max()):.6f}"
    )

    print()

    print(
        "MGD std range:"
    )

    print(
        f"{float(std.min()):.6f} "
        f"to "
        f"{float(std.max()):.6f}"
    )

    print()

    print(
        "Statistics saved:"
    )

    print(
        MGD_TRAIN_STATS
    )

    print()

    print(
        "Validation/Test leakage : NONE"
    )

    print(
        "Raw MGD files           : UNCHANGED"
    )

    print()

    print(
        "MGD TRAINING-ONLY "
        "STATISTICS: PASSED"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Tahap 5 MGD extraction, "
            "distribution QC, and "
            "training-only normalization statistics."
        )
    )

    parser.add_argument(
        "--condition",

        choices=[
            "clean",
            "simulated",
            "real",
            "all",
        ],

        default="all",
    )

    parser.add_argument(
        "--limit",

        type=int,

        default=None,

        help=(
            "Pilot limit per condition. "
            "Example: --limit 20"
        ),
    )

    parser.add_argument(
        "--overwrite",

        action="store_true",

        help=(
            "Recompute existing MGD files."
        ),
    )

    parser.add_argument(
        "--compute-stats",

        action="store_true",

        help=(
            "Compute MGD normalization "
            "mean/std from split=train only. "
            "No extraction is performed."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # STATISTICS MODE
    # ========================================================

    if args.compute_stats:

        print("=" * 75)

        print(
            "MGD NORMALIZATION STATISTICS"
        )

        print("=" * 75)

        print()

        print(
            "Mode:"
        )

        print(
            "TRAINING-ONLY statistics"
        )

        print()

        print(
            "No audio extraction will run."
        )

        print(
            "No WAV file will be modified."
        )

        print(
            "No raw MGD file will be modified."
        )

        compute_training_statistics()

        return

    # ========================================================
    # EXTRACTION HEADER
    # ========================================================

    print("=" * 75)

    print(
        "MODIFIED GROUP DELAY "
        "EXTRACTION PIPELINE"
    )

    print("=" * 75)

    print()

    print(
        f"Device       : "
        f"{DEVICE}"
    )

    print(
        f"Target SR    : "
        f"{TARGET_SR}"
    )

    print(
        f"FFT          : "
        f"{N_FFT}"
    )

    print(
        f"Hop          : "
        f"{HOP_LENGTH}"
    )

    print(
        f"Mel bins     : "
        f"{N_MELS}"
    )

    print(
        f"Alpha        : "
        f"{ALPHA}"
    )

    print(
        f"Gamma        : "
        f"{GAMMA}"
    )

    print()

    print(
        "Audio files  : READ ONLY"
    )

    print(
        "MGD output   : separate .npz"
    )

    # ========================================================
    # EXTRACTOR
    # ========================================================

    extractor = SpectralPhase(
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop=HOP_LENGTH,
        n_mels=N_MELS,
    )

    extractor = extractor.to(
        DEVICE
    )

    extractor.eval()

    # ========================================================
    # CONDITIONS
    # ========================================================

    if args.condition == "all":

        conditions = [
            "clean",
            "simulated",
            "real",
        ]

    else:

        conditions = [
            args.condition
        ]

    manifest_frames = []

    qc_frames = []

    # ========================================================
    # EXTRACTION
    # ========================================================

    for condition in conditions:

        manifest_df, qc_df = (
            process_condition(
                condition=condition,

                extractor=extractor,

                limit=args.limit,

                overwrite=args.overwrite,
            )
        )

        manifest_frames.append(
            manifest_df
        )

        qc_frames.append(
            qc_df
        )

    new_manifest = pd.concat(
        manifest_frames,
        ignore_index=True,
    )

    qc_all = pd.concat(
        qc_frames,
        ignore_index=True,
    )

    # ========================================================
    # QC
    # ========================================================

    summarize_qc(
        qc_all
    )

    # ========================================================
    # PILOT
    # ========================================================

    if args.limit is not None:

        MGD_PILOT_MANIFEST.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        new_manifest.to_csv(
            MGD_PILOT_MANIFEST,
            index=False,
        )

        MGD_PILOT_QC.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        qc_all.to_csv(
            MGD_PILOT_QC,
            index=False,
        )

        print()

        print(
            "Pilot manifest:"
        )

        print(
            MGD_PILOT_MANIFEST
        )

        print()

        print(
            "Pilot QC:"
        )

        print(
            MGD_PILOT_QC
        )

        validate_manifest(
            new_manifest
        )

    # ========================================================
    # FULL MODE
    # ========================================================

    else:

        final_manifest = (
            save_final_manifest(
                new_df=new_manifest,

                conditions=conditions,
            )
        )

        print()

        print(
            "Final MGD manifest:"
        )

        print(
            MGD_MANIFEST
        )

        validate_manifest(
            final_manifest
        )

    # ========================================================
    # DONE
    # ========================================================

    print()

    print("=" * 75)

    print(
        "MGD EXTRACTION PASSED"
    )

    print("=" * 75)

    if args.limit is not None:

        print()

        print(
            "Mode: PILOT"
        )

        print(
            "Inspect distribution QC "
            "before full extraction."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()