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
# PROJECT PATH
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
# FEATURE OUTPUT
# ============================================================

FEATURE_ROOT = (
    ROOT
    / "data"
    / "features"
    / "spectral_phase"
)

RAW_FEATURE_ROOT = (
    FEATURE_ROOT
    / "raw"
)

OUTPUT_MANIFEST = (
    MANIFEST_DIR
    / "manifest_spectral_phase.csv"
)

PILOT_MANIFEST = (
    MANIFEST_DIR
    / "manifest_spectral_phase_pilot.csv"
)

TRAIN_STATS_PATH = (
    FEATURE_ROOT
    / "train_stats.npz"
)


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

TARGET_SR = 16000

N_FFT = 512

HOP_LENGTH = 160

N_MELS = 80

EXPECTED_CHANNELS = 2

STAT_EPS = 1e-12


# ============================================================
# COMPUTE DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MANIFEST CONFIGURATION
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

        # Controlled real replay:
        #
        # prioritaskan model copy 16 kHz / 16-bit.
        #
        # Master 48 kHz / 24-bit tetap dipertahankan
        # dan tidak dimodifikasi.
        "audio_columns": [
            "model_file_path",
            "master_file_path",
        ],
    },
}


# ============================================================
# SAFE FILE NAME
# ============================================================

def safe_name(value):
    """
    Membuat ID aman digunakan sebagai nama file.
    """

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
# PATH RESOLVER
# ============================================================

def resolve_audio_path(value):
    """
    Mendukung:
        - absolute path
        - relative path dari project root

    Tidak mengubah file sumber.
    """

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
    """
    Mengambil kolom path audio pertama
    yang tersedia pada manifest.
    """

    for column in candidates:

        if column in dataframe.columns:

            return column

    raise RuntimeError(
        "\nTidak ditemukan kolom path audio.\n"
        f"Candidates : {candidates}\n"
        f"Available  : {list(dataframe.columns)}"
    )


# ============================================================
# OPTIONAL METADATA
# ============================================================

def optional_value(
    row,
    column,
    default="",
):
    """
    Mengambil metadata bila tersedia.
    """

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
# AUDIO LOADER
# ============================================================

def load_audio_read_only(
    path,
    target_sr=16000,
):
    """
    Membaca audio tanpa memodifikasi WAV asli.

    Pipeline:

        file WAV
            ↓
        read ke RAM
            ↓
        mono conversion di RAM
            ↓
        resampling di RAM jika diperlukan
            ↓
        float32 waveform

    Tidak ada operasi write terhadap audio.
    """

    audio, sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )

    # ========================================================
    # EMPTY CHECK
    # ========================================================

    if audio.size == 0:

        raise RuntimeError(
            f"Audio kosong: {path}"
        )

    # ========================================================
    # FINITE CHECK
    # ========================================================

    if not np.isfinite(
        audio
    ).all():

        raise RuntimeError(
            f"Audio mengandung NaN/Inf: "
            f"{path}"
        )

    # ========================================================
    # MONO CONVERSION IN MEMORY
    # ========================================================

    audio = audio.mean(
        axis=1
    )

    # ========================================================
    # RESAMPLE IN MEMORY ONLY
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

    # ========================================================
    # FINAL QC
    # ========================================================

    if not np.isfinite(
        audio
    ).all():

        raise RuntimeError(
            f"Audio menjadi NaN/Inf "
            f"setelah preprocessing: {path}"
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
    Membuat feature ID unik.

    Priority:

        real:
            recording_id

        clean:
            source_file_id

        simulated:
            source_file_id + replay metadata

    Fallback:
        condition + row index
    """

    # ========================================================
    # RECORDING ID
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
    ):

        return safe_name(
            row[
                "recording_id"
            ]
        )

    # ========================================================
    # SOURCE FILE ID
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

        # ----------------------------------------------------
        # SIMULATED REPLAY
        # ----------------------------------------------------

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

            # Fallback simulated replay
            # agar derivative tetap unik.

            return (
                f"{source_id}"
                f"__simrow{row_index}"
            )

        # ----------------------------------------------------
        # CLEAN
        # ----------------------------------------------------

        return source_id

    # ========================================================
    # GENERAL FALLBACK
    # ========================================================

    return (
        f"{condition}"
        f"__row_{row_index:07d}"
    )


# ============================================================
# SAVE FEATURE
# ============================================================

def save_feature(
    feature,
    destination,
):
    """
    Menyimpan raw spectral-phase feature.

    Shape:
        [2,80,T]

    Raw feature TIDAK dinormalisasi di sini.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        destination,

        feature=feature.astype(
            np.float32,
            copy=False,
        ),
    )


# ============================================================
# VALIDATE FEATURE ARRAY
# ============================================================

def validate_feature_array(
    feature,
    feature_id,
):
    """
    Hard QC untuk setiap feature.
    """

    # ========================================================
    # DIMENSION
    # ========================================================

    if feature.ndim != 3:

        raise RuntimeError(
            f"{feature_id}: "
            f"feature bukan tensor 3D. "
            f"Shape={feature.shape}"
        )

    # ========================================================
    # CHANNEL
    # ========================================================

    if (
        feature.shape[0]
        != EXPECTED_CHANNELS
    ):

        raise RuntimeError(
            f"{feature_id}: "
            f"expected "
            f"{EXPECTED_CHANNELS} channels, "
            f"actual={feature.shape[0]}"
        )

    # ========================================================
    # MEL BINS
    # ========================================================

    if (
        feature.shape[1]
        != N_MELS
    ):

        raise RuntimeError(
            f"{feature_id}: "
            f"expected {N_MELS} Mel bins, "
            f"actual={feature.shape[1]}"
        )

    # ========================================================
    # TIME FRAMES
    # ========================================================

    if (
        feature.shape[2]
        <= 0
    ):

        raise RuntimeError(
            f"{feature_id}: "
            "jumlah frame <= 0."
        )

    # ========================================================
    # FINITE
    # ========================================================

    if not np.isfinite(
        feature
    ).all():

        raise RuntimeError(
            f"{feature_id}: "
            "feature mengandung NaN/Inf."
        )


# ============================================================
# PROCESS ONE CONDITION
# ============================================================

def process_condition(
    condition,
    extractor,
    limit=None,
    overwrite=False,
):
    """
    Ekstrak satu condition:

        clean
        simulated
        real
    """

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
    # MANIFEST CHECK
    # ========================================================

    if not manifest_path.exists():

        raise FileNotFoundError(
            f"Manifest tidak ditemukan:\n"
            f"{manifest_path}"
        )

    # ========================================================
    # LOAD MANIFEST
    # ========================================================

    df = pd.read_csv(
        manifest_path,
        low_memory=False,
    )

    if df.empty:

        raise RuntimeError(
            f"{manifest_path.name} kosong."
        )

    # ========================================================
    # AUDIO PATH COLUMN
    # ========================================================

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
    # INFORMATION
    # ========================================================

    print()
    print("=" * 75)
    print(
        f"PROCESSING CONDITION: "
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
        f"Device        : "
        f"{DEVICE}"
    )

    if limit is not None:

        print(
            f"Pilot limit   : "
            f"{limit}"
        )

    # ========================================================
    # DESTINATION
    # ========================================================

    condition_dir = (
        RAW_FEATURE_ROOT
        / condition
    )

    condition_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    successful_rows = []

    errors = []

    # ========================================================
    # ITERATE
    # ========================================================

    for row_index, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=condition,
    ):

        try:

            # =================================================
            # AUDIO PATH
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
            # FEATURE ID
            # =================================================

            feature_id = (
                build_feature_id(
                    row=row,
                    condition=condition,
                    row_index=row_index,
                )
            )

            feature_path = (
                condition_dir
                / f"{feature_id}.npz"
            )

            # =================================================
            # REUSE EXISTING FEATURE
            # =================================================

            if (
                feature_path.exists()
                and
                not overwrite
            ):

                with np.load(
                    feature_path
                ) as saved:

                    feature = (
                        saved[
                            "feature"
                        ]
                    )

                validate_feature_array(
                    feature,
                    feature_id,
                )

            # =================================================
            # COMPUTE NEW FEATURE
            # =================================================

            else:

                # ---------------------------------------------
                # LOAD AUDIO
                # ---------------------------------------------

                waveform, sr = (
                    load_audio_read_only(
                        audio_path,
                        target_sr=TARGET_SR,
                    )
                )

                if sr != TARGET_SR:

                    raise RuntimeError(
                        f"Unexpected sample rate "
                        f"after preprocessing: "
                        f"{sr}"
                    )

                # ---------------------------------------------
                # NUMPY -> TORCH
                # ---------------------------------------------

                wav_tensor = (
                    torch.from_numpy(
                        waveform
                    )
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(
                        DEVICE
                    )
                )

                # [1,1,T]

                # ---------------------------------------------
                # SPECTRAL-PHASE EXTRACTION
                # ---------------------------------------------

                with torch.no_grad():

                    feature_tensor = (
                        extractor(
                            wav_tensor
                        )
                    )

                # Expected:
                # [1,2,80,T]

                if (
                    feature_tensor.ndim
                    != 4
                ):

                    raise RuntimeError(
                        "Extractor output "
                        "bukan tensor 4D."
                    )

                # ---------------------------------------------
                # REMOVE BATCH DIMENSION
                # ---------------------------------------------

                feature = (
                    feature_tensor[
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

                # [2,80,T]

                # ---------------------------------------------
                # HARD QC
                # ---------------------------------------------

                validate_feature_array(
                    feature,
                    feature_id,
                )

                # ---------------------------------------------
                # SAVE RAW FEATURE
                # ---------------------------------------------

                save_feature(
                    feature,
                    feature_path,
                )

            # =================================================
            # AUDIO PATH RELATIVE TO PROJECT
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

            feature_relative = str(
                feature_path.relative_to(
                    ROOT
                )
            )

            # =================================================
            # OUTPUT MANIFEST ROW
            # =================================================

            successful_rows.append(
                {
                    "global_feature_id":
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

                    "feature_path":
                        feature_relative,

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

                    "distance_m":
                        optional_value(
                            row,
                            "distance_m",
                        ),

                    "angle_deg":
                        optional_value(
                            row,
                            "angle_deg",
                        ),

                    "repetition":
                        optional_value(
                            row,
                            "repetition",
                        ),

                    "sample_rate":
                        TARGET_SR,

                    "n_fft":
                        N_FFT,

                    "hop_length":
                        HOP_LENGTH,

                    "n_mels":
                        N_MELS,

                    "n_channels":
                        int(
                            feature.shape[0]
                        ),

                    "n_frames":
                        int(
                            feature.shape[2]
                        ),

                    "finite":
                        bool(
                            np.isfinite(
                                feature
                            ).all()
                        ),
                }
            )

        except Exception as exc:

            errors.append(
                {
                    "row_index":
                        row_index,

                    "error":
                        str(
                            exc
                        ),
                }
            )

    # ========================================================
    # CONDITION SUMMARY
    # ========================================================

    print()
    print("=" * 75)
    print(
        f"{condition.upper()} "
        "EXTRACTION SUMMARY"
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
            "First extraction errors:"
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
            f"{len(errors)} extraction "
            "error(s) ditemukan."
        )

    return pd.DataFrame(
        successful_rows
    )


# ============================================================
# SAVE / MERGE FEATURE MANIFEST
# ============================================================

def save_feature_manifest(
    new_df,
    conditions,
    replace_condition=True,
):
    """
    Menyimpan manifest spectral-phase final.

    Saat satu condition diekstrak ulang:
        condition tersebut diganti.

    Condition lain:
        tetap dipertahankan.
    """

    if (
        OUTPUT_MANIFEST.exists()
        and
        replace_condition
    ):

        old_df = pd.read_csv(
            OUTPUT_MANIFEST,
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

    # ========================================================
    # DUPLICATE GLOBAL FEATURE ID
    # ========================================================

    duplicate_mask = (
        combined[
            "global_feature_id"
        ]
        .duplicated(
            keep=False
        )
    )

    if duplicate_mask.any():

        duplicate_rows = (
            combined[
                duplicate_mask
            ][
                [
                    "global_feature_id",
                    "condition",
                    "audio_path",
                ]
            ]
        )

        print()
        print(
            "DUPLICATE FEATURE IDS:"
        )

        print(
            duplicate_rows.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Duplicate global_feature_id "
            "ditemukan."
        )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        OUTPUT_MANIFEST,
        index=False,
    )

    return combined


# ============================================================
# FEATURE MANIFEST QC
# ============================================================

def validate_feature_manifest(
    feature_manifest,
):
    """
    Validasi metadata feature extraction.
    """

    print()
    print("=" * 75)
    print(
        "SPECTRAL-PHASE MANIFEST QC"
    )
    print("=" * 75)

    if feature_manifest.empty:

        raise RuntimeError(
            "Feature manifest kosong."
        )

    # ========================================================
    # DUPLICATES
    # ========================================================

    if (
        feature_manifest[
            "global_feature_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate global_feature_id."
        )

    # ========================================================
    # CHANNELS
    # ========================================================

    if not (
        feature_manifest[
            "n_channels"
        ]
        .astype(int)
        .eq(
            EXPECTED_CHANNELS
        )
        .all()
    ):

        raise RuntimeError(
            "Ada feature yang bukan "
            "2-channel."
        )

    # ========================================================
    # MEL BINS
    # ========================================================

    if not (
        feature_manifest[
            "n_mels"
        ]
        .astype(int)
        .eq(
            N_MELS
        )
        .all()
    ):

        raise RuntimeError(
            "Ada feature dengan "
            "jumlah Mel bin != 80."
        )

    # ========================================================
    # FINITE FLAG
    # ========================================================

    finite_values = (
        feature_manifest[
            "finite"
        ]
        .astype(str)
        .str.lower()
    )

    if not finite_values.eq(
        "true"
    ).all():

        raise RuntimeError(
            "Ada feature NaN/Inf."
        )

    # ========================================================
    # FEATURE FILE EXISTS
    # ========================================================

    missing_features = []

    for value in (
        feature_manifest[
            "feature_path"
        ]
    ):

        path = (
            ROOT
            / str(value)
        )

        if not path.exists():

            missing_features.append(
                str(path)
            )

    if missing_features:

        raise RuntimeError(
            f"{len(missing_features)} "
            "feature file missing."
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        f"Total features : "
        f"{len(feature_manifest)}"
    )

    print()
    print(
        "Condition distribution:"
    )

    print(
        feature_manifest[
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
        feature_manifest[
            "split"
        ]
        .value_counts(
            dropna=False
        )
    )

    print()
    print(
        "Channel check   : PASSED"
    )

    print(
        "Mel-bin check   : PASSED"
    )

    print(
        "Finite check    : PASSED"
    )

    print(
        "Missing files   : 0"
    )

    print(
        "Duplicate IDs   : 0"
    )


# ============================================================
# TRAINING-ONLY NORMALIZATION STATISTICS
# ============================================================

def compute_training_statistics():
    """
    Menghitung statistik standardisasi hanya dari:

        split = train

    Statistik dihitung per:
        channel × Mel-bin

    Input:
        [2,80,T]

    Output:
        mean = [2,80,1]
        std  = [2,80,1]

    Validation dan test TIDAK pernah digunakan.
    """

    print()
    print("=" * 75)
    print(
        "TRAINING-ONLY NORMALIZATION STATISTICS"
    )
    print("=" * 75)

    # ========================================================
    # FEATURE MANIFEST
    # ========================================================

    if not OUTPUT_MANIFEST.exists():

        raise FileNotFoundError(
            "manifest_spectral_phase.csv "
            "tidak ditemukan:\n"
            f"{OUTPUT_MANIFEST}"
        )

    df = pd.read_csv(
        OUTPUT_MANIFEST,
        low_memory=False,
    )

    if df.empty:

        raise RuntimeError(
            "manifest_spectral_phase.csv kosong."
        )

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required_columns = {
        "feature_path",
        "split",
        "condition",
        "n_channels",
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
            "Feature manifest kehilangan "
            "kolom:\n"
            f"{sorted(missing_columns)}"
        )

    # ========================================================
    # FILTER TRAIN ONLY
    # ========================================================

    split_normalized = (
        df[
            "split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    train_df = (
        df[
            split_normalized.eq(
                "train"
            )
        ]
        .copy()
    )

    if train_df.empty:

        raise RuntimeError(
            "Tidak ditemukan feature "
            "dengan split=train."
        )

    # ========================================================
    # INFORMATION
    # ========================================================

    print()
    print(
        f"All features      : "
        f"{len(df)}"
    )

    print(
        f"Training features : "
        f"{len(train_df)}"
    )

    print()
    print(
        "Training condition distribution:"
    )

    train_condition_counts = (
        train_df[
            "condition"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        train_condition_counts
    )

    print()
    print(
        "Validation/test are excluded "
        "from normalization statistics."
    )

    # ========================================================
    # RUNNING ACCUMULATORS
    # ========================================================
    #
    # float64 dipakai agar akumulasi lebih stabil.
    #
    # Shape:
    #
    #   [channel, mel]
    #
    # ========================================================

    running_sum = np.zeros(
        (
            EXPECTED_CHANNELS,
            N_MELS,
        ),
        dtype=np.float64,
    )

    running_squared_sum = np.zeros(
        (
            EXPECTED_CHANNELS,
            N_MELS,
        ),
        dtype=np.float64,
    )

    total_frames = 0

    processed_files = 0

    errors = []

    # ========================================================
    # ITERATE TRAINING FEATURES
    # ========================================================

    for _, row in tqdm(
        train_df.iterrows(),
        total=len(train_df),
        desc="train statistics",
    ):

        try:

            # =================================================
            # FEATURE PATH
            # =================================================

            feature_path = (
                ROOT
                / str(
                    row[
                        "feature_path"
                    ]
                )
            )

            if not feature_path.exists():

                raise FileNotFoundError(
                    f"Missing feature: "
                    f"{feature_path}"
                )

            # =================================================
            # LOAD RAW FEATURE
            # =================================================

            with np.load(
                feature_path
            ) as data:

                if (
                    "feature"
                    not in data.files
                ):

                    raise RuntimeError(
                        "NPZ tidak memiliki "
                        "key 'feature'."
                    )

                feature = (
                    data[
                        "feature"
                    ]
                    .astype(
                        np.float64,
                        copy=False,
                    )
                )

            # =================================================
            # SHAPE QC
            # =================================================

            if feature.ndim != 3:

                raise RuntimeError(
                    f"Invalid ndim: "
                    f"{feature.shape}"
                )

            if (
                feature.shape[0]
                != EXPECTED_CHANNELS
            ):

                raise RuntimeError(
                    f"Expected "
                    f"{EXPECTED_CHANNELS} channels, "
                    f"got {feature.shape}"
                )

            if (
                feature.shape[1]
                != N_MELS
            ):

                raise RuntimeError(
                    f"Expected "
                    f"{N_MELS} Mel bins, "
                    f"got {feature.shape}"
                )

            if (
                feature.shape[2]
                <= 0
            ):

                raise RuntimeError(
                    "Feature mempunyai "
                    "0 time frames."
                )

            # =================================================
            # FINITE QC
            # =================================================

            if not np.isfinite(
                feature
            ).all():

                raise RuntimeError(
                    "Feature mengandung NaN/Inf."
                )

            # =================================================
            # ACCUMULATE ALONG TIME DIMENSION
            # =================================================

            running_sum += (
                feature.sum(
                    axis=2
                )
            )

            running_squared_sum += (
                np.square(
                    feature
                ).sum(
                    axis=2
                )
            )

            total_frames += int(
                feature.shape[2]
            )

            processed_files += 1

        except Exception as exc:

            errors.append(
                {
                    "feature_path":
                        str(
                            row[
                                "feature_path"
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
            "Training statistics errors:"
        )

        for item in (
            errors[:10]
        ):

            print(
                f"{item['feature_path']} "
                f"-> "
                f"{item['error']}"
            )

        raise RuntimeError(
            f"{len(errors)} feature gagal "
            "saat menghitung statistics."
        )

    if (
        processed_files
        != len(
            train_df
        )
    ):

        raise RuntimeError(
            "Jumlah training feature "
            "yang diproses tidak sesuai."
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

    # ========================================================
    # NUMERICAL PROTECTION
    # ========================================================

    variance = np.maximum(
        variance,
        STAT_EPS,
    )

    # ========================================================
    # STANDARD DEVIATION
    # ========================================================

    std = np.sqrt(
        variance
    )

    # ========================================================
    # [2,80] -> [2,80,1]
    # ========================================================

    mean = (
        mean[
            ...,
            None
        ]
        .astype(
            np.float32
        )
    )

    std = (
        std[
            ...,
            None
        ]
        .astype(
            np.float32
        )
    )

    expected_shape = (
        EXPECTED_CHANNELS,
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

    clean_train_count = int(
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

    simulated_train_count = int(
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

    real_train_count = int(
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
    # SAVE TRAINING STATISTICS
    # ========================================================

    TRAIN_STATS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        TRAIN_STATS_PATH,

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
            clean_train_count,
            dtype=np.int64,
        ),

        simulated_training_files=np.array(
            simulated_train_count,
            dtype=np.int64,
        ),

        real_training_files=np.array(
            real_train_count,
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

        channels=np.array(
            EXPECTED_CHANNELS,
            dtype=np.int64,
        ),
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 75)
    print(
        "TRAINING STATISTICS COMPLETE"
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
        f"{clean_train_count}"
    )

    print(
        f"Simulated  : "
        f"{simulated_train_count}"
    )

    print(
        f"Real       : "
        f"{real_train_count}"
    )

    print()
    print(
        f"Mean shape     : "
        f"{mean.shape}"
    )

    print(
        f"Std shape      : "
        f"{std.shape}"
    )

    print()
    print(
        "Channel 0 = Log-Mel"
    )

    print(
        "Channel 1 = Phase-derived"
    )

    print()
    print(
        "Log-Mel mean range:"
    )

    print(
        f"{float(mean[0].min()):.6f} "
        f"to "
        f"{float(mean[0].max()):.6f}"
    )

    print()
    print(
        "Log-Mel std range:"
    )

    print(
        f"{float(std[0].min()):.6f} "
        f"to "
        f"{float(std[0].max()):.6f}"
    )

    print()
    print(
        "Phase mean range:"
    )

    print(
        f"{float(mean[1].min()):.6f} "
        f"to "
        f"{float(mean[1].max()):.6f}"
    )

    print()
    print(
        "Phase std range:"
    )

    print(
        f"{float(std[1].min()):.6f} "
        f"to "
        f"{float(std[1].max()):.6f}"
    )

    print()
    print(
        "Statistics saved:"
    )

    print(
        TRAIN_STATS_PATH
    )

    print()
    print(
        "Validation/Test leakage : NONE"
    )

    print(
        "Raw feature files       : UNCHANGED"
    )

    print()
    print(
        "TRAINING-ONLY STATISTICS: PASSED"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract synchronized "
            "Log-Mel + phase-derived "
            "features and compute "
            "training-only normalization statistics."
        )
    )

    # ========================================================
    # CONDITION
    # ========================================================

    parser.add_argument(
        "--condition",

        choices=[
            "clean",
            "simulated",
            "real",
            "all",
        ],

        default="real",
    )

    # ========================================================
    # PILOT LIMIT
    # ========================================================

    parser.add_argument(
        "--limit",

        type=int,

        default=None,

        help=(
            "Pilot extraction. "
            "Example: --limit 20"
        ),
    )

    # ========================================================
    # OVERWRITE
    # ========================================================

    parser.add_argument(
        "--overwrite",

        action="store_true",

        help=(
            "Recompute feature even if "
            "NPZ already exists."
        ),
    )

    # ========================================================
    # TRAINING STATISTICS MODE
    # ========================================================

    parser.add_argument(
        "--compute-stats",

        action="store_true",

        help=(
            "Compute normalization mean/std "
            "from split=train only. "
            "No feature extraction is performed."
        ),
    )

    # ========================================================
    # PARSE
    # ========================================================

    args = parser.parse_args()

    # ========================================================
    # STATISTICS MODE
    # ========================================================

    if args.compute_stats:

        print("=" * 75)
        print(
            "SPECTRAL-PHASE NORMALIZATION "
            "STATISTICS"
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
            "No raw feature file will be "
            "modified."
        )

        compute_training_statistics()

        return

    # ========================================================
    # EXTRACTION MODE
    # ========================================================

    print("=" * 75)
    print(
        "LOG-MEL + PHASE-DERIVED "
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
        f"{TARGET_SR} Hz"
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
        "Channels     : "
        "2"
    )

    print(
        "Channel 0    : "
        "Log-Mel"
    )

    print(
        "Channel 1    : "
        "Phase-derived"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Audio source files are "
        "READ ONLY."
    )

    print(
        "Resampling, if required, "
        "occurs only in RAM."
    )

    # ========================================================
    # FEATURE EXTRACTOR
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

    extracted_frames = []

    # ========================================================
    # EXTRACT
    # ========================================================

    for condition in conditions:

        result_df = process_condition(
            condition=condition,

            extractor=extractor,

            limit=args.limit,

            overwrite=args.overwrite,
        )

        extracted_frames.append(
            result_df
        )

    new_df = pd.concat(
        extracted_frames,
        ignore_index=True,
    )

    # ========================================================
    # PILOT MODE
    # ========================================================
    #
    # Pilot feature tidak digabungkan ke
    # manifest final.
    # ========================================================

    if args.limit is not None:

        new_df.to_csv(
            PILOT_MANIFEST,
            index=False,
        )

        print()
        print(
            "Pilot feature manifest:"
        )

        print(
            PILOT_MANIFEST
        )

        validate_feature_manifest(
            new_df
        )

    # ========================================================
    # FINAL EXTRACTION MODE
    # ========================================================

    else:

        final_manifest = (
            save_feature_manifest(
                new_df=new_df,

                conditions=conditions,

                replace_condition=True,
            )
        )

        print()
        print(
            "Final feature manifest:"
        )

        print(
            OUTPUT_MANIFEST
        )

        validate_feature_manifest(
            final_manifest
        )

    # ========================================================
    # SUCCESS
    # ========================================================

    print()
    print("=" * 75)
    print(
        "SPECTRAL-PHASE EXTRACTION PASSED"
    )
    print("=" * 75)

    print()
    print(
        "Audio originals remain unchanged."
    )

    print(
        "Raw feature tensors stored "
        "separately as .npz."
    )

    if args.limit is not None:

        print()
        print(
            "Mode: PILOT"
        )

        print(
            "Run full extraction only after "
            "pilot validation passes."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()