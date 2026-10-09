from pathlib import Path
import random

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio


# =========================================================
# PATH
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

SOURCE_MANIFEST = (
    ROOT
    / "manifests"
    / "source_manifest.csv"
)

RIR_MANIFEST = (
    ROOT
    / "manifests"
    / "rir_manifest.csv"
)

OUTPUT_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "simulated_replay"
)

OUTPUT_MANIFEST = (
    ROOT
    / "manifests"
    / "manifest_replay_sim.csv"
)


# =========================================================
# GENERAL CONFIGURATION
# =========================================================

TARGET_SR = 16000

SEED = 2026

SNR_VALUES = [
    20,
    10,
    5,
]


# =========================================================
# RIR CONFIGURATION
# =========================================================
#
# Panduan:
# minimal 5 RIR seen
# minimal 2 RIR unseen
#
# Pemilihan kita menjaga diversity room.
# =========================================================

N_SEEN_RIR = 5
N_UNSEEN_RIR = 2

SEEN_ROOMS = [
    "lecture",
    "meeting",
    "phone",
    "stairway",
]

UNSEEN_ROOMS = [
    "booth",
    "office",
]


# =========================================================
# PILOT MODE
# =========================================================
#
# Pilot digunakan sebelum scale-up.
#
# Total source:
#
# train:
#   10 bona fide
#   10 spoof
#
# validation:
#   8 bona fide
#   7 spoof
#
# test:
#   7 bona fide
#   8 spoof
#
# TOTAL:
#   bona fide = 25
#   spoof     = 25
#   source    = 50
#
# 50 source × 3 SNR = 150 output
# =========================================================

PILOT_MODE = True

PILOT_LABEL_QUOTA = {
    "train": {
        0: 10,
        1: 10,
    },
    "validation": {
        0: 8,
        1: 7,
    },
    "test": {
        0: 7,
        1: 8,
    },
}


# =========================================================
# OUTPUT PEAK
# =========================================================
#
# Hindari waveform menempel pada ±1.
# =========================================================

TARGET_PEAK = 0.95


# =========================================================
# RANDOM SEED
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================================================
# LOAD AUDIO
# =========================================================

def load_audio(path: Path):
    """
    Membaca WAV atau FLAC menggunakan SoundFile.

    SoundFile:
        [time, channels]

    PyTorch:
        [channels, time]
    """

    audio, sr = sf.read(
        str(path),
        always_2d=True,
        dtype="float32",
    )

    wav = torch.from_numpy(
        audio.T
    )

    # Ubah ke mono jika multi-channel
    if wav.shape[0] > 1:

        wav = wav.mean(
            dim=0,
            keepdim=True,
        )

    return wav, sr


# =========================================================
# RESAMPLE AUDIO
# =========================================================

def resample_audio(
    wav: torch.Tensor,
    original_sr: int,
    target_sr: int,
):
    """
    Resample source/RIR ke 16 kHz.
    """

    if original_sr == target_sr:
        return wav

    return torchaudio.functional.resample(
        wav,
        original_sr,
        target_sr,
    )


# =========================================================
# APPLY RIR
# =========================================================
#
# Struktur utama mengikuti starter panduan.
#
# =========================================================

def apply_rir(
    wav: torch.Tensor,
    rir: torch.Tensor,
):
    """
    Melakukan FFT convolution antara waveform dan RIR.

    Args:
        wav : [1, time]
        rir : [1, time]

    Returns:
        reverberated waveform
    """

    # =====================================================
    # NORMALIZE RIR
    # =====================================================

    rir = (
        rir
        / rir.abs().max().clamp_min(1e-6)
    )

    # =====================================================
    # CONVOLUTION LENGTH
    # =====================================================

    n = (
        wav.shape[-1]
        + rir.shape[-1]
        - 1
    )

    # =====================================================
    # FFT
    # =====================================================

    wav_fft = torch.fft.rfft(
        wav,
        n=n,
    )

    rir_fft = torch.fft.rfft(
        rir,
        n=n,
    )

    # =====================================================
    # MULTIPLICATION IN FREQUENCY DOMAIN
    # =====================================================

    y = torch.fft.irfft(
        wav_fft * rir_fft,
        n=n,
    )

    # =====================================================
    # MATCH SOURCE LENGTH
    # =====================================================
    #
    # Mengikuti starter panduan.
    # =====================================================

    y = y[
        ...,
        :wav.shape[-1]
    ]

    # =====================================================
    # NORMALIZE REVERBERATED SIGNAL
    # =====================================================

    peak = (
        y
        .abs()
        .max()
        .clamp_min(1e-6)
    )

    y = y / peak

    return y


# =========================================================
# ADD NOISE AT TARGET SNR
# =========================================================

def add_noise_snr(
    wav: torch.Tensor,
    snr_db: float,
    target_peak: float = TARGET_PEAK,
):
    """
    Menambahkan Gaussian noise sesuai target SNR.

    Setelah waveform + noise digabungkan,
    seluruh mixture di-scale jika peak melebihi target_peak.

    Scaling dilakukan pada signal + noise secara bersamaan,
    sehingga rasio SNR tetap dipertahankan.
    """

    # =====================================================
    # GENERATE NOISE
    # =====================================================

    noise = torch.randn_like(
        wav
    )

    # =====================================================
    # SIGNAL RMS
    # =====================================================

    signal_rms = (
        wav
        .pow(2)
        .mean()
        .sqrt()
        .clamp_min(1e-6)
    )

    # =====================================================
    # NOISE RMS
    # =====================================================

    noise_rms = (
        noise
        .pow(2)
        .mean()
        .sqrt()
        .clamp_min(1e-6)
    )

    # =====================================================
    # SCALE NOISE TO TARGET SNR
    # =====================================================

    noise = (
        noise
        * signal_rms
        / (
            10 ** (snr_db / 20.0)
            * noise_rms
        )
    )

    # =====================================================
    # MIX
    # =====================================================

    mixture = wav + noise

    # =====================================================
    # CONSERVATIVE PEAK NORMALIZATION
    # =====================================================

    peak = (
        mixture
        .abs()
        .max()
        .clamp_min(1e-6)
    )

    if peak > target_peak:

        mixture = (
            mixture
            / peak
            * target_peak
        )

    return mixture


# =========================================================
# SAVE AUDIO
# =========================================================

def save_audio(
    path: Path,
    wav: torch.Tensor,
):
    """
    Menyimpan simulated replay sebagai WAV PCM16.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audio = (
        wav
        .squeeze(0)
        .cpu()
        .numpy()
    )

    sf.write(
        str(path),
        audio,
        TARGET_SR,
        subtype="PCM_16",
    )


# =========================================================
# OUTPUT QUALITY CONTROL
# =========================================================

def inspect_output(
    wav: torch.Tensor,
):
    """
    Mengukur peak, RMS, silence, dan clipping.
    """

    peak = float(
        wav
        .abs()
        .max()
        .item()
    )

    rms = float(
        wav
        .pow(2)
        .mean()
        .sqrt()
        .item()
    )

    silent = (
        rms < 1e-6
    )

    clipped = (
        peak >= 0.999
    )

    return {
        "peak": peak,
        "rms": rms,
        "silent": silent,
        "clipped": clipped,
    }


# =========================================================
# SELECT RIR SUBSET
# =========================================================

def select_rir_subset(
    rir_manifest: pd.DataFrame,
):
    """
    Memilih:

        5 seen RIR
        2 unseen RIR

    dengan diversity room.

    Seen:
        lecture  -> minimal 1
        meeting  -> minimal 1
        phone    -> minimal 1
        stairway -> minimal 1

    lalu satu RIR tambahan.

    Unseen:
        booth  -> 1
        office -> 1
    """

    # =====================================================
    # SEEN DATAFRAME
    # =====================================================

    seen_df = (
        rir_manifest[
            rir_manifest["rir_split"]
            == "seen"
        ]
        .copy()
    )

    selected_seen = []

    # =====================================================
    # ONE RIR FROM EACH SEEN ROOM
    # =====================================================

    for room_index, room in enumerate(
        SEEN_ROOMS
    ):

        room_df = (
            seen_df[
                seen_df["room"]
                == room
            ]
        )

        if room_df.empty:

            raise ValueError(
                f"Tidak tersedia seen RIR "
                f"untuk room: {room}"
            )

        # seed berbeda per room tetapi deterministic
        sample = room_df.sample(
            n=1,
            random_state=SEED + room_index,
        )

        selected_seen.append(
            sample
        )

    seen = pd.concat(
        selected_seen,
        ignore_index=True,
    )

    # =====================================================
    # FIFTH SEEN RIR
    # =====================================================

    remaining_seen = (
        seen_df[
            ~seen_df["rir_id"].isin(
                seen["rir_id"]
            )
        ]
    )

    extra_seen = (
        remaining_seen
        .sample(
            n=1,
            random_state=SEED + 100,
        )
    )

    seen = pd.concat(
        [
            seen,
            extra_seen,
        ],
        ignore_index=True,
    )

    # =====================================================
    # UNSEEN DATAFRAME
    # =====================================================

    unseen_df = (
        rir_manifest[
            rir_manifest["rir_split"]
            == "unseen"
        ]
        .copy()
    )

    selected_unseen = []

    # =====================================================
    # ONE RIR FROM EACH UNSEEN ROOM
    # =====================================================

    for room_index, room in enumerate(
        UNSEEN_ROOMS
    ):

        room_df = (
            unseen_df[
                unseen_df["room"]
                == room
            ]
        )

        if room_df.empty:

            raise ValueError(
                f"Tidak tersedia unseen RIR "
                f"untuk room: {room}"
            )

        sample = room_df.sample(
            n=1,
            random_state=SEED + 200 + room_index,
        )

        selected_unseen.append(
            sample
        )

    unseen = pd.concat(
        selected_unseen,
        ignore_index=True,
    )

    # =====================================================
    # HARD CHECK
    # =====================================================

    assert len(seen) == N_SEEN_RIR, (
        "Jumlah seen RIR harus 5."
    )

    assert len(unseen) == N_UNSEEN_RIR, (
        "Jumlah unseen RIR harus 2."
    )

    assert set(SEEN_ROOMS).issubset(
        set(seen["room"])
    ), (
        "Tidak semua seen room terwakili."
    )

    assert set(UNSEEN_ROOMS) == (
        set(unseen["room"])
    ), (
        "Unseen RIR harus mencakup booth dan office."
    )

    return seen, unseen


# =========================================================
# CREATE BALANCED PILOT SOURCE
# =========================================================

def create_pilot_source(
    source_manifest: pd.DataFrame,
):
    """
    Membuat pilot dataset yang balanced.

    Total:
        25 bona fide
        25 spoof

    Split tetap dipertahankan.
    """

    if not PILOT_MODE:

        return (
            source_manifest
            .copy()
            .reset_index(drop=True)
        )

    sampled = []

    for split, label_quota in (
        PILOT_LABEL_QUOTA.items()
    ):

        for label, n in (
            label_quota.items()
        ):

            subset = (
                source_manifest[
                    (
                        source_manifest["split"]
                        == split
                    )
                    &
                    (
                        source_manifest["label"]
                        == label
                    )
                ]
            )

            if len(subset) < n:

                raise ValueError(
                    f"Split={split}, label={label} "
                    f"hanya memiliki {len(subset)} source, "
                    f"pilot membutuhkan {n}."
                )

            # deterministic per split + label
            seed_offset = (
                len(sampled)
                * 10
            )

            sample = subset.sample(
                n=n,
                random_state=(
                    SEED + seed_offset
                ),
            )

            sampled.append(
                sample
            )

    pilot = pd.concat(
        sampled,
        ignore_index=True,
    )

    # Shuffle supaya label tidak tersusun blok
    pilot = (
        pilot
        .sample(
            frac=1.0,
            random_state=SEED,
        )
        .reset_index(drop=True)
    )

    # =====================================================
    # PILOT VALIDATION
    # =====================================================

    assert len(pilot) == 50, (
        "Pilot harus memiliki 50 source."
    )

    label_counts = (
        pilot["label"]
        .value_counts()
    )

    assert label_counts.get(0, 0) == 25, (
        "Pilot harus memiliki 25 bona fide."
    )

    assert label_counts.get(1, 0) == 25, (
        "Pilot harus memiliki 25 spoof."
    )

    return pilot


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("SIMULATED REPLAY GENERATOR")
    print("=" * 70)

    # =====================================================
    # CHECK INPUT FILES
    # =====================================================

    if not SOURCE_MANIFEST.exists():

        raise FileNotFoundError(
            f"Source manifest tidak ditemukan:\n"
            f"{SOURCE_MANIFEST}"
        )

    if not RIR_MANIFEST.exists():

        raise FileNotFoundError(
            f"RIR manifest tidak ditemukan:\n"
            f"{RIR_MANIFEST}"
        )

    # =====================================================
    # LOAD MANIFESTS
    # =====================================================

    source_df = pd.read_csv(
        SOURCE_MANIFEST
    )

    rir_df = pd.read_csv(
        RIR_MANIFEST
    )

    # =====================================================
    # VALIDATE SOURCE MANIFEST
    # =====================================================

    required_source_columns = {
        "source_file_id",
        "file_path",
        "speaker_id",
        "attack_id",
        "label",
        "dataset",
        "split",
    }

    missing_source_columns = (
        required_source_columns
        - set(source_df.columns)
    )

    if missing_source_columns:

        raise ValueError(
            "Source manifest kehilangan kolom: "
            f"{sorted(missing_source_columns)}"
        )

    # =====================================================
    # VALIDATE RIR MANIFEST
    # =====================================================

    required_rir_columns = {
        "rir_id",
        "file_path",
        "room",
        "rir_split",
        "sample_rate",
    }

    missing_rir_columns = (
        required_rir_columns
        - set(rir_df.columns)
    )

    if missing_rir_columns:

        raise ValueError(
            "RIR manifest kehilangan kolom: "
            f"{sorted(missing_rir_columns)}"
        )

    # =====================================================
    # SELECT RIR
    # =====================================================

    seen_rirs, unseen_rirs = (
        select_rir_subset(
            rir_df
        )
    )

    print()
    print("Selected seen RIR:")

    print(
        seen_rirs[
            [
                "rir_id",
                "room",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("Selected unseen RIR:")

    print(
        unseen_rirs[
            [
                "rir_id",
                "room",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # =====================================================
    # SOURCE SELECTION
    # =====================================================

    working_sources = (
        create_pilot_source(
            source_df
        )
    )

    print()
    print("Source mode:")

    if PILOT_MODE:
        print("PILOT")
    else:
        print("FULL")

    # =====================================================
    # SOURCE SUMMARY
    # =====================================================

    print()
    print("Selected sources per split:")

    print(
        working_sources[
            "split"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print("Selected source labels:")

    print(
        working_sources[
            "label"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print("Split x label:")

    print(
        pd.crosstab(
            working_sources["split"],
            working_sources["label"],
        )
    )

    print()
    print(
        "Total source:",
        len(working_sources)
    )

    # =====================================================
    # CREATE OUTPUT ROOT
    # =====================================================

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    failed = []

    total_expected = (
        len(working_sources)
        * len(SNR_VALUES)
    )

    generated = 0

    # =====================================================
    # GENERATION LOOP
    # =====================================================

    for source_index, source in (
        working_sources
        .reset_index(drop=True)
        .iterrows()
    ):

        source_id = (
            source[
                "source_file_id"
            ]
        )

        source_split = (
            source[
                "split"
            ]
        )

        # =================================================
        # SPLIT POLICY
        # =================================================

        if source_split in {
            "train",
            "validation",
        }:

            available_rirs = (
                seen_rirs
            )

            expected_rir_split = (
                "seen"
            )

        elif source_split == "test":

            available_rirs = (
                unseen_rirs
            )

            expected_rir_split = (
                "unseen"
            )

        else:

            raise ValueError(
                f"Split tidak dikenali: "
                f"{source_split}"
            )

        # =================================================
        # SOURCE PATH
        # =================================================

        source_path = (
            ROOT
            / source[
                "file_path"
            ]
        )

        try:

            # =============================================
            # LOAD SOURCE
            # =============================================

            wav, source_sr = (
                load_audio(
                    source_path
                )
            )

            # =============================================
            # RESAMPLE SOURCE
            # =============================================

            wav = resample_audio(
                wav,
                original_sr=source_sr,
                target_sr=TARGET_SR,
            )

            # =============================================
            # GENERATE EACH SNR
            # =============================================

            for snr_index, snr_db in enumerate(
                SNR_VALUES
            ):

                # =========================================
                # CYCLIC RIR ASSIGNMENT
                # =========================================

                rir_index = (
                    source_index
                    + snr_index
                ) % len(
                    available_rirs
                )

                rir_row = (
                    available_rirs
                    .iloc[
                        rir_index
                    ]
                )

                rir_id = (
                    rir_row[
                        "rir_id"
                    ]
                )

                rir_room = (
                    rir_row[
                        "room"
                    ]
                )

                rir_split = (
                    rir_row[
                        "rir_split"
                    ]
                )

                # =========================================
                # SPLIT SAFETY
                # =========================================

                if (
                    rir_split
                    != expected_rir_split
                ):

                    raise RuntimeError(
                        "RIR split policy violation."
                    )

                # =========================================
                # LOAD RIR
                # =========================================

                rir_path = (
                    ROOT
                    / rir_row[
                        "file_path"
                    ]
                )

                rir, rir_sr = (
                    load_audio(
                        rir_path
                    )
                )

                if rir_sr != TARGET_SR:

                    rir = resample_audio(
                        rir,
                        original_sr=rir_sr,
                        target_sr=TARGET_SR,
                    )

                # =========================================
                # RIR CONVOLUTION
                # =========================================

                replay = apply_rir(
                    wav,
                    rir,
                )

                # =========================================
                # ADD NOISE
                # =========================================

                replay = add_noise_snr(
                    replay,
                    snr_db=snr_db,
                    target_peak=TARGET_PEAK,
                )

                # =========================================
                # QUALITY CONTROL
                # =========================================

                qc = inspect_output(
                    replay
                )

                if qc["silent"]:

                    raise RuntimeError(
                        "Generated replay is silent."
                    )

                if qc["clipped"]:

                    raise RuntimeError(
                        "Generated replay is clipping."
                    )

                # =========================================
                # FILE ID
                # =========================================

                file_id = (
                    f"{source_id}"
                    f"__rir-{rir_id}"
                    f"__snr-{snr_db}"
                )

                # =========================================
                # OUTPUT DIRECTORY
                # =========================================

                output_dir = (
                    OUTPUT_ROOT
                    / source_split
                )

                output_path = (
                    output_dir
                    / f"{file_id}.wav"
                )

                # =========================================
                # SAVE
                # =========================================

                save_audio(
                    output_path,
                    replay,
                )

                # =========================================
                # OUTPUT MANIFEST ROW
                # =========================================

                rows.append(
                    {
                        "file_id":
                            file_id,

                        "source_file_id":
                            source_id,

                        "file_path":
                            output_path
                            .relative_to(ROOT)
                            .as_posix(),

                        "source_file_path":
                            source[
                                "file_path"
                            ],

                        "speaker_id":
                            source[
                                "speaker_id"
                            ],

                        "attack_id":
                            source[
                                "attack_id"
                            ],

                        "label":
                            int(
                                source[
                                    "label"
                                ]
                            ),

                        "dataset":
                            source[
                                "dataset"
                            ],

                        "split":
                            source_split,

                        "condition":
                            "simulated_replay",

                        "rir_id":
                            rir_id,

                        "rir_room":
                            rir_room,

                        "rir_split":
                            rir_split,

                        "snr_db":
                            snr_db,

                        "sample_rate":
                            TARGET_SR,

                        "peak":
                            round(
                                qc["peak"],
                                8,
                            ),

                        "rms":
                            round(
                                qc["rms"],
                                8,
                            ),

                        "silent":
                            qc["silent"],

                        "clipped":
                            qc["clipped"],
                    }
                )

                generated += 1

        except Exception as e:

            failed.append(
                {
                    "source_file_id":
                        source_id,

                    "split":
                        source_split,

                    "error":
                        str(e),
                }
            )

            print()
            print(
                f"[ERROR] "
                f"{source_id}: {e}"
            )

        # =================================================
        # PROGRESS
        # =================================================

        if (
            (source_index + 1)
            % 10
            == 0
            or
            (
                source_index + 1
                == len(
                    working_sources
                )
            )
        ):

            print(
                f"Processed sources "
                f"{source_index + 1}/"
                f"{len(working_sources)} "
                f"| generated "
                f"{generated}/"
                f"{total_expected}"
            )

    # =====================================================
    # CREATE OUTPUT MANIFEST
    # =====================================================

    manifest = pd.DataFrame(
        rows
    )

    OUTPUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        OUTPUT_MANIFEST,
        index=False,
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    print()
    print("=" * 70)
    print("SIMULATED REPLAY SUMMARY")
    print("=" * 70)

    print()
    print("Source count:")
    print(
        len(
            working_sources
        )
    )

    print()
    print("Expected output:")
    print(
        total_expected
    )

    print()
    print("Generated output:")
    print(
        len(
            manifest
        )
    )

    print()
    print("Failed source:")
    print(
        len(
            failed
        )
    )

    if not manifest.empty:

        print()
        print("Output per split:")

        print(
            manifest[
                "split"
            ]
            .value_counts()
            .sort_index()
        )

        print()
        print("RIR split:")

        print(
            manifest[
                "rir_split"
            ]
            .value_counts()
            .sort_index()
        )

        print()
        print("SNR:")

        print(
            manifest[
                "snr_db"
            ]
            .value_counts()
            .sort_index()
        )

        print()
        print("Label:")

        print(
            manifest[
                "label"
            ]
            .value_counts()
            .sort_index()
        )

        print()
        print("Split x label:")

        print(
            pd.crosstab(
                manifest["split"],
                manifest["label"],
            )
        )

        print()
        print("Split x RIR split:")

        print(
            pd.crosstab(
                manifest["split"],
                manifest["rir_split"],
            )
        )

        print()
        print("RIR room:")

        print(
            manifest[
                "rir_room"
            ]
            .value_counts()
            .sort_index()
        )

        print()
        print("Peak statistics:")

        print(
            manifest[
                "peak"
            ]
            .describe()
        )

        print()
        print("RMS statistics:")

        print(
            manifest[
                "rms"
            ]
            .describe()
        )

        print()
        print("Silent outputs:")

        print(
            int(
                manifest[
                    "silent"
                ]
                .sum()
            )
        )

        print()
        print("Clipped outputs:")

        print(
            int(
                manifest[
                    "clipped"
                ]
                .sum()
            )
        )

    # =====================================================
    # HARD VALIDATION
    # =====================================================

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    # =====================================================
    # NO FAILED SOURCE
    # =====================================================

    assert len(failed) == 0, (
        f"Ada {len(failed)} "
        f"source gagal diproses."
    )

    # =====================================================
    # EXPECTED OUTPUT COUNT
    # =====================================================

    assert (
        len(manifest)
        == total_expected
    ), (
        "Jumlah output simulated replay "
        "tidak sesuai target."
    )

    # =====================================================
    # NO SILENCE
    # =====================================================

    assert (
        manifest[
            "silent"
        ].sum()
        == 0
    ), (
        "Terdapat simulated replay "
        "yang silent."
    )

    # =====================================================
    # NO CLIPPING
    # =====================================================

    assert (
        manifest[
            "clipped"
        ].sum()
        == 0
    ), (
        "Terdapat simulated replay "
        "yang clipping."
    )

    # =====================================================
    # PEAK SAFETY
    # =====================================================

    assert (
        manifest[
            "peak"
        ].max()
        <= TARGET_PEAK + 1e-5
    ), (
        "Peak output melebihi "
        "target peak."
    )

    # =====================================================
    # TRAIN → SEEN
    # =====================================================

    train_rows = (
        manifest[
            manifest["split"]
            == "train"
        ]
    )

    if len(train_rows) > 0:

        assert (
            train_rows[
                "rir_split"
            ]
            == "seen"
        ).all(), (
            "Train mengandung unseen RIR."
        )

    # =====================================================
    # VALIDATION → SEEN
    # =====================================================

    validation_rows = (
        manifest[
            manifest["split"]
            == "validation"
        ]
    )

    if len(validation_rows) > 0:

        assert (
            validation_rows[
                "rir_split"
            ]
            == "seen"
        ).all(), (
            "Validation mengandung unseen RIR."
        )

    # =====================================================
    # TEST → UNSEEN
    # =====================================================

    test_rows = (
        manifest[
            manifest["split"]
            == "test"
        ]
    )

    if len(test_rows) > 0:

        assert (
            test_rows[
                "rir_split"
            ]
            == "unseen"
        ).all(), (
            "Test mengandung seen RIR."
        )

    # =====================================================
    # SOURCE SPLIT PRESERVATION
    # =====================================================

    source_split_map = (
        working_sources
        .set_index(
            "source_file_id"
        )[
            "split"
        ]
        .to_dict()
    )

    for _, row in (
        manifest.iterrows()
    ):

        expected_split = (
            source_split_map[
                row[
                    "source_file_id"
                ]
            ]
        )

        assert (
            expected_split
            == row[
                "split"
            ]
        ), (
            "Source split leakage ditemukan."
        )

    # =====================================================
    # PILOT LABEL BALANCE
    # =====================================================

    if PILOT_MODE:

        source_label_counts = (
            working_sources[
                "label"
            ]
            .value_counts()
        )

        assert (
            source_label_counts
            .get(
                0,
                0
            )
            == 25
        ), (
            "Pilot bona fide "
            "tidak berjumlah 25."
        )

        assert (
            source_label_counts
            .get(
                1,
                0
            )
            == 25
        ), (
            "Pilot spoof "
            "tidak berjumlah 25."
        )

        output_label_counts = (
            manifest[
                "label"
            ]
            .value_counts()
        )

        assert (
            output_label_counts
            .get(
                0,
                0
            )
            == 75
        ), (
            "Output pilot bona fide "
            "harus 75."
        )

        assert (
            output_label_counts
            .get(
                1,
                0
            )
            == 75
        ), (
            "Output pilot spoof "
            "harus 75."
        )

    # =====================================================
    # RIR DIVERSITY
    # =====================================================

    seen_rooms_used = set(
        seen_rirs[
            "room"
        ]
    )

    unseen_rooms_used = set(
        unseen_rirs[
            "room"
        ]
    )

    assert (
        set(SEEN_ROOMS)
        .issubset(
            seen_rooms_used
        )
    ), (
        "Tidak seluruh seen room "
        "terwakili."
    )

    assert (
        set(UNSEEN_ROOMS)
        == unseen_rooms_used
    ), (
        "Unseen RIR harus mencakup "
        "booth dan office."
    )

    # =====================================================
    # FINAL
    # =====================================================

    print()
    print("VALIDATION PASSED")

    print()
    print("Audio saved to:")
    print(
        OUTPUT_ROOT
    )

    print()
    print("Manifest saved to:")
    print(
        OUTPUT_MANIFEST
    )

    print("=" * 70)

if __name__ == "__main__":
    main()