from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
import torch
import torchaudio


# =========================================================
# PATH
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

ACQUISITION_PLAN = (
    ROOT
    / "manifests"
    / "acquisition_plan.csv"
)

REAL_REPLAY_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "real_replay"
)

MASTER_DIR = (
    REAL_REPLAY_ROOT
    / "master_48k24"
)

MODEL_DIR = (
    REAL_REPLAY_ROOT
    / "model_16k16"
)

REAL_MANIFEST = (
    ROOT
    / "manifests"
    / "manifest_real_replay.csv"
)


# =========================================================
# AUDIO CONFIGURATION
# =========================================================

MASTER_SR = 48000
MODEL_SR = 16000

MASTER_SUBTYPE = "PCM_24"
MODEL_SUBTYPE = "PCM_16"

CHANNELS = 1

PRE_ROLL_SECONDS = 0.5
POST_ROLL_SECONDS = 0.5

PLAYBACK_TARGET_PEAK = 0.95

SILENCE_RMS_THRESHOLD = 1e-6
CLIP_THRESHOLD = 0.999


# =========================================================
# LOAD SOURCE
# =========================================================

def load_source(path: Path):
    """
    Membaca source audio menggunakan SoundFile.

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

    # Safety:
    # ubah ke mono jika source memiliki >1 channel
    if wav.shape[0] > 1:

        wav = wav.mean(
            dim=0,
            keepdim=True,
        )

    return wav, sr


# =========================================================
# RESAMPLE
# =========================================================

def resample_audio(
    wav: torch.Tensor,
    source_sr: int,
    target_sr: int,
):
    """
    Resample waveform jika sample rate berbeda.
    """

    if source_sr == target_sr:
        return wav

    return torchaudio.functional.resample(
        wav,
        source_sr,
        target_sr,
    )


# =========================================================
# PREPARE PLAYBACK
# =========================================================

def prepare_playback(
    wav: torch.Tensor,
    source_sr: int,
):
    """
    Menyiapkan source untuk playback pada 48 kHz.

    File source asli tidak diubah.
    Resampling hanya dilakukan di memory.
    """

    # =====================================================
    # RESAMPLE TO MASTER SR
    # =====================================================

    wav = resample_audio(
        wav,
        source_sr,
        MASTER_SR,
    )

    # =====================================================
    # PEAK SAFETY
    # =====================================================

    peak = (
        wav
        .abs()
        .max()
        .clamp_min(1e-8)
    )

    if peak > PLAYBACK_TARGET_PEAK:

        wav = (
            wav
            / peak
            * PLAYBACK_TARGET_PEAK
        )

    # =====================================================
    # PyTorch -> NumPy
    # =====================================================
    #
    # PyTorch:
    # [channels, time]
    #
    # sounddevice:
    # [samples, channels]
    # =====================================================

    playback = (
        wav
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    playback = playback.reshape(
        -1,
        1,
    )

    return playback


# =========================================================
# VALIDATE AUDIO DEVICES
# =========================================================

def validate_audio_devices(
    input_device: int,
    output_device: int,
):
    """
    Memastikan input dan output device menerima
    konfigurasi 48 kHz mono float32.
    """

    sd.check_input_settings(
        device=input_device,
        channels=CHANNELS,
        samplerate=MASTER_SR,
        dtype="float32",
    )

    sd.check_output_settings(
        device=output_device,
        channels=CHANNELS,
        samplerate=MASTER_SR,
        dtype="float32",
    )


# =========================================================
# PLAY + RECORD
# =========================================================

def perform_playback_recording(
    playback: np.ndarray,
    input_device: int,
    output_device: int,
):
    """
    Memutar source melalui output device dan merekam
    secara bersamaan melalui input device.

    Untuk full-duplex sounddevice:
        device=(input_device, output_device)
    """

    # =====================================================
    # PRE / POST ROLL
    # =====================================================

    pre_samples = int(
        PRE_ROLL_SECONDS
        * MASTER_SR
    )

    post_samples = int(
        POST_ROLL_SECONDS
        * MASTER_SR
    )

    silence_pre = np.zeros(
        (
            pre_samples,
            CHANNELS,
        ),
        dtype=np.float32,
    )

    silence_post = np.zeros(
        (
            post_samples,
            CHANNELS,
        ),
        dtype=np.float32,
    )

    # =====================================================
    # PLAYBACK BUFFER
    # =====================================================

    playback_full = np.concatenate(
        [
            silence_pre,
            playback,
            silence_post,
        ],
        axis=0,
    )

    # =====================================================
    # FULL-DUPLEX PLAY + RECORD
    # =====================================================

    recording = sd.playrec(
        playback_full,
        samplerate=MASTER_SR,
        channels=CHANNELS,
        dtype="float32",
        device=(
            input_device,
            output_device,
        ),
        blocking=True,
    )

    return recording


# =========================================================
# QUALITY CONTROL
# =========================================================

def compute_qc(audio: np.ndarray):
    """
    Menghitung QC dasar recording:
        peak
        rms
        finite
        silent
        clipped
    """

    if audio.size == 0:

        raise ValueError(
            "Recording kosong."
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
        rms
        < SILENCE_RMS_THRESHOLD
    )

    clipped = (
        peak
        >= CLIP_THRESHOLD
    )

    return {
        "peak": peak,
        "rms": rms,
        "finite": finite,
        "silent": silent,
        "clipped": clipped,
    }


# =========================================================
# SAVE MASTER
# =========================================================

def save_master(
    path: Path,
    recording: np.ndarray,
):
    """
    Simpan master sebagai:
        48 kHz
        PCM 24-bit
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        str(path),
        recording,
        MASTER_SR,
        subtype=MASTER_SUBTYPE,
    )


# =========================================================
# CREATE MODEL COPY
# =========================================================

def create_model_copy(
    master_path: Path,
    model_path: Path,
):
    """
    Membuat turunan untuk model:
        16 kHz
        PCM 16-bit

    Master 48 kHz / 24-bit tetap disimpan.
    """

    audio, sr = sf.read(
        str(master_path),
        always_2d=True,
        dtype="float32",
    )

    wav = torch.from_numpy(
        audio.T
    )

    # Safety mono
    if wav.shape[0] > 1:

        wav = wav.mean(
            dim=0,
            keepdim=True,
        )

    # =====================================================
    # RESAMPLE MASTER -> MODEL
    # =====================================================

    wav = resample_audio(
        wav,
        sr,
        MODEL_SR,
    )

    # =====================================================
    # PYTORCH -> NUMPY
    # =====================================================

    audio_16k = (
        wav
        .squeeze(0)
        .cpu()
        .numpy()
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        str(model_path),
        audio_16k,
        MODEL_SR,
        subtype=MODEL_SUBTYPE,
    )


# =========================================================
# APPEND MANIFEST
# =========================================================

def append_manifest(row: dict):
    """
    Tambahkan satu recording ke:
        manifest_real_replay.csv

    Recording ID tidak boleh duplicate.
    """

    new_row = pd.DataFrame(
        [row]
    )

    # =====================================================
    # EXISTING MANIFEST
    # =====================================================

    if REAL_MANIFEST.exists():

        old = pd.read_csv(
            REAL_MANIFEST
        )

        # =================================================
        # DUPLICATE RECORDING PROTECTION
        # =================================================

        if (
            "recording_id"
            in old.columns
            and
            row["recording_id"]
            in set(
                old[
                    "recording_id"
                ]
            )
        ):

            raise RuntimeError(
                "Recording ID sudah ada "
                "di manifest:\n"
                f"{row['recording_id']}"
            )

        combined = pd.concat(
            [
                old,
                new_row,
            ],
            ignore_index=True,
        )

    else:

        combined = new_row

    # =====================================================
    # SAVE MANIFEST
    # =====================================================

    REAL_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        REAL_MANIFEST,
        index=False,
    )


# =========================================================
# MAIN
# =========================================================

def main():

    # =====================================================
    # ARGUMENTS
    # =====================================================

    parser = argparse.ArgumentParser(
        description=(
            "Controlled physical replay "
            "recording tool"
        )
    )

    parser.add_argument(
        "--row",
        type=int,
        required=True,
        help=(
            "Nomor row acquisition_plan.csv "
            "(dimulai dari 0)"
        ),
    )

    parser.add_argument(
        "--input-device",
        type=int,
        required=True,
        help=(
            "sounddevice input device ID"
        ),
    )

    parser.add_argument(
        "--output-device",
        type=int,
        required=True,
        help=(
            "sounddevice output device ID"
        ),
    )

    parser.add_argument(
        "--noise-floor-db",
        type=float,
        default=np.nan,
        help=(
            "Noise floor ruangan jika tersedia. "
            "Contoh: -43.5"
        ),
    )

    parser.add_argument(
        "--level",
        type=str,
        default="fixed",
        help=(
            "Catatan playback level. "
            "Contoh: fixed_70_percent"
        ),
    )

    parser.add_argument(
        "--agc",
        type=str,
        default="unknown",
        choices=[
            "disabled",
            "enabled",
            "unknown",
        ],
        help=(
            "Status automatic gain control "
            "pada recording device"
        ),
    )

    args = parser.parse_args()

    # =====================================================
    # CHECK ACQUISITION PLAN
    # =====================================================

    if not ACQUISITION_PLAN.exists():

        raise FileNotFoundError(
            "Acquisition plan tidak ditemukan:\n"
            f"{ACQUISITION_PLAN}"
        )

    plan = pd.read_csv(
        ACQUISITION_PLAN
    )

    # =====================================================
    # REQUIRED PLAN COLUMNS
    # =====================================================

    required_columns = {
        "source_file_id",
        "file_path",
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

    missing_columns = (
        required_columns
        - set(
            plan.columns
        )
    )

    if missing_columns:

        raise ValueError(
            "acquisition_plan.csv kehilangan kolom: "
            f"{sorted(missing_columns)}"
        )

    # =====================================================
    # CHECK ROW INDEX
    # =====================================================

    if (
        args.row < 0
        or
        args.row >= len(plan)
    ):

        raise IndexError(
            "Row harus antara 0 dan "
            f"{len(plan) - 1}"
        )

    row = plan.iloc[
        args.row
    ]

    # =====================================================
    # DEVICE INFORMATION
    # =====================================================

    input_info = sd.query_devices(
        args.input_device
    )

    output_info = sd.query_devices(
        args.output_device
    )

    input_device_name = (
        input_info["name"]
    )

    output_device_name = (
        output_info["name"]
    )

    # =====================================================
    # DISPLAY CONDITION
    # =====================================================

    print("=" * 70)
    print("CONTROLLED REAL REPLAY")
    print("=" * 70)

    print()
    print(
        "Acquisition row:",
        args.row,
    )

    print()

    for column in plan.columns:

        print(
            f"{column}: "
            f"{row[column]}"
        )

    print()

    print(
        f"Input device : "
        f"{args.input_device} "
        f"({input_device_name})"
    )

    print(
        f"Output device: "
        f"{args.output_device} "
        f"({output_device_name})"
    )

    # =====================================================
    # VALIDATE DEVICE SETTINGS
    # =====================================================

    validate_audio_devices(
        input_device=args.input_device,
        output_device=args.output_device,
    )

    print()
    print(
        "Audio device validation: PASSED"
    )

    # =====================================================
    # SOURCE PATH
    # =====================================================

    source_path = (
        ROOT
        / row[
            "file_path"
        ]
    )

    if not source_path.exists():

        raise FileNotFoundError(
            "Source audio tidak ditemukan:\n"
            f"{source_path}"
        )

    # =====================================================
    # RECORDING IDENTIFIERS
    # =====================================================

    source_id = (
        row[
            "source_file_id"
        ]
    )

    config_id = (
        row[
            "config_id"
        ]
    )

    repetition = int(
        row[
            "repetition"
        ]
    )

    recording_id = (
        f"{source_id}"
        f"__{config_id}"
        f"__rep{repetition}"
    )

    # =====================================================
    # OUTPUT PATHS
    # =====================================================

    master_path = (
        MASTER_DIR
        / str(config_id)
        / f"{recording_id}.wav"
    )

    model_path = (
        MODEL_DIR
        / str(config_id)
        / f"{recording_id}.wav"
    )

    # =====================================================
    # DUPLICATE PROTECTION
    # =====================================================

    if master_path.exists():

        raise FileExistsError(
            "Master recording sudah ada:\n"
            f"{master_path}\n\n"
            "Gunakan row lain atau hapus file "
            "uji tersebut jika memang ingin "
            "direkam ulang."
        )

    if model_path.exists():

        raise FileExistsError(
            "Model copy sudah ada:\n"
            f"{model_path}"
        )

    if REAL_MANIFEST.exists():

        existing_manifest = pd.read_csv(
            REAL_MANIFEST
        )

        if (
            "recording_id"
            in existing_manifest.columns
            and
            recording_id
            in set(
                existing_manifest[
                    "recording_id"
                ]
            )
        ):

            raise RuntimeError(
                "Recording ID sudah terdapat "
                "pada manifest:\n"
                f"{recording_id}"
            )

    # =====================================================
    # LOAD SOURCE
    # =====================================================

    wav, source_sr = load_source(
        source_path
    )

    # =====================================================
    # PREPARE PLAYBACK
    # =====================================================

    playback = prepare_playback(
        wav,
        source_sr,
    )

    source_duration = (
        wav.shape[-1]
        / source_sr
    )

    print()
    print(
        f"Source SR      : "
        f"{source_sr}"
    )

    print(
        f"Source duration: "
        f"{source_duration:.3f} s"
    )

    print(
        f"Recording SR   : "
        f"{MASTER_SR}"
    )

    print(
        f"Master subtype : "
        f"{MASTER_SUBTYPE}"
    )

    print(
        f"Model SR       : "
        f"{MODEL_SR}"
    )

    print(
        f"Model subtype  : "
        f"{MODEL_SUBTYPE}"
    )

    # =====================================================
    # DISPLAY PHYSICAL CONDITION
    # =====================================================

    print()
    print("-" * 70)
    print("PHYSICAL CONDITION REQUIRED")
    print("-" * 70)

    print(
        f"Config       : "
        f"{row['config_id']}"
    )

    print(
        f"Room         : "
        f"{row['room']}"
    )

    print(
        f"Playback     : "
        f"{row['playback']}"
    )

    print(
        f"Microphone   : "
        f"{row['mic']}"
    )

    print(
        f"Distance     : "
        f"{row['distance']} m"
    )

    print(
        f"Angle        : "
        f"{row['angle']} deg"
    )

    print(
        f"Repetition   : "
        f"{repetition}"
    )

    print(
        f"Level note   : "
        f"{args.level}"
    )

    print(
        f"AGC status   : "
        f"{args.agc}"
    )

    print(
        f"Noise floor  : "
        f"{args.noise_floor_db}"
    )

    print("-" * 70)

    # =====================================================
    # WAIT FOR USER
    # =====================================================

    input(
        "\nPastikan kondisi fisik sudah benar. "
        "Tekan ENTER untuk mulai recording..."
    )

    print()
    print("Recording dimulai...")

    # Short preparation delay
    time.sleep(1)

    # =====================================================
    # PLAY + RECORD
    # =====================================================

    recording = (
        perform_playback_recording(
            playback=playback,
            input_device=args.input_device,
            output_device=args.output_device,
        )
    )

    print(
        "Recording selesai."
    )

    # =====================================================
    # QC
    # =====================================================

    qc = compute_qc(
        recording
    )

    recording_duration = (
        recording.shape[0]
        / MASTER_SR
    )

    print()
    print("=" * 70)
    print("QC")
    print("=" * 70)

    print(
        f"Duration: "
        f"{recording_duration:.3f} s"
    )

    print(
        f"Peak    : "
        f"{qc['peak']:.6f}"
    )

    print(
        f"RMS     : "
        f"{qc['rms']:.6f}"
    )

    print(
        f"Finite  : "
        f"{qc['finite']}"
    )

    print(
        f"Silent  : "
        f"{qc['silent']}"
    )

    print(
        f"Clipped : "
        f"{qc['clipped']}"
    )

    # =====================================================
    # HARD QC
    # =====================================================

    if not qc["finite"]:

        raise RuntimeError(
            "Recording mengandung NaN/Inf. "
            "Recording tidak disimpan."
        )

    if qc["silent"]:

        raise RuntimeError(
            "Recording terdeteksi silent. "
            "Periksa microphone/input device. "
            "Recording tidak disimpan."
        )

    if qc["clipped"]:

        raise RuntimeError(
            "Recording terdeteksi clipping. "
            "Kurangi playback/input level lalu "
            "rekam ulang. Recording tidak disimpan."
        )

    print()
    print(
        "Basic recording QC: PASSED"
    )

    # =====================================================
    # SAVE MASTER
    # =====================================================

    save_master(
        master_path,
        recording,
    )

    # =====================================================
    # VERIFY MASTER
    # =====================================================

    master_info = sf.info(
        str(master_path)
    )

    if (
        master_info.samplerate
        != MASTER_SR
    ):

        raise RuntimeError(
            "Master sample rate tidak sesuai."
        )

    # =====================================================
    # CREATE MODEL COPY
    # =====================================================

    create_model_copy(
        master_path,
        model_path,
    )

    # =====================================================
    # VERIFY MODEL COPY
    # =====================================================

    model_info = sf.info(
        str(model_path)
    )

    if (
        model_info.samplerate
        != MODEL_SR
    ):

        raise RuntimeError(
            "Model copy sample rate "
            "tidak sesuai."
        )

    # =====================================================
    # MANIFEST ROW
    # =====================================================

    manifest_row = {
        "recording_id":
            recording_id,

        "acquisition_row":
            int(
                args.row
            ),

        "source_file_id":
            source_id,

        "source_file_path":
            row[
                "file_path"
            ],

        "master_file_path":
            master_path
            .relative_to(ROOT)
            .as_posix(),

        "model_file_path":
            model_path
            .relative_to(ROOT)
            .as_posix(),

        "speaker_id":
            (
                row["speaker_id"]
                if "speaker_id"
                in row.index
                else ""
            ),

        "attack_id":
            (
                row["attack_id"]
                if "attack_id"
                in row.index
                else ""
            ),

        "label":
            int(
                row[
                    "label"
                ]
            ),

        "dataset":
            (
                row["dataset"]
                if "dataset"
                in row.index
                else ""
            ),

        "split":
            row[
                "split"
            ],

        "config_id":
            config_id,

        "room":
            row[
                "room"
            ],

        "playback_device":
            row[
                "playback"
            ],

        "microphone":
            row[
                "mic"
            ],

        "distance_m":
            float(
                row[
                    "distance"
                ]
            ),

        "angle_deg":
            float(
                row[
                    "angle"
                ]
            ),

        "repetition":
            repetition,

        "input_device_id":
            args.input_device,

        "input_device_name":
            input_device_name,

        "output_device_id":
            args.output_device,

        "output_device_name":
            output_device_name,

        "noise_floor_db":
            args.noise_floor_db,

        "level":
            args.level,

        "agc":
            args.agc,

        "master_sample_rate":
            MASTER_SR,

        "master_bit_depth":
            24,

        "master_subtype":
            MASTER_SUBTYPE,

        "model_sample_rate":
            MODEL_SR,

        "model_bit_depth":
            16,

        "model_subtype":
            MODEL_SUBTYPE,

        "recording_duration_s":
            round(
                recording_duration,
                6,
            ),

        "peak":
            qc[
                "peak"
            ],

        "rms":
            qc[
                "rms"
            ],

        "finite":
            qc[
                "finite"
            ],

        "silent":
            qc[
                "silent"
            ],

        "clipped":
            qc[
                "clipped"
            ],
    }

    # =====================================================
    # APPEND MANIFEST
    # =====================================================

    append_manifest(
        manifest_row
    )

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print()
    print("=" * 70)
    print("RECORDING SAVED")
    print("=" * 70)

    print()
    print("Recording ID:")
    print(
        recording_id
    )

    print()
    print("Master:")
    print(
        master_path
    )

    print()
    print(
        f"Master format: "
        f"{MASTER_SR} Hz / "
        f"{MASTER_SUBTYPE}"
    )

    print()
    print("Model copy:")
    print(
        model_path
    )

    print()
    print(
        f"Model format: "
        f"{MODEL_SR} Hz / "
        f"{MODEL_SUBTYPE}"
    )

    print()
    print("Manifest:")
    print(
        REAL_MANIFEST
    )

    print()
    print(
        "CONTROLLED REAL REPLAY RECORDING: PASSED"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()