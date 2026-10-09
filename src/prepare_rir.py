from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torchaudio


# =========================================================
# PATH
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

RIR_AUDIT = ROOT / "manifests" / "rir_audit.csv"

OUTPUT_ROOT = ROOT / "data" / "processed" / "RIR"

SEEN_DIR = OUTPUT_ROOT / "seen"
UNSEEN_DIR = OUTPUT_ROOT / "unseen"

OUTPUT_MANIFEST = ROOT / "manifests" / "rir_manifest.csv"


# =========================================================
# CONFIGURATION
# =========================================================

TARGET_SR = 16000

# Pembagian room untuk eksperimen seen/unseen.
# Ini adalah implementasi kita berdasarkan tujuan generalization.
#
# Seen:
# - lecture
# - meeting
# - phone
# - stairway
#
# Unseen:
# - booth
# - office

SEEN_ROOMS = {
    "lecture",
    "meeting",
    "phone",
    "stairway",
}

UNSEEN_ROOMS = {
    "booth",
    "office",
}


# =========================================================
# LOAD RIR
# =========================================================

def load_rir(path: Path):
    """
    Membaca RIR menggunakan SoundFile.

    SoundFile menghasilkan array:
        [time, channels]

    Sedangkan PyTorch biasanya menggunakan:
        [channels, time]

    Oleh karena itu array ditranspose sebelum digunakan.
    """

    audio, sr = sf.read(
        str(path),
        always_2d=True,
        dtype="float32",
    )

    # [time, channels]
    # menjadi:
    # [channels, time]

    wav = torch.from_numpy(audio.T)

    # Safety:
    # hasil audit AIR menunjukkan seluruh file mono,
    # tetapi jika ada file multi-channel, ubah ke mono.

    if wav.shape[0] > 1:
        wav = wav.mean(
            dim=0,
            keepdim=True,
        )

    return wav, sr


# =========================================================
# RESAMPLE RIR
# =========================================================

def resample_rir(
    wav: torch.Tensor,
    original_sr: int,
    target_sr: int,
):
    """
    Resample waveform RIR ke target sample rate.

    Pada AIR:
        original_sr = 48000

    Target penelitian:
        16000 Hz
    """

    if original_sr == target_sr:
        return wav

    wav_resampled = torchaudio.functional.resample(
        wav,
        original_sr,
        target_sr,
    )

    return wav_resampled


# =========================================================
# SAVE RIR
# =========================================================

def save_rir(
    path: Path,
    wav: torch.Tensor,
    sample_rate: int,
):
    """
    Menyimpan RIR menggunakan SoundFile.

    File disimpan sebagai PCM 16-bit.
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
        sample_rate,
        subtype="PCM_16",
    )


# =========================================================
# DETERMINE SPLIT
# =========================================================

def determine_rir_split(room: str):
    """
    Menentukan apakah suatu room masuk:
        seen
    atau:
        unseen
    """

    if room in SEEN_ROOMS:
        return "seen"

    if room in UNSEEN_ROOMS:
        return "unseen"

    raise ValueError(
        f"Room belum memiliki split seen/unseen: {room}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("PREPARE RIR - SEEN / UNSEEN + RESAMPLE")
    print("=" * 70)

    # =====================================================
    # CHECK INPUT
    # =====================================================

    if not RIR_AUDIT.exists():
        raise FileNotFoundError(
            f"RIR audit tidak ditemukan:\n"
            f"{RIR_AUDIT}\n\n"
            f"Jalankan check_rir.py terlebih dahulu."
        )

    df = pd.read_csv(RIR_AUDIT)

    required_columns = {
        "rir_id",
        "file_path",
        "room",
        "sample_rate",
        "channels",
        "duration_s",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "rir_audit.csv tidak memiliki kolom: "
            f"{sorted(missing_columns)}"
        )

    # =====================================================
    # CREATE OUTPUT DIRECTORIES
    # =====================================================

    SEEN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    UNSEEN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =====================================================
    # PREPARE STORAGE
    # =====================================================

    rows = []

    failed = []

    # =====================================================
    # PROCESS EACH RIR
    # =====================================================

    print()
    print(f"Total input RIR: {len(df)}")
    print()

    for index, row in df.iterrows():

        rir_id = row["rir_id"]
        room = row["room"]

        try:

            # =============================================
            # DETERMINE SEEN / UNSEEN
            # =============================================

            rir_split = determine_rir_split(
                room
            )

            if rir_split == "seen":
                output_dir = SEEN_DIR

            else:
                output_dir = UNSEEN_DIR

            # =============================================
            # INPUT PATH
            # =============================================

            input_path = (
                ROOT
                / row["file_path"]
            )

            if not input_path.exists():
                raise FileNotFoundError(
                    f"File tidak ditemukan: {input_path}"
                )

            # =============================================
            # LOAD
            # =============================================

            wav, original_sr = load_rir(
                input_path
            )

            # =============================================
            # RESAMPLE
            # =============================================

            wav = resample_rir(
                wav=wav,
                original_sr=original_sr,
                target_sr=TARGET_SR,
            )

            # =============================================
            # OUTPUT PATH
            # =============================================

            output_path = (
                output_dir
                / f"{rir_id}.wav"
            )

            # =============================================
            # SAVE
            # =============================================

            save_rir(
                path=output_path,
                wav=wav,
                sample_rate=TARGET_SR,
            )

            # =============================================
            # VERIFY OUTPUT
            # =============================================

            info = sf.info(
                str(output_path)
            )

            duration_s = (
                info.frames
                / info.samplerate
            )

            # =============================================
            # MANIFEST ROW
            # =============================================

            rows.append(
                {
                    "rir_id": rir_id,

                    "source_file_path":
                        input_path
                        .relative_to(ROOT)
                        .as_posix(),

                    "file_path":
                        output_path
                        .relative_to(ROOT)
                        .as_posix(),

                    "room": room,

                    "rir_split": rir_split,

                    "original_sr": int(
                        original_sr
                    ),

                    "sample_rate": int(
                        info.samplerate
                    ),

                    "channels": int(
                        info.channels
                    ),

                    "duration_s": round(
                        duration_s,
                        6,
                    ),
                }
            )

        except Exception as e:

            failed.append(
                {
                    "rir_id": rir_id,
                    "room": room,
                    "error": str(e),
                }
            )

            print(
                f"[ERROR] {rir_id}: {e}"
            )

        # =============================================
        # OPTIONAL PROGRESS
        # =============================================

        if (
            (index + 1) % 50 == 0
            or
            (index + 1) == len(df)
        ):
            print(
                f"Processed "
                f"{index + 1}/{len(df)}"
            )

    # =====================================================
    # CREATE MANIFEST
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
    print("RIR PREPARATION SUMMARY")
    print("=" * 70)

    print()
    print("Total input RIR:")
    print(
        len(df)
    )

    print()
    print("Successfully processed:")
    print(
        len(manifest)
    )

    print()
    print("Failed:")
    print(
        len(failed)
    )

    if not manifest.empty:

        # =================================================
        # SPLIT DISTRIBUTION
        # =================================================

        print()
        print("RIR split:")
        print(
            manifest[
                "rir_split"
            ]
            .value_counts()
            .sort_index()
        )

        # =================================================
        # ROOM x SPLIT
        # =================================================

        print()
        print("Room x split:")
        print(
            pd.crosstab(
                manifest["room"],
                manifest["rir_split"],
            )
        )

        # =================================================
        # SAMPLE RATE
        # =================================================

        print()
        print("Sample rate:")
        print(
            manifest[
                "sample_rate"
            ]
            .value_counts()
            .sort_index()
        )

        # =================================================
        # CHANNEL
        # =================================================

        print()
        print("Channels:")
        print(
            manifest[
                "channels"
            ]
            .value_counts()
            .sort_index()
        )

        # =================================================
        # ORIGINAL SAMPLE RATE
        # =================================================

        print()
        print("Original sample rate:")
        print(
            manifest[
                "original_sr"
            ]
            .value_counts()
            .sort_index()
        )

        # =================================================
        # DURATION
        # =================================================

        print()
        print("Duration statistics:")
        print(
            manifest[
                "duration_s"
            ]
            .describe()
        )

    # =====================================================
    # HARD VALIDATION
    # =====================================================

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    # Semua RIR harus berhasil diproses
    assert len(failed) == 0, (
        f"Ada {len(failed)} RIR gagal diproses."
    )

    # Jumlah output harus sama dengan input
    assert len(manifest) == len(df), (
        "Jumlah RIR output tidak sama dengan input."
    )

    # Tidak boleh duplicate RIR ID
    assert (
        manifest["rir_id"]
        .nunique()
        == len(manifest)
    ), (
        "Duplicate RIR ID ditemukan."
    )

    # Semua harus 16 kHz
    assert (
        manifest[
            "sample_rate"
        ]
        == TARGET_SR
    ).all(), (
        "Tidak semua RIR berhasil menjadi 16 kHz."
    )

    # Semua harus mono
    assert (
        manifest[
            "channels"
        ]
        == 1
    ).all(), (
        "Tidak semua RIR mono."
    )

    # =====================================================
    # CHECK SEEN / UNSEEN LEAKAGE
    # =====================================================

    seen_rooms_in_manifest = set(
        manifest.loc[
            manifest["rir_split"]
            == "seen",
            "room",
        ].unique()
    )

    unseen_rooms_in_manifest = set(
        manifest.loc[
            manifest["rir_split"]
            == "unseen",
            "room",
        ].unique()
    )

    overlap = (
        seen_rooms_in_manifest
        &
        unseen_rooms_in_manifest
    )

    assert len(overlap) == 0, (
        f"Room leakage antara seen/unseen: "
        f"{sorted(overlap)}"
    )

    # Seen harus hanya berasal dari SEEN_ROOMS
    assert (
        seen_rooms_in_manifest
        <= SEEN_ROOMS
    ), (
        "Seen split memiliki room "
        "yang tidak terdaftar."
    )

    # Unseen harus hanya berasal dari UNSEEN_ROOMS
    assert (
        unseen_rooms_in_manifest
        <= UNSEEN_ROOMS
    ), (
        "Unseen split memiliki room "
        "yang tidak terdaftar."
    )

    # =====================================================
    # CHECK MINIMUM REQUIREMENT
    # =====================================================

    seen_count = (
        manifest[
            manifest["rir_split"]
            == "seen"
        ]
        .shape[0]
    )

    unseen_count = (
        manifest[
            manifest["rir_split"]
            == "unseen"
        ]
        .shape[0]
    )

    assert seen_count >= 5, (
        "Panduan membutuhkan minimal "
        "5 RIR seen."
    )

    assert unseen_count >= 2, (
        "Panduan membutuhkan minimal "
        "2 RIR unseen."
    )

    # =====================================================
    # FINAL
    # =====================================================

    print()
    print("VALIDATION PASSED")

    print()
    print("Seen RIR:")
    print(
        seen_count
    )

    print()
    print("Unseen RIR:")
    print(
        unseen_count
    )

    print()
    print("Processed RIR directory:")
    print(
        OUTPUT_ROOT
    )

    print()
    print("RIR manifest saved to:")
    print(
        OUTPUT_MANIFEST
    )

    print("=" * 70)

if __name__ == "__main__":
    main()