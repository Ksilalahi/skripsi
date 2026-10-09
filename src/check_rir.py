from pathlib import Path

import pandas as pd
import soundfile as sf


# =========================================================
# PATH
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

RIR_ROOT = ROOT / "data" / "raw" / "RIR" / "AIR_wav_files"

OUTPUT_CSV = ROOT / "manifests" / "rir_audit.csv"


# =========================================================
# HELPER
# =========================================================

def infer_room(filename: str) -> str:
    """
    Mengambil kategori room dari nama file AIR.

    Contoh:
    air_binaural_stairway_...
    air_booth_...
    air_lecture_...
    air_meeting_...
    air_office_...
    air_phone_...
    """

    name = filename.lower()

    known_rooms = [
        "booth",
        "lecture",
        "meeting",
        "office",
        "phone",
        "stairway",
    ]

    for room in known_rooms:
        if room in name:
            return room

    return "unknown"


def inspect_rir(path: Path):
    """
    Membaca metadata WAV RIR tanpa memuat seluruh audio.
    """

    info = sf.info(str(path))

    duration_s = info.frames / info.samplerate

    return {
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration_s": duration_s,
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("AACHEN AIR DATABASE AUDIT")
    print("=" * 70)

    if not RIR_ROOT.exists():
        raise FileNotFoundError(
            f"Folder RIR tidak ditemukan:\n{RIR_ROOT}"
        )

    wav_files = sorted(
        RIR_ROOT.rglob("*.wav")
    )

    print()
    print("RIR root:")
    print(RIR_ROOT)

    print()
    print("Total WAV files:")
    print(len(wav_files))

    if len(wav_files) == 0:
        raise RuntimeError(
            "Tidak ditemukan file .wav di AIR_wav_files."
        )

    rows = []
    failed = []

    for path in wav_files:

        try:
            info = inspect_rir(path)

        except Exception as e:

            failed.append(
                {
                    "file": str(path),
                    "error": str(e),
                }
            )

            continue

        room = infer_room(path.name)

        rows.append(
            {
                "rir_id": path.stem,
                "file_path": path.relative_to(ROOT).as_posix(),
                "room": room,
                "sample_rate": info["sample_rate"],
                "channels": info["channels"],
                "frames": info["frames"],
                "duration_s": round(
                    info["duration_s"],
                    6
                ),
            }
        )

    df = pd.DataFrame(rows)

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    print()
    print("=" * 70)
    print("RIR SUMMARY")
    print("=" * 70)

    print()
    print("Valid RIR:")
    print(len(df))

    print()
    print("Failed RIR:")
    print(len(failed))

    print()
    print("RIR per room:")
    print(
        df["room"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Sample rate:")
    print(
        df["sample_rate"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Channels:")
    print(
        df["channels"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Duration statistics:")
    print(
        df["duration_s"]
        .describe()
    )

    print()
    print("Unknown room:")
    print(
        (df["room"] == "unknown").sum()
    )

    print()
    print("Audit saved to:")
    print(OUTPUT_CSV)

    print("=" * 70)


if __name__ == "__main__":
    main()