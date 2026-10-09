from pathlib import Path
import hashlib

import pandas as pd
import soundfile as sf


# =========================================================
# PATH PROJECT
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = ROOT / "data" / "raw" / "ASVspoof2019_LA"

TRAIN_AUDIO_DIR = DATASET_ROOT / "ASVspoof2019_LA_train" / "flac"
DEV_AUDIO_DIR = DATASET_ROOT / "ASVspoof2019_LA_dev" / "flac"
EVAL_AUDIO_DIR = DATASET_ROOT / "ASVspoof2019_LA_eval" / "flac"

PROTOCOL_DIR = DATASET_ROOT / "ASVspoof2019_LA_cm_protocols"

TRAIN_PROTOCOL = PROTOCOL_DIR / "ASVspoof2019.LA.cm.train.trn.txt"
DEV_PROTOCOL = PROTOCOL_DIR / "ASVspoof2019.LA.cm.dev.trl.txt"
EVAL_PROTOCOL = PROTOCOL_DIR / "ASVspoof2019.LA.cm.eval.trl.txt"

OUTPUT_MANIFEST = ROOT / "manifests" / "source_manifest.csv"
OUTPUT_MISSING = ROOT / "manifests" / "missing_protocol_files.txt"

MIN_DURATION_S = 1.0


# =========================================================
# SHA256
# =========================================================

def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """
    Menghitung SHA256 file untuk identitas dan pengecekan duplikasi.
    """

    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(block_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# =========================================================
# AUDIO INSPECTION
# =========================================================

def inspect_audio(path: Path):
    """
    Membaca metadata audio tanpa memuat seluruh waveform.
    """

    info = sf.info(str(path))

    duration_s = info.frames / info.samplerate

    return {
        "duration_s": duration_s,
        "sample_rate": info.samplerate,
        "channels": info.channels,
    }


# =========================================================
# READ PROTOCOL
# =========================================================

def read_protocol(
    protocol_path: Path,
    split_name: str,
    audio_dir: Path,
):
    """
    Membaca protocol resmi ASVspoof2019 LA.

    Format utama protocol:
    speaker_id file_id - attack_id key
    """

    rows = []
    missing_files = []
    too_short = []

    with protocol_path.open("r", encoding="utf-8") as f:

        for line_number, raw_line in enumerate(f, start=1):

            line = raw_line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                print(
                    f"[WARNING] Format protocol tidak sesuai "
                    f"pada baris {line_number}: {line}"
                )
                continue

            speaker_id = parts[0]
            file_id = parts[1]
            attack_id = parts[3]
            key = parts[4].lower()

            # =================================================
            # LABEL MAPPING
            # =================================================

            if key == "bonafide":
                label = 0

            elif key == "spoof":
                label = 1

            else:
                print(
                    f"[WARNING] Label tidak dikenali "
                    f"pada {file_id}: {key}"
                )
                continue

            # Untuk bonafide biasanya attack_id = "-"
            if attack_id == "-":
                attack_id = "bonafide"

            # =================================================
            # AUDIO FILE
            # =================================================

            audio_path = audio_dir / f"{file_id}.flac"

            if not audio_path.exists():

                missing_files.append(
                    {
                        "split": split_name,
                        "file_id": file_id,
                        "expected_path": str(audio_path),
                    }
                )

                continue

            # =================================================
            # AUDIO AUDIT
            # =================================================

            try:
                info = inspect_audio(audio_path)

            except Exception as e:

                print(
                    f"[ERROR] Gagal membaca "
                    f"{audio_path.name}: {e}"
                )

                continue

            duration_s = info["duration_s"]

            # Panduan meminta audio < 1 detik difilter
            if duration_s < MIN_DURATION_S:

                too_short.append(
                    {
                        "split": split_name,
                        "file_id": file_id,
                        "duration_s": duration_s,
                    }
                )

                continue

            # =================================================
            # MANIFEST ROW
            # =================================================

            relative_path = audio_path.relative_to(ROOT)

            rows.append(
                {
                    "source_file_id": file_id,
                    "file_path": relative_path.as_posix(),
                    "speaker_id": speaker_id,
                    "attack_id": attack_id,
                    "label": label,
                    "dataset": "ASVspoof2019_LA",
                    "split": split_name,
                    "duration_s": round(duration_s, 6),
                    "sha256": sha256_file(audio_path),
                }
            )

    return rows, missing_files, too_short


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("BUILD SOURCE MANIFEST - ASVspoof2019 LA")
    print("=" * 70)

    datasets = [
        (
            "TRAIN",
            TRAIN_PROTOCOL,
            "train",
            TRAIN_AUDIO_DIR,
        ),
        (
            "DEV",
            DEV_PROTOCOL,
            "validation",
            DEV_AUDIO_DIR,
        ),
        (
            "EVAL",
            EVAL_PROTOCOL,
            "test",
            EVAL_AUDIO_DIR,
        ),
    ]

    all_rows = []
    all_missing = []
    all_too_short = []

    # =====================================================
    # CHECK PATH
    # =====================================================

    print("\nChecking dataset paths...\n")

    for name, protocol, split, audio_dir in datasets:

        print(f"{name}")
        print(f"  Protocol : {protocol}")
        print(f"  Audio    : {audio_dir}")

        if not protocol.exists():
            raise FileNotFoundError(
                f"Protocol tidak ditemukan:\n{protocol}"
            )

        if not audio_dir.exists():
            raise FileNotFoundError(
                f"Folder audio tidak ditemukan:\n{audio_dir}"
            )

    # =====================================================
    # PROCESS
    # =====================================================

    for name, protocol, split, audio_dir in datasets:

        print()
        print("-" * 70)
        print(f"Processing {name}")
        print("-" * 70)

        rows, missing, too_short = read_protocol(
            protocol_path=protocol,
            split_name=split,
            audio_dir=audio_dir,
        )

        print(f"Valid audio : {len(rows)}")
        print(f"Missing     : {len(missing)}")
        print(f"< 1 second  : {len(too_short)}")

        all_rows.extend(rows)
        all_missing.extend(missing)
        all_too_short.extend(too_short)

    # =====================================================
    # CREATE MANIFEST
    # =====================================================

    df = pd.DataFrame(all_rows)

    OUTPUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_MANIFEST,
        index=False,
    )

    # =====================================================
    # SAVE MISSING FILES
    # =====================================================

    if all_missing:

        missing_df = pd.DataFrame(all_missing)

        missing_df.to_csv(
            OUTPUT_MISSING,
            index=False,
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    print()
    print("=" * 70)
    print("MANIFEST SUMMARY")
    print("=" * 70)

    print(f"Total rows       : {len(df)}")
    print(f"Missing files    : {len(all_missing)}")
    print(f"Audio < 1 second : {len(all_too_short)}")

    if not df.empty:

        print()
        print("Rows per split:")
        print(
            df["split"]
            .value_counts()
            .sort_index()
        )

        print()
        print("Rows per label:")
        print(
            df["label"]
            .value_counts()
            .sort_index()
        )

        print()
        print("Split x label:")
        print(
            df.groupby(
                ["split", "label"]
            )
            .size()
        )

        print()
        print("Unique source IDs:")
        print(
            df["source_file_id"].nunique()
        )

        print()
        print("Duplicate source IDs:")
        duplicate_ids = (
            df["source_file_id"]
            .duplicated()
            .sum()
        )

        print(duplicate_ids)

        print()
        print("Duplicate SHA256:")
        duplicate_sha = (
            df["sha256"]
            .duplicated()
            .sum()
        )

        print(duplicate_sha)

    print()
    print(f"Manifest saved to:")
    print(OUTPUT_MANIFEST)

    if all_missing:
        print()
        print("Missing-file report saved to:")
        print(OUTPUT_MISSING)

    print("=" * 70)


if __name__ == "__main__":
    main()