"""Validate simulated replay V2 against its frozen plan and actual WAV files.

Place under PROJECT_ROOT/src/validate_simulated_replay_v2.py then run:
  python src/validate_simulated_replay_v2.py

Read-only; does not modify the original plan, WAVs, manifests, or checkpoints.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

EXPECTED = {"train": 2400, "validation": 300, "test": 300}
SEEN = {"lecture", "meeting", "phone", "stairway"}
UNSEEN = {"booth", "office"}
EXPECTED_SNRS = {5, 10, 20}
REQUIRED = {
    "file_id", "file_path", "source_file_id", "label", "split", "condition",
    "rir_id", "rir_room", "rir_split", "snr_db", "sample_rate",
    "peak", "rms", "silent", "clipped", "duration_s",
}


def verify(root: Path, *, smoke: bool):
    manifest_file = root / "manifests" / ("manifest_replay_sim_v2_smoke.csv" if smoke else "manifest_replay_sim_v2.csv")
    plan_file = root / "manifests" / "plan_replay_sim_v2.csv"
    if not manifest_file.is_file() or not plan_file.is_file():
        raise FileNotFoundError(f"File tidak ada: {manifest_file} atau {plan_file}")
    d = pd.read_csv(manifest_file, keep_default_na=False)
    p = pd.read_csv(plan_file, keep_default_na=False)
    missing = REQUIRED - set(d.columns)
    if missing:
        raise ValueError(f"Kolom manifest hilang: {sorted(missing)}")
    expect = {x: 4 for x in EXPECTED} if smoke else EXPECTED
    errors = []

    def check(ok, msg):
        if not bool(ok):
            errors.append(msg)

    counts = d["split"].value_counts().to_dict()
    print(f"Manifest: {manifest_file}\nTotal audio: {len(d)}")
    print("Counts by split:", counts)
    check(counts == expect, f"Jumlah split salah: ditemukan {counts}, target {expect}")
    check(not d["file_id"].duplicated().any(), "file_id duplikat")
    check(not d["file_path"].duplicated().any(), "file_path duplikat")
    check(d["condition"].astype(str).eq("simulated_replay").all(), "Condition bukan simulated_replay")
    check(set(pd.to_numeric(d["label"], errors="coerce").dropna().astype(int)) == {0, 1}, "Label tidak 0/1")
    check(d.groupby("source_file_id")["split"].nunique().le(1).all(), "SOURCE LEAKAGE antar split")
    check(d["source_file_id"].nunique() == len(d), "Sumber audio dipakai berulang (rencana V2 ini harus 1x/source)")
    check(d.groupby("rir_id")["rir_split"].nunique().le(1).all(), "RIR ID seen/unseen tercampur")
    check(set(d.loc[d["split"] == "test", "rir_room"]).issubset(UNSEEN), "Test mengandung RIR room seen")
    check(set(d.loc[d["split"] != "test", "rir_room"]).issubset(SEEN), "Train/val mengandung RIR room unseen")
    check(d.loc[d["split"] == "test", "rir_split"].eq("unseen").all(), "Test rir_split bukan unseen")
    check(d.loc[d["split"] != "test", "rir_split"].eq("seen").all(), "Train/val rir_split bukan seen")
    check(not (set(d.loc[d["split"] == "test", "rir_id"]) & set(d.loc[d["split"] != "test", "rir_id"])),
          "RIR yang sama dipakai pada train/validation dan test")
    check(set(pd.to_numeric(d["snr_db"], errors="coerce")) == EXPECTED_SNRS, "Kumpulan SNR tidak sesuai 5/10/20 dB")
    check((pd.to_numeric(d["sample_rate"], errors="coerce") == 16000).all(), "Manifest sample rate bukan 16000")
    for col in ("silent", "clipped"):
        check(d[col].astype(str).str.lower().isin(["true", "false"]).all(), f"Kolom {col} mengandung nilai tidak valid")
        check(not d[col].astype(str).str.lower().eq("true").any(), f"Ada {col}=True")

    # Same plan rows including all generator metadata, apart from QC fields.
    by_id = p.set_index("file_id", verify_integrity=True)
    if d["file_id"].isin(p["file_id"]).all():
        for field in set(p.columns).intersection(d.columns) - {"file_id", "file_path"}:
            a = d.set_index("file_id")[field].astype(str).sort_index()
            b = by_id.loc[a.index, field].astype(str)
            if not a.equals(b):
                errors.append(f"Nilai manifest berbeda dari plan untuk kolom: {field}")
        if not smoke:
            check(d.set_index("file_id")["file_path"].astype(str).sort_index().equals(
                  by_id.loc[sorted(d["file_id"]), "file_path"].astype(str)), "Path WAV berbeda dari plan")
    else:
        errors.append("Ada file_id manifest yang tidak tercantum dalam plan")

    ctab = pd.crosstab([d["split"], d["label"]], d["snr_db"])
    print("\nSplit x label x SNR:\n" + ctab.to_string())
    for split, n in expect.items():
        sub = d[d["split"] == split]
        check(sub["label"].value_counts().to_dict() == {0: n//2, 1: n//2}, f"Label imbalance pada {split}")
        if not smoke:
            for label in (0, 1):
                for snr in EXPECTED_SNRS:
                    match = ((sub["label"] == label) & (sub["snr_db"] == snr)).sum()
                    check(match == n // 6, f"Distribusi {split}/label={label}/snr={snr}: {match}, harus {n//6}")

    # Verify each actual WAV, not just QC flags written to CSV.
    missing_files = 0
    bad_wavs = 0
    examples = []
    print("\nMembaca metadata dan waveform WAV (ini mungkin memerlukan beberapa saat)...", flush=True)
    for i, row in enumerate(d.itertuples(index=False), start=1):
        path = Path(str(row.file_path))
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            missing_files += 1
            if len(examples) < 10:
                examples.append(f"MISSING: {path}")
            continue
        try:
            info = sf.info(str(path))
            if info.samplerate != 16000 or info.channels != 1 or info.subtype != "PCM_16" or info.frames < 16000:
                raise ValueError(f"format invalid: {info.samplerate} Hz, {info.channels} ch, {info.subtype}, {info.frames} frames")
            wav, sr = sf.read(str(path), dtype="float32")
            if not np.isfinite(wav).all():
                raise ValueError("NaN/Inf in saved waveform")
            peak = float(np.max(np.abs(wav)))
            rms = float(np.sqrt(np.mean(np.square(wav, dtype=np.float64))))
            duration = len(wav) / sr
            if rms < 1e-6 or peak >= .999 or peak > .9501:
                raise ValueError(f"silent/clipped/peak too high: peak={peak}, rms={rms}")
            if abs(peak - float(row.peak)) > 1e-5 or abs(rms - float(row.rms)) > 1e-5:
                raise ValueError("QC peak/rms disagrees with saved waveform")
            if abs(duration - float(row.duration_s)) > 1e-5:
                raise ValueError("QC duration disagrees with saved waveform")
        except Exception as exc:
            bad_wavs += 1
            if len(examples) < 10:
                examples.append(f"INVALID: {path} -> {exc}")
        if i % 500 == 0 or i == len(d):
            print(f"Verified {i}/{len(d)} | missing={missing_files}, invalid={bad_wavs}", flush=True)
    check(missing_files == 0, f"WAV tidak ditemukan: {missing_files}")
    check(bad_wavs == 0, f"WAV tidak valid: {bad_wavs}")
    print(f"\nUnique RIR per split:\n{d.groupby('split')['rir_id'].nunique().to_string()}")
    if errors:
        print("\nVALIDASI GAGAL:")
        for msg in errors:
            print(" -", msg)
        for msg in examples:
            print(" -", msg)
        raise SystemExit(1)
    print("\nVALIDASI V2: PASSED. Manifest dan WAV tersedia, format/QC cocok, distribusi sesuai, no source/RIR room leakage.")
    print("Catatan: verifikasi ini tidak mengukur ulang SNR aktual terhadap clean convolution.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--smoke", action="store_true", help="Cek smoke test 12 file untuk pengujian skrip")
    args = parser.parse_args()
    verify(args.root.resolve(), smoke=args.smoke)


if __name__ == "__main__":
    main()