"""Simulated replay V2: separate, reproducible, leakage-audited large-scale generator.

Requirements: numpy, scipy, pandas, soundfile.
Run this file under PROJECT_ROOT/src/simulate_replay_v2.py.
Plan: python src/simulate_replay_v2.py --mode plan
Smoke: python src/simulate_replay_v2.py --mode smoke
Generate: python src/simulate_replay_v2.py --mode generate

Does NOT overwrite V1 simulated audio or V1 manifest. V2 is experimental;
feature extraction, dataset manifest integration, and training changes are separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

SEEN_ROOMS = ("lecture", "meeting", "phone", "stairway")
UNSEEN_ROOMS = ("booth", "office")
TARGET_SR = 16000
PEAK_LIMIT = 0.95
PLAN_COLUMNS = (
    "file_id", "source_file_id", "file_path", "source_file_path", "speaker_id",
    "attack_id", "label", "dataset", "split", "condition", "rir_id",
    "rir_room", "rir_split", "rir_file_path", "snr_db", "noise_seed",
    "sample_rate", "simulation_version",
)
QC_COLUMNS = ("peak", "rms", "silent", "clipped", "duration_s")


def require_columns(df: pd.DataFrame, names, description: str):
    missing = set(names) - set(df.columns)
    if missing:
        raise ValueError(f"{description} missing columns: {sorted(missing)}")


def resolve_path(root: Path, raw: str) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else root / path


def read_inputs(root: Path):
    source_path = root / "manifests" / "source_manifest.csv"
    rir_path = root / "manifests" / "rir_manifest.csv"
    if not source_path.is_file() or not rir_path.is_file():
        raise FileNotFoundError(
            f"Required inputs:\n  {source_path}\n  {rir_path}\n"
            "Upload/verify both parent manifests before generation."
        )
    sources = pd.read_csv(source_path, keep_default_na=False)
    rirs = pd.read_csv(rir_path, keep_default_na=False)
    require_columns(sources, ["source_file_id", "file_path", "speaker_id", "attack_id",
                              "label", "dataset", "split"], "source_manifest")
    require_columns(rirs, ["rir_id", "file_path", "room", "rir_split"], "rir_manifest")
    sources = sources.copy()
    rirs = rirs.copy()
    sources["split"] = sources["split"].astype(str).str.strip().str.lower()
    sources["label"] = pd.to_numeric(sources["label"], errors="raise").astype(int)
    rirs["room"] = rirs["room"].astype(str).str.strip().str.lower()
    rirs["rir_split"] = rirs["rir_split"].astype(str).str.strip().str.lower()
    if not sources["split"].isin(["train", "validation", "test"]).all():
        raise ValueError("Unexpected source split. Expected train/validation/test.")
    if not sources["label"].isin([0, 1]).all():
        raise ValueError("Source labels must be 0/1.")
    if sources["source_file_id"].duplicated().any():
        raise ValueError("Duplicate source_file_id in source manifest; fix before generating V2.")
    if rirs["rir_id"].duplicated().any():
        raise ValueError("Duplicate rir_id in RIR manifest.")
    if sources["source_file_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Empty source ID.")
    if not rirs["rir_split"].isin(["seen", "unseen"]).all():
        raise ValueError("RIR split must be seen or unseen.")
    if not set(rirs.loc[rirs["rir_split"] == "seen", "room"]).issuperset(SEEN_ROOMS):
        raise ValueError("Missing seen room(s).")
    if not set(rirs.loc[rirs["rir_split"] == "unseen", "room"]).issuperset(UNSEEN_ROOMS):
        raise ValueError("Missing unseen room(s).")
    forbidden = rirs.loc[
        ((rirs["rir_split"] == "seen") & rirs["room"].isin(UNSEEN_ROOMS)) |
        ((rirs["rir_split"] == "unseen") & rirs["room"].isin(SEEN_ROOMS))
    ]
    if not forbidden.empty:
        raise ValueError("Inconsistent rir_split vs room identity: seen/unseen room leakage.")
    # Prevent accidentally reusing a physically identical file in seen and unseen.
    if rirs.groupby("file_path")["rir_split"].nunique().gt(1).any():
        raise ValueError("Same RIR file is assigned both seen and unseen.")
    return sources, rirs, source_path, rir_path


def _source_draws(pool, quota: int, rng, *, unique_required: bool):
    if not pool:
        raise ValueError("No source candidates for a requested split/label.")
    if unique_required and quota > len(pool):
        raise ValueError(f"Need {quota} unique sources, but only {len(pool)} available.")
    out = []
    while len(out) < quota:
        permutation = rng.permutation(len(pool))
        out.extend(pool[int(i)] for i in permutation)
    return out[:quota]


def build_plan(root: Path, sources, rirs, counts: dict, seed: int, snrs: list[int]):
    rng = np.random.default_rng(seed)
    by_room = {}
    for rir_split, rooms in (("seen", SEEN_ROOMS), ("unseen", UNSEEN_ROOMS)):
        for room in rooms:
            part = rirs[(rirs["rir_split"] == rir_split) & (rirs["room"] == room)]
            if part.empty:
                raise ValueError(f"No RIRs: {rir_split}/{room}")
            by_room[(rir_split, room)] = part.to_dict("records")
    all_rows = []
    for split in ("train", "validation", "test"):
        total = counts[split]
        if total <= 0 or total % 2:
            raise ValueError(f"{split} count must be a positive even integer.")
        rir_split = "unseen" if split == "test" else "seen"
        rooms = UNSEEN_ROOMS if split == "test" else SEEN_ROOMS
        candidates = []
        for label in (0, 1):
            pool = sources[(sources["split"] == split) & (sources["label"] == label)].to_dict("records")
            selected = _source_draws(pool, total // 2, rng, unique_required=(split != "train"))
            # Balanced within each label across room and SNR: randomize ordering, then cycle.
            room_indices = np.arange(len(selected)) % len(rooms)
            snr_indices = np.arange(len(selected)) % len(snrs)
            rng.shuffle(room_indices)
            rng.shuffle(snr_indices)
            for j, source in enumerate(selected):
                room = rooms[int(room_indices[j])]
                rir_pool = by_room[(rir_split, room)]
                rir = rir_pool[int(rng.integers(0, len(rir_pool)))]
                candidates.append({
                    "source_file_id": str(source["source_file_id"]),
                    "source_file_path": str(source["file_path"]),
                    "speaker_id": str(source["speaker_id"]),
                    "attack_id": str(source["attack_id"]),
                    "label": int(label), "dataset": str(source["dataset"]),
                    "split": split, "condition": "simulated_replay",
                    "rir_id": str(rir["rir_id"]), "rir_room": room,
                    "rir_split": rir_split, "rir_file_path": str(rir["file_path"]),
                    "snr_db": int(snrs[int(snr_indices[j])]),
                    "sample_rate": TARGET_SR, "simulation_version": "v2",
                })
        rng.shuffle(candidates)
        for i, candidate in enumerate(candidates):
            file_id = f"simv2_{split}_{i:06d}"
            candidate["file_id"] = file_id
            candidate["file_path"] = f"data/processed/simulated_replay_v2/{split}/{file_id}.wav"
            candidate["noise_seed"] = int.from_bytes(
                hashlib.sha256(f"{seed}:{file_id}".encode()).digest()[:4], "big"
            )
            all_rows.append(candidate)
    plan = pd.DataFrame.from_records(all_rows, columns=list(PLAN_COLUMNS))
    audit_plan(plan, counts, sources)
    # Early file reference check on only unique inputs (not 10,000 repeated paths).
    unique_paths = set(plan["source_file_path"]) | set(plan["rir_file_path"])
    missing = [p for p in sorted(unique_paths) if not resolve_path(root, p).is_file()]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"Referenced audio files missing ({len(missing)}):\n{preview}")
    return plan


def audit_plan(plan, counts, sources):
    if plan["file_id"].duplicated().any() or plan["file_path"].duplicated().any():
        raise AssertionError("Duplicate V2 file_id/file_path")
    for split, count in counts.items():
        subset = plan[plan["split"] == split]
        if len(subset) != count:
            raise AssertionError(f"Expected {count} {split}; got {len(subset)}")
        if subset["label"].value_counts().to_dict() != {0: count//2, 1: count//2}:
            raise AssertionError(f"Label imbalance in {split}")
        if not subset["rir_split"].eq("unseen" if split == "test" else "seen").all():
            raise AssertionError("Seen/unseen protocol mismatch")
    if plan.groupby("source_file_id")["split"].nunique().gt(1).any():
        raise AssertionError("Source leakage across splits")
    if plan.groupby("rir_id")["rir_split"].nunique().gt(1).any():
        raise AssertionError("RIR leakage across seen/unseen")
    if plan["label"].isna().any():
        raise AssertionError("Missing label")
    if plan[plan["split"] == "test"]["source_file_id"].duplicated().any():
        raise AssertionError("Test should have unique source IDs")
    if plan[plan["split"] == "validation"]["source_file_id"].duplicated().any():
        raise AssertionError("Validation should have unique source IDs")
    source_split = sources.set_index("source_file_id")["split"]
    if not (plan["split"].to_numpy() == plan["source_file_id"].map(source_split).to_numpy()).all():
        raise AssertionError("Source split preservation failed")


def load_mono_16k(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if sr != TARGET_SR:
        gcd = math.gcd(int(sr), TARGET_SR)
        mono = resample_poly(mono, TARGET_SR // gcd, int(sr) // gcd).astype(np.float32)
    if not len(mono) or not np.isfinite(mono).all():
        raise ValueError(f"Invalid audio: {path}")
    return mono


def create_simulation(source: np.ndarray, rir: np.ndarray, snr_db: float, noise_seed: int):
    rir_peak = float(np.max(np.abs(rir)))
    if rir_peak < 1e-9:
        raise ValueError("Silent RIR")
    rir = rir / rir_peak
    conv = fftconvolve(source, rir, mode="full")[:len(source)].astype(np.float32)
    conv_peak = float(np.max(np.abs(conv)))
    if conv_peak < 1e-9:
        raise ValueError("Silent convolution output")
    conv /= conv_peak
    rng = np.random.default_rng(noise_seed)
    noise = rng.standard_normal(len(conv)).astype(np.float32)
    signal_rms = float(np.sqrt(np.mean(np.square(conv, dtype=np.float64))))
    noise_rms = float(np.sqrt(np.mean(np.square(noise, dtype=np.float64))))
    if noise_rms < 1e-9:
        raise ValueError("Silent Gaussian noise")
    scale = signal_rms / (10.0 ** (snr_db / 20.0) * noise_rms)
    mixture = conv + (noise * scale)
    peak = float(np.max(np.abs(mixture)))
    if peak > PEAK_LIMIT:
        mixture *= (PEAK_LIMIT / peak)
    return mixture.astype(np.float32)


def check_waveform(wav: np.ndarray):
    peak = float(np.max(np.abs(wav)))
    rms = float(np.sqrt(np.mean(np.square(wav, dtype=np.float64))))
    quality = {"peak": round(peak, 8), "rms": round(rms, 8),
               "silent": bool(rms < 1e-6), "clipped": bool(peak >= 0.999),
               "duration_s": round(len(wav) / TARGET_SR, 6)}
    if not np.isfinite(wav).all() or quality["silent"] or quality["clipped"] or peak > PEAK_LIMIT + (1.0 / 32768.0):
        raise ValueError(f"Invalid simulated audio quality: {quality}")
    return quality


def atomic_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def generate(root: Path, plan: pd.DataFrame, *, smoke=False, smoke_per_split=4, flush_every=100):
    if smoke:
        selected = []
        for split in ("train", "validation", "test"):
            part = plan[plan["split"] == split]
            for label in (0, 1):
                selected.append(part[part["label"] == label].head(max(1, smoke_per_split//2)))
        selected_plan = pd.concat(selected, ignore_index=True)
        manifest_path = root / "manifests" / "manifest_replay_sim_v2_smoke.csv"
    else:
        selected_plan = plan
        manifest_path = root / "manifests" / "manifest_replay_sim_v2.csv"

    source_cache = {}
    rir_cache = {}
    rows = []
    reused = 0
    produced = 0
    total = len(selected_plan)
    for index, (_, row) in enumerate(selected_plan.iterrows(), start=1):
        output_path = resolve_path(root, row["file_path"])
        if smoke:
            output_path = root / "data" / "processed" / "simulated_replay_v2_smoke" / row["split"] / f"{row['file_id']}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.is_file():
            info = sf.info(str(output_path))
            if info.samplerate != TARGET_SR or info.channels != 1 or info.frames < 16000:
                raise RuntimeError(f"Existing output invalid; inspect manually: {output_path}")
            wav, _ = sf.read(str(output_path), dtype="float32")
            qc = check_waveform(wav)
            reused += 1
        else:
            source_path = resolve_path(root, row["source_file_path"])
            rir_path = resolve_path(root, row["rir_file_path"])
            if source_path not in source_cache:
                source_cache[source_path] = load_mono_16k(source_path)
            if rir_path not in rir_cache:
                rir_cache[rir_path] = load_mono_16k(rir_path)
            wave = create_simulation(source_cache[source_path], rir_cache[rir_path],
                                     float(row["snr_db"]), int(row["noise_seed"]))
            check_waveform(wave)
            sf.write(str(output_path), wave, TARGET_SR, subtype="PCM_16")
            # Check saved PCM16, not just pre-quantization samples.
            saved, _ = sf.read(str(output_path), dtype="float32")
            qc = check_waveform(saved)
            produced += 1
        record = row.to_dict()
        if smoke:
            record["file_path"] = output_path.relative_to(root).as_posix()
        record.update(qc)
        rows.append(record)
        # Bound cache memory while retaining good I/O performance.
        if len(source_cache) > 24:
            source_cache.clear()
        if len(rir_cache) > 64:
            rir_cache.clear()
        if index % flush_every == 0 or index == total:
            atomic_csv(pd.DataFrame(rows), manifest_path)
            print(f"{index}/{total} | new={produced}, existing={reused} | {manifest_path.name}", flush=True)
    if not smoke:
        audit_plan(pd.DataFrame(rows), {s: int((plan["split"] == s).sum()) for s in ("train", "validation", "test")},
                   pd.DataFrame({"source_file_id": plan["source_file_id"], "split": plan["split"]}).drop_duplicates())
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "smoke", "generate"), default="plan")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train", type=int, default=8000)
    parser.add_argument("--validation", type=int, default=1000)
    parser.add_argument("--test", type=int, default=1000)
    parser.add_argument("--snrs", type=str, default="5,10,15,20,25,30")
    parser.add_argument("--smoke-per-split", type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve()
    counts = {"train": args.train, "validation": args.validation, "test": args.test}
    snrs = [int(s) for s in args.snrs.split(",")]
    if not snrs or any(s <= 0 or s > 60 for s in snrs) or len(snrs) != len(set(snrs)):
        raise ValueError("SNR list should contain distinct values in (0, 60] dB")
    sources, rirs, source_path, rir_path = read_inputs(root)
    plan = build_plan(root, sources, rirs, counts, args.seed, snrs)
    plan_path = root / "manifests" / "plan_replay_sim_v2.csv"
    if plan_path.is_file():
        old = pd.read_csv(plan_path, keep_default_na=False)
        # Never silently replace a plan that may already have been used for generation.
        if not old.astype(str).equals(plan.astype(str)):
            raise RuntimeError(f"Existing V2 plan differs from current arguments/input. Archive first: {plan_path}")
    else:
        atomic_csv(plan, plan_path)
    print(f"PLAN : {plan_path}\nROWS : {len(plan)}")
    print("Split x label:\n" + pd.crosstab(plan["split"], plan["label"]).to_string())
    print("Split x RIR room:\n" + pd.crosstab(plan["split"], plan["rir_room"]).to_string())
    print("Unique source by split:\n" + plan.groupby("split")["source_file_id"].nunique().to_string())
    print("Unique RIR by split:\n" + plan.groupby("split")["rir_id"].nunique().to_string())
    print("SOURCE LEAKAGE: NONE | RIR ROOM LEAKAGE: NONE")
    if args.mode != "plan":
        manifest_path = generate(root, plan, smoke=(args.mode == "smoke"),
                                 smoke_per_split=args.smoke_per_split)
        print(f"OUTPUT MANIFEST : {manifest_path}")
    else:
        print("PLAN ONLY: no audio generated. Next: --mode smoke")


if __name__ == "__main__":
    main()