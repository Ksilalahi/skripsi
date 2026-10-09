"""Read-only logit-margin audit for controlled real replay.

Place next to evaluate.py, model_replay.py, dataset_replay.py in the src folder.
It evaluates only the ORIGINAL test batches containing real-replay samples. This is
important: replay_collate_fn dynamically pads each batch, so regrouping only the
212 real samples could change feature padding and, in turn, the model outputs.

It does not alter checkpoints, training, evaluation.py, or existing results.
It does NOT select a decision threshold using the held-out test set.

Example:
    python src/evaluate_logit_audit.py --checkpoint checkpoints/replay_resnet/seed_2027/best_model.pt --legacy-run baseline_seed_2027 --batch-size 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_replay import ReplayDataset, replay_collate_fn
from model_replay import ReplayResNet

ROOT = Path(__file__).resolve().parents[1]


def eer_nearest(y: np.ndarray, scores: np.ndarray) -> float:
    """Replicate the original evaluate.py EER algorithm exactly."""
    fpr, tpr, _ = roc_curve(y, scores, pos_label=1)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def eer_interpolated(y: np.ndarray, scores: np.ndarray) -> float:
    """EER by linear interpolation of crossing FPR-FNR in ROC space."""
    fpr, tpr, _ = roc_curve(y, scores, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    difference = fpr - fnr
    right_candidates = np.flatnonzero(difference >= 0)
    if len(right_candidates) == 0:
        raise ValueError("ROC curve did not cross equal-error line")
    j = int(right_candidates[0])
    if j == 0:
        return float(fpr[j])
    i = j - 1
    denom = difference[j] - difference[i]
    frac = 0.0 if denom == 0 else float(-difference[i] / denom)
    return float(fpr[i] + frac * (fpr[j] - fpr[i]))


def tied_across_labels(y: np.ndarray, scores: np.ndarray) -> tuple[int, int]:
    """Return number of tied score values and number of rows involved."""
    common = set(scores[y == 0]).intersection(set(scores[y == 1]))
    n_rows = sum(int(np.count_nonzero(scores == value)) for value in common)
    return len(common), n_rows


def score_stats(y: np.ndarray, scores: np.ndarray, name: str) -> dict:
    if not np.isfinite(scores).all():
        raise ValueError(f"{name} contains non-finite scores")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError(f"{name}: both classes (0 and 1) are required")
    cross_values, cross_rows = tied_across_labels(y, scores)
    return {
        f"{name}_auc": float(roc_auc_score(y, scores)),
        f"{name}_eer_nearest": eer_nearest(y, scores),
        f"{name}_eer_interpolated": eer_interpolated(y, scores),
        f"{name}_unique": int(np.unique(scores).size),
        f"{name}_exact_one": int(np.count_nonzero(scores == 1.0)),
        f"{name}_cross_label_tied_values": int(cross_values),
        f"{name}_cross_label_tied_rows": int(cross_rows),
    }


class OriginalBatchGroups:
    """Yield full original batches that contain >=1 real sample, in order."""

    def __init__(self, real_indices: list[int], n_samples: int, batch_size: int):
        self.n_samples = int(n_samples)
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.starts = sorted({(int(i) // self.batch_size) * self.batch_size for i in real_indices})

    def __iter__(self) -> Iterator[list[int]]:
        for first in self.starts:
            yield list(range(first, min(first + self.batch_size, self.n_samples)))

    def __len__(self) -> int:
        return len(self.starts)


def load_checkpoint_model(checkpoint_path: Path, device: torch.device):
    # weights_only=False is needed for existing checkpoint dictionaries.
    # Only load checkpoints generated on your own trusted machine.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise RuntimeError("Checkpoint missing model_state_dict")
    state = ckpt["model_state_dict"]
    config = ckpt.get("config") or {}
    if not isinstance(config, dict):
        config = {}
    grl = any(str(k).startswith("domain_head.") for k in state.keys())
    n_domain = int(config.get("n_domain") or 4)
    if grl:
        # Match original evaluate.py fallback when no config domain count exists.
        domain_weights = [v for k, v in state.items()
                          if str(k).startswith("domain_head.") and
                          str(k).endswith(".weight") and isinstance(v, torch.Tensor) and v.ndim == 2]
        if domain_weights and "n_domain" not in config:
            n_domain = int(domain_weights[-1].shape[0])
    model = ReplayResNet(
        n_transform=int(config.get("n_transform") or 2),
        embedding_dim=int(config.get("embedding_dim") or 256),
        n_domain=n_domain,
        use_domain_head=grl,
        grl_strength=float(config.get("grl_strength") or 1.0),
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    return model, ckpt, grl


@torch.no_grad()
def evaluate_selected_real_batches(model, loader, real_ids: set[str], device):
    rows = []
    for batch in tqdm(loader, desc="AUDIT REAL", unit="batch", dynamic_ncols=True):
        x = batch["features"].to(device)
        class_logits, _, _ = model(x)
        # Keep original softmax in its original dtype for faithful comparison.
        fake_prob = torch.softmax(class_logits, dim=1)[:, 1]
        # Difference of original float32 logits expressed in float64.
        margin = class_logits[:, 1].to(torch.float64) - class_logits[:, 0].to(torch.float64)
        ids = [str(z) for z in batch["global_feature_ids"]]
        conditions = [str(z).strip().lower() for z in batch["conditions"]]
        labels = batch["class_labels"].cpu().numpy()
        for i, (gid, condition) in enumerate(zip(ids, conditions)):
            if gid not in real_ids:
                continue
            if condition != "real":
                raise RuntimeError(f"Real ID {gid} has unexpected batch condition {condition}")
            rows.append({
                "global_feature_id": gid,
                "label": int(labels[i]),
                "condition": condition,
                "fake_score": float(fake_prob[i].item()),
                "spoof_logit_margin": float(margin[i].item()),
                "logit_bonafide": float(class_logits[i, 0].item()),
                "logit_spoof": float(class_logits[i, 1].item()),
            })
    df = pd.DataFrame(rows)
    if len(df) != len(real_ids) or df["global_feature_id"].duplicated().any():
        raise RuntimeError(f"Expected {len(real_ids)} unique real samples, found {len(df)}")
    return df


def compare_legacy(new_df: pd.DataFrame, folder: Path) -> dict:
    file = folder / "scores.csv"
    if not file.is_file():
        raise FileNotFoundError(f"Legacy scores missing: {file}")
    old = pd.read_csv(file, low_memory=False, usecols=["global_feature_id", "label", "condition", "fake_score"])
    old = old.loc[old["condition"].astype(str).str.lower().str.strip().eq("real")]
    if old["global_feature_id"].duplicated().any():
        raise RuntimeError("Duplicate IDs in legacy scores")
    joined = new_df.merge(old, on="global_feature_id", how="outer", validate="one_to_one",
                          suffixes=("_new", "_old"), indicator=True)
    if not joined["_merge"].eq("both").all():
        raise RuntimeError(f"Legacy/new IDs not identical: {joined['_merge'].value_counts().to_dict()}")
    if not (joined["label_new"] == joined["label_old"]).all():
        raise RuntimeError("Legacy/new labels differ")
    diffs = np.abs(joined["fake_score_new"].to_numpy(dtype=float) -
                   joined["fake_score_old"].to_numpy(dtype=float))
    return {
        "legacy_run": folder.name,
        "legacy_scores_count": len(old),
        "legacy_prob_abs_max": float(diffs.max()),
        "legacy_prob_abs_mean": float(diffs.mean()),
        "legacy_prob_rows_differ_gt_1e_minus_6": int((diffs > 1e-6).sum()),
    }


def main():
    ap = argparse.ArgumentParser(description="Real-replay raw-logit-margin EER audit (read-only)")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--legacy-run", type=str, default=None,
                    help="Existing folder name under results/evaluation for score identity check")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="MUST match batch size used in original full-test evaluation")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--run-name", type=str, default=None)
    args = ap.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        ap.error("batch-size must be > 0 and num-workers >= 0")
    p = Path(args.checkpoint)
    if not p.is_absolute():
        p = ROOT / p
    p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt, grl = load_checkpoint_model(p, device)
    config = ckpt.get("config") or {}
    seed = config.get("seed", "unknown") if isinstance(config, dict) else "unknown"
    run_name = args.run_name or (args.legacy_run + "_logits" if args.legacy_run else f"seed_{seed}_{'grl' if grl else 'baseline'}_logits")
    if not run_name or Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("run-name must be a single folder name")
    output = ROOT / "results" / "logit_margin_audit" / run_name
    if output.exists():
        raise FileExistsError(f"Audit output folder already exists: {output}. Choose --run-name")
    if args.legacy_run and not (ROOT / "results" / "evaluation" / args.legacy_run / "scores.csv").is_file():
        raise FileNotFoundError(f"Legacy scores.csv not found for --legacy-run {args.legacy_run}")
    print(f"Checkpoint: {p}\nSeed: {seed}; GRL: {grl}; checkpoint epoch: {ckpt.get('epoch')}\nDevice: {device}")
    print("Loading test data and locating only controlled real replay samples...")
    dataset = ReplayDataset(split="test")
    manifest = dataset.df.reset_index(drop=True)
    if "condition" not in manifest or "global_feature_id" not in manifest:
        raise RuntimeError("Test manifest missing required condition/global_feature_id columns")
    real_mask = manifest["condition"].astype(str).str.strip().str.lower().eq("real")
    real_indices = np.flatnonzero(real_mask.to_numpy()).tolist()
    real_ids = set(manifest.loc[real_mask, "global_feature_id"].astype(str).tolist())
    if not real_indices or len(real_ids) != len(real_indices):
        raise RuntimeError("No controlled real replay samples or duplicate feature IDs")
    orig_groups = OriginalBatchGroups(real_indices, len(dataset), args.batch_size)
    print(f"Real samples={len(real_indices)} | original batches to evaluate={len(orig_groups)} "
          f"(vs {int(np.ceil(len(dataset)/args.batch_size))} full-test batches)")
    loader = DataLoader(dataset, batch_sampler=orig_groups, num_workers=args.num_workers,
                        collate_fn=replay_collate_fn, pin_memory=(device.type == "cuda"))
    new_df = evaluate_selected_real_batches(model, loader, real_ids, device)
    y = new_df["label"].to_numpy(dtype=int)
    probs = new_df["fake_score"].to_numpy(dtype=float)
    margins = new_df["spoof_logit_margin"].to_numpy(dtype=float)
    summary = {
        "seed": seed, "checkpoint": str(p), "checkpoint_epoch": ckpt.get("epoch"),
        "use_grl": grl, "n_real": len(new_df), "n_bonafide": int((y==0).sum()),
        "n_spoof": int((y==1).sum()), "batch_size": args.batch_size,
        **score_stats(y, probs, "prob"),
        **score_stats(y, margins, "margin"),
    }
    if args.legacy_run:
        summary.update(compare_legacy(new_df, ROOT / "results" / "evaluation" / args.legacy_run))
    output.mkdir(parents=True, exist_ok=False)
    new_df.to_csv(output / "real_logits_scores.csv", index=False, float_format="%.17g")
    pd.DataFrame([summary]).to_csv(output / "audit_summary.csv", index=False)
    with open(output / "audit_config.json", "w", encoding="utf-8") as f:
        json.dump({"checkpoint": str(p), "legacy_run": args.legacy_run,
                   "batch_size": args.batch_size, "test_condition": "real", "method": "logit margin"}, f, indent=2)
    print("\nRESULTS (EER as percent):")
    print(f" Probability       : nearest={summary['prob_eer_nearest']*100:.4f}% "
          f"interpolated={summary['prob_eer_interpolated']*100:.4f}% "
          f"AUC={summary['prob_auc']:.6f} unique={summary['prob_unique']}")
    print(f" Raw logit margin  : nearest={summary['margin_eer_nearest']*100:.4f}% "
          f"interpolated={summary['margin_eer_interpolated']*100:.4f}% "
          f"AUC={summary['margin_auc']:.6f} unique={summary['margin_unique']}")
    print(f" Cross-label tied rows: prob={summary['prob_cross_label_tied_rows']} "
          f"margin={summary['margin_cross_label_tied_rows']}")
    if args.legacy_run:
        print(f" Score match to original CSV (max absolute diff): {summary['legacy_prob_abs_max']:.3g}")
        if summary["legacy_prob_rows_differ_gt_1e_minus_6"]:
            print("[CHECK] Some probabilities differ from original evaluation (>1e-6). "
                  "Do not compare EER until the data/batch/procedure difference is explained.")
    print(f"\nSaved: {output / 'real_logits_scores.csv'}")
    print(f"Saved: {output / 'audit_summary.csv'}")
    print("Note: Scores and EER here describe bona fide/spoof source classification, NOT replay transformation detection.")


if __name__ == "__main__":
    main()