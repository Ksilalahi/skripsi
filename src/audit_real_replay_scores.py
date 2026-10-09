"""Audit controlled real replay classifier scores for all available evaluation runs.

Copy into src/audit_real_replay_scores.py. Run from project root:
 python src/audit_real_replay_scores.py
 python src/audit_real_replay_scores.py --runs baseline_seed_2027 grl_seed_2027_ep5
No checkpoints or previous evaluation outputs are modified.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
SCORE_CANDIDATES = ('fake_score', 'spoof_score', 'score_fake', 'score_spoof', 'score')


def interpolated_eer(labels, scores):
    """Compute FPR=FNR intersection from ROC with linear interpolation."""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    diff = fpr - fnr
    candidates = np.flatnonzero(diff >= 0)
    if not len(candidates):
        raise ValueError('ROC FPR/FNR intersection not found')
    j = int(candidates[0])
    if j == 0:
        return float((fpr[0] + fnr[0]) / 2)
    i = j - 1
    denom = diff[j] - diff[i]
    w = float(-diff[i] / denom) if denom else 0.0
    return float(fpr[i] + w * (fpr[j] - fpr[i]))


def inspect_run(folder):
    df = pd.read_csv(folder / 'scores.csv')
    score_name = next((x for x in SCORE_CANDIDATES if x in df.columns), None)
    if score_name is None or not {'condition','label'}.issubset(df.columns):
        raise ValueError(f"Unexpected scores columns: {df.columns.tolist()}")
    real = df.loc[df.condition.astype(str).str.lower().str.strip().eq('real')].copy()
    if real.empty:
        raise ValueError(f'No condition=real; found {df.condition.unique().tolist()}')
    y = pd.to_numeric(real.label, errors='raise').to_numpy(dtype=np.int32)
    s = pd.to_numeric(real[score_name], errors='raise').to_numpy(dtype=np.float64)
    if set(np.unique(y)) != {0,1} or not np.isfinite(s).all():
        raise ValueError('Invalid labels (expected 0/1), or non-finite scores')
    union = set(s[y==0]).intersection(set(s[y==1]))
    tied_rows = int(sum(np.count_nonzero(s == v) for v in union))
    row = dict(run=folder.name, n_real=len(real), n_bonafide=int((y==0).sum()),
               n_spoof=int((y==1).sum()), score_column=score_name,
               eer_interpolated=interpolated_eer(y,s), roc_auc_recomputed=float(roc_auc_score(y,s)),
               n_unique_scores=int(len(np.unique(s))), exactly_one=int((s==1).sum()),
               exactly_zero=int((s==0).sum()), near_one_float32_eps=int((s >= 1 - np.finfo(np.float32).eps).sum()),
               n_cross_class_tied_values=len(union), n_cross_class_tied_rows=tied_rows,
               score_min=float(s.min()), score_median=float(np.median(s)), score_max=float(s.max()))
    published = folder / 'condition_summary.csv'
    if published.exists():
        prev = pd.read_csv(published)
        rr = prev.loc[prev.condition.astype(str).str.lower().str.strip().eq('real')]
        if len(rr)==1:
            row['eer_reported'] = float(rr.eer.iloc[0]); row['roc_auc_reported'] = float(rr.roc_auc.iloc[0])
            row['eer_difference'] = row['eer_interpolated'] - row['eer_reported']
            row['roc_auc_difference'] = row['roc_auc_recomputed'] - row['roc_auc_reported']
    detail=[]
    for label, cls in ((0,'bonafide'),(1,'spoof')):
        ss=s[y==label]
        detail.append(dict(run=folder.name, label=label, class_name=cls, n=len(ss),
                           unique=int(len(np.unique(ss))), exactly_one=int((ss==1).sum()),
                           exactly_zero=int((ss==0).sum()), min=float(ss.min()),
                           q25=float(np.quantile(ss,.25)), median=float(np.median(ss)),
                           q75=float(np.quantile(ss,.75)), max=float(ss.max())))
    return row,detail


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT/'results'/'evaluation')
    parser.add_argument('--runs',nargs='+',default=None)
    parser.add_argument('--out',type=Path,default=ROOT/'results'/'audit_real_scores')
    args=parser.parse_args()
    folders=[args.root / r for r in args.runs] if args.runs else sorted(
        set(p.parent for p in args.root.rglob('scores.csv')))
    if not folders:
        raise SystemExit(f'No scores.csv found in {args.root}')
    rows=[]; detail=[]
    print(f'AUDIT SCORES: checking {len(folders)} run folders')
    for folder in folders:
        try:
            row, classes=inspect_run(folder)
        except (OSError, ValueError, KeyError) as e:
            print(f'[SKIP] {folder.name}: {e}')
            continue
        rows.append(row); detail.extend(classes)
        print(f"[OK] {folder.name} | n={row['n_real']} | EER={row['eer_interpolated']*100:.3f}% "
              f"| AUC={row['roc_auc_recomputed']:.5f} | unique={row['n_unique_scores']} "
              f"| exact1={row['exactly_one']} | cross-label tied rows={row['n_cross_class_tied_rows']}")
        if abs(row.get('eer_difference',0))>0.005:
            print('    [CHECK] EER differs >0.5 pp from condition_summary; methods may differ')
        if row['exactly_one'] or row['n_cross_class_tied_rows']:
            print('    [CHECK] Probability saturation / ties; consider audit using raw logits')
    if not rows: raise SystemExit('No valid runs found')
    args.out.mkdir(parents=True,exist_ok=True)
    a=args.out/'audit_real_scores_summary.csv'; b=args.out/'audit_real_scores_by_label.csv'
    pd.DataFrame(rows).to_csv(a,index=False)
    pd.DataFrame(detail).to_csv(b,index=False)
    print(f'Saved: {a}');print(f'Saved: {b}')
    print('Reminder: A CSV audit cannot recover precision lost in rounded/saturated probabilities.')

if __name__=='__main__': main()