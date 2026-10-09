#!/usr/bin/env python3
"""Isolated V2 3-channel feature extraction with the ORIGINAL V1 algorithms.

Place inside PROJECT_ROOT/src. Requires the existing, unchanged scripts:
  spectral_phase.py, extract_mgd.py, build_three_channel_features.py

First run:
  python src/extract_three_channel_v2.py --mode plan
  python src/extract_three_channel_v2.py --mode smoke
After smoke passes:
  python src/extract_three_channel_v2.py --mode extract
After extraction passes:
  python src/extract_three_channel_v2.py --mode combine

Fixed V1 training-only normalization stats are deliberately reused. No old files
are modified; output filenames and manifests live in separate *_v2 paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from spectral_phase import SpectralPhase
from extract_mgd import load_audio_read_only
from build_three_channel_features import load_stats, build_three_channel

ROOT = Path(__file__).resolve().parents[1]
V2_AUDIO_MANIFEST = ROOT / 'manifests' / 'manifest_replay_sim_v2.csv'
V1_FEATURE_MANIFEST = ROOT / 'manifests' / 'manifest_three_channel.csv'
V2_SIM_FEATURE_MANIFEST = ROOT / 'manifests' / 'manifest_three_channel_sim_v2.csv'
V2_SMOKE_FEATURE_MANIFEST = ROOT / 'manifests' / 'manifest_three_channel_sim_v2_smoke.csv'
V2_COMBINED_MANIFEST = ROOT / 'manifests' / 'manifest_three_channel_v2.csv'
V2_FEATURE_ROOT = ROOT / 'data' / 'features' / 'three_channel_v2'
V2_SMOKE_FEATURE_ROOT = ROOT / 'data' / 'features' / 'three_channel_v2_smoke'
ERROR_REPORT = ROOT / 'results' / 'v2_feature_extraction_errors.csv'
EXPECTED = {'train': 2400, 'validation': 300, 'test': 300}
REQUIRED = {'file_id','file_path','source_file_id','speaker_id','attack_id',
            'label','dataset','split','condition','rir_id','rir_room','rir_split','snr_db'}
ALLOWED_ROOMS = {'seen': {'lecture','meeting','phone','stairway'},
                 'unseen': {'booth','office'}}


def resolve_path(text):
    p = Path(str(text))
    return p if p.is_absolute() else ROOT / p


def sha256(path):
    dig = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            dig.update(block)
    return dig.hexdigest()


def safe_id(value):
    value = str(value)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',value):
        raise ValueError(f'Unsafe feature/file ID: {value!r}')
    return value


def atomic_csv(df,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.csv.tmp')
    df.to_csv(tmp,index=False)
    os.replace(tmp,path)


def read_audio_manifest():
    if not V2_AUDIO_MANIFEST.is_file():
        raise FileNotFoundError(V2_AUDIO_MANIFEST)
    df=pd.read_csv(V2_AUDIO_MANIFEST,low_memory=False)
    missing=REQUIRED-set(df.columns)
    if missing:
        raise ValueError(f'V2 manifest missing columns: {sorted(missing)}')
    df=df.copy()
    for c in ('split','condition','rir_split','rir_room'):
        df[c]=df[c].astype(str).str.strip().str.lower()
    df['label']=pd.to_numeric(df['label'],errors='raise').astype(int)
    df['snr_db']=pd.to_numeric(df['snr_db'],errors='raise').astype(int)
    if df['file_id'].isna().any() or df['file_id'].duplicated().any():
        raise ValueError('Duplicate/NA file_id in V2 manifest')
    if df['file_path'].duplicated().any():
        raise ValueError('Duplicate file_path in V2 manifest')
    if df['source_file_id'].isna().any():
        raise ValueError('Missing source_file_id')
    if not df['condition'].eq('simulated_replay').all():
        raise ValueError(f'Unexpected original V2 conditions: {df.condition.unique().tolist()}')
    if not df['label'].isin([0,1]).all() or not df['snr_db'].isin([5,10,20]).all():
        raise ValueError('Unexpected class label or SNR')
    if len(df)!=sum(EXPECTED.values()):
        raise ValueError(f'Expected 3000 V2 rows; got {len(df)}')
    for split,n in EXPECTED.items():
        part=df[df.split==split]
        if len(part)!=n or part.label.value_counts().to_dict()!={0:n//2,1:n//2}:
            raise ValueError(f'Unexpected counts in {split}')
        desired='unseen' if split=='test' else 'seen'
        if not part.rir_split.eq(desired).all():
            raise ValueError(f'RIR split protocol failed for {split}')
        if not set(part.rir_room).issubset(ALLOWED_ROOMS[desired]):
            raise ValueError(f'RIR room protocol failed for {split}')
    if df.groupby('source_file_id').split.nunique().gt(1).any():
        raise ValueError('Source leakage across splits')
    if df.groupby('rir_id').rir_split.nunique().gt(1).any():
        raise ValueError('RIR ID leakage across seen/unseen')
    if not df.file_path.map(lambda v: resolve_path(v).is_file()).all():
        bad=df.loc[~df.file_path.map(lambda v:resolve_path(v).is_file()),'file_path']
        raise FileNotFoundError(f'WAV files missing: {bad.head(3).tolist()}')
    return df


def smoke_subset(df):
    # 2 bona fide + 2 spoof per split; intentionally deterministic.
    return (df.sort_values(['split','label','file_id'])
            .groupby(['split','label'],sort=True).head(2)
            .reset_index(drop=True))


def feature_output_path(row,smoke):
    base=V2_SMOKE_FEATURE_ROOT if smoke else V2_FEATURE_ROOT
    return base / str(row['split']) / (safe_id(row['file_id'])+'.npz')


def check_saved_feature(path):
    from build_three_channel_features import EXPECTED_MELS
    with np.load(path,allow_pickle=False) as archive:
        if 'feature' not in archive.files:
            raise ValueError(f"Feature key missing: {path}")
        ft=np.asarray(archive['feature'])
    if ft.ndim!=3 or ft.shape[:2]!=(3,EXPECTED_MELS) or ft.shape[2]<1:
        raise ValueError(f'Invalid feature shape {ft.shape}: {path}')
    if not np.isfinite(ft).all():
        raise ValueError(f'Nonfinite values in {path}')
    return ft


def make_feature(row,extractor,stats):
    waveform,sr=load_audio_read_only(resolve_path(row['file_path']),target_sr=16000)
    if sr!=16000:
        raise ValueError(f'Unexpected sample rate after load: {sr}')
    x=torch.from_numpy(waveform).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        spectral=extractor(x)[0].cpu().numpy().astype(np.float32)
        mgd=extractor.modified_group_delay(x,alpha=0.4,gamma=0.9)[0].cpu().numpy().astype(np.float32)
    return build_three_channel(spectral,mgd,*stats)


def row_from_feature(row,rel_path,feature):
    # Exact naming/format requirements of V1 ReplayDataset, with V2 IDs.
    return {
        'global_feature_id':'simulated_v2__'+safe_id(row['file_id']),
        'feature_id':safe_id(row['file_id']),
        'condition':'simulated',
        'source_file_id':str(row['source_file_id']),
        'recording_id':'',
        'speaker_id':row.get('speaker_id',''),
        'attack_id':row.get('attack_id',''),
        'label':int(row['label']),
        'dataset':row.get('dataset',''),
        'split':str(row['split']),
        'config_id':'', 'room':'','playback_device':'','microphone':'',
        'rir_id':str(row['rir_id']),
        'rir_room':str(row['rir_room']),
        'rir_split':str(row['rir_split']),
        'snr_db':int(row['snr_db']),
        'feature_path':str(rel_path).replace('\\','/'),
        'n_channels':int(feature.shape[0]),
        'n_mels':int(feature.shape[1]),
        'n_frames':int(feature.shape[2]),
        'finite':True,
        'simulation_version':'v2',
    }


def extract(df,smoke=False,limit=None):
    if limit is not None and not smoke:
        raise ValueError('--limit only permitted in smoke mode')
    part=smoke_subset(df) if smoke else df
    if limit is not None:
        part=part.head(limit)
    outmanifest=V2_SMOKE_FEATURE_MANIFEST if smoke else V2_SIM_FEATURE_MANIFEST
    stats=load_stats()  # ORIGINAL V1 training-only stats, never refitted from V2 test/val.
    extractor=SpectralPhase(sr=16000,n_fft=512,hop=160,n_mels=80)
    extractor.eval().to('cpu')
    rows=[]
    errors=[]
    generated=0
    reused=0
    for _,row in tqdm(part.iterrows(),total=len(part),desc='V2 3-channel smoke' if smoke else 'V2 3-channel'):
        output=feature_output_path(row,smoke)
        try:
            if output.exists():
                feature=check_saved_feature(output)
                reused+=1
            else:
                feature=make_feature(row,extractor,stats)
                output.parent.mkdir(parents=True,exist_ok=True)
                temp=output.with_suffix('.npz.tmp')
                with open(temp,'wb') as f:
                    np.savez_compressed(f,feature=feature.astype(np.float32,copy=False))
                os.replace(temp,output)
                generated+=1
            rows.append(row_from_feature(row,output.relative_to(ROOT),feature))
        except Exception as e:
            errors.append({'file_id':row['file_id'],'error':repr(e)})
            break
        if len(rows)%100==0 and not smoke:
            atomic_csv(pd.DataFrame(rows),outmanifest.with_name('manifest_three_channel_sim_v2_partial.csv'))
            print(f"{len(rows)}/{len(part)} | new={generated} | reused={reused}")
    if errors:
        ERROR_REPORT.parent.mkdir(parents=True,exist_ok=True)
        atomic_csv(pd.DataFrame(errors),ERROR_REPORT)
        raise RuntimeError(f"Feature extraction stopped: {errors[0]}. Existing output can be resumed safely.")
    outdf=pd.DataFrame(rows)
    if len(outdf)!=len(part) or outdf.global_feature_id.duplicated().any():
        raise RuntimeError('Feature count or ID integrity failed')
    if not outdf.finite.eq(True).all() or not outdf.n_channels.eq(3).all() or not outdf.n_mels.eq(80).all():
        raise RuntimeError('Feature shape/finite QC failed')
    atomic_csv(outdf,outmanifest)
    print(f"PASSED: {len(outdf)} features | new={generated} reused={reused}")
    print('Feature manifest:',outmanifest)
    print('Feature example:',outdf.iloc[0].feature_path, '| frames:',outdf.iloc[0].n_frames)
    return outdf


def combine(df):
    if not V1_FEATURE_MANIFEST.exists():
        raise FileNotFoundError(f'Missing original V1 features manifest: {V1_FEATURE_MANIFEST}')
    if not V2_SIM_FEATURE_MANIFEST.exists():
        raise FileNotFoundError(f'First run --mode extract: {V2_SIM_FEATURE_MANIFEST}')
    old=pd.read_csv(V1_FEATURE_MANIFEST,low_memory=False)
    new=pd.read_csv(V2_SIM_FEATURE_MANIFEST,low_memory=False)
    if new.shape[0]!=3000:
        raise ValueError(f'Expected 3000 extracted V2 features, got {new.shape[0]}')
    if not new.condition.eq('simulated').all():
        raise ValueError('New features have wrong condition; must be simulated')
    if not set(['clean','real','simulated']).issuperset(old.condition.dropna().unique()):
        raise ValueError('Unexpected original V1 conditions')
    if old.condition.eq('simulated').sum()!=150:
        raise ValueError('Expected 150 simulated V1 features in original manifest')
    clean_real=old.loc[old.condition.isin(['clean','real'])].copy()
    if clean_real.condition.value_counts().to_dict()!={'clean':120184,'real':480}:
        raise ValueError(f'Unexpected clean/real V1 counts: {clean_real.condition.value_counts().to_dict()}')
    if 'source_file_id' not in clean_real:
        raise ValueError('Original manifest needs source_file_id')
    # Ensure simulated V2 uses the source-level splits assigned to clean originals.
    original_sources=(clean_real.loc[clean_real.condition=='clean',['source_file_id','split']]
                      .drop_duplicates())
    if original_sources.groupby('source_file_id').split.nunique().gt(1).any():
        raise ValueError('Original clean audio have conflicting source splits')
    source_split=original_sources.set_index('source_file_id')['split']
    actual_split=new.source_file_id.map(source_split)
    if actual_split.isna().any():
        bad=new.loc[actual_split.isna(),'source_file_id'].head(5).tolist()
        raise ValueError(f'V2 source IDs absent in V1 clean: {bad}')
    if not new.split.astype(str).eq(actual_split.astype(str)).all():
        raise ValueError('Cross-manifest source split violation')
    # No V2 output may shadow V1 features or old IDs.
    if set(new.global_feature_id).intersection(set(clean_real.global_feature_id)):
        raise ValueError('Global ID overlap with V1 clean or real')
    if set(new.feature_path).intersection(set(clean_real.feature_path)):
        raise ValueError('Feature path overlap with V1')
    combined=pd.concat([clean_real,new],ignore_index=True,sort=False)
    if combined.global_feature_id.duplicated().any():
        raise ValueError('Duplicate combined global_feature_id')
    if len(combined)!=123664:
        raise ValueError(f'Expected 123664 combined rows, got {len(combined)}')
    counts=pd.crosstab(combined.split,combined.condition)
    print('Combined V2 split x condition:\n'+counts.to_string())
    for split,n in EXPECTED.items():
        check=new[new.split==split]
        if len(check)!=n or not check.rir_split.eq('unseen' if split=='test' else 'seen').all():
            raise ValueError(f'Combined V2 simulated protocol failed for {split}')
    # Check 3000 new feature paths exist, without loading the 120k old files.
    if not new.feature_path.map(lambda p:resolve_path(p).is_file()).all():
        raise FileNotFoundError('Some V2 feature npz are missing')
    atomic_csv(combined,V2_COMBINED_MANIFEST)
    print('COMBINE PASSED:',len(combined),'rows')
    print('Combined feature manifest:',V2_COMBINED_MANIFEST)
    print('V1 feature manifest UNCHANGED:',V1_FEATURE_MANIFEST)
    return combined


def main():
    parser=argparse.ArgumentParser(description='V2 three-channel extraction with unchanged V1 feature methods.')
    parser.add_argument('--mode',choices=['plan','smoke','extract','combine'],required=True)
    parser.add_argument('--limit',type=int,default=None,help='Only for smoke, max 12')
    args=parser.parse_args()
    if args.limit is not None and (args.mode!='smoke' or args.limit<=0 or args.limit>12):
        parser.error('--limit only allowed for smoke, between 1 and 12')
    if args.mode=='combine':
        combine(read_audio_manifest()); return
    df=read_audio_manifest()
    print('V2 manifest:',V2_AUDIO_MANIFEST)
    print('V2 audio:',len(df),'| counts:',df.split.value_counts().to_dict())
    print('RIR seen rooms:',sorted(set(df.loc[df.split!='test','rir_room'])))
    print('RIR unseen rooms:',sorted(set(df.loc[df.split=='test','rir_room'])))
    print('Reusing V1 train stats (no fitting on V2 val/test):')
    # Verify stats before any WAV feature computation.
    stats=load_stats()
    print('Normalization stats shapes:',[tuple(x.shape) for x in stats])
    if args.mode=='plan':
        print('PLAN PASSED: NO FEATURE WRITES')
        return
    extract(df,smoke=(args.mode=='smoke'),limit=args.limit)

if __name__=='__main__':
    main()