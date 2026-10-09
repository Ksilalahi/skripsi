import torch, torchaudio
import torch.nn.functional as F

def load_audio_4s(path, training=False, target_sr=16000, seconds=4):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    target = target_sr * seconds
    n = wav.shape[-1]
    if n < target:
        wav = F.pad(wav, (0, target - n))
    elif n > target:
        if training:
            start = torch.randint(0, n - target + 1, (1,)).item()
        else:
            start = (n - target) // 2
        wav = wav[:, start:start + target]

    peak = wav.abs().max().clamp_min(1e-6)
    return (wav / peak).float()