"""Envelope-wobble isolation probe: representation vs sampling-dynamics.

Loads a trained RF STFT-transformer checkpoint + its MATPAC conditioner, picks the
most *sustained* real samples (low envelope variation), and for each generates a
reconstruction (cond = MATPAC(real)) at several NFE counts and guidance scales.

Logic: RF steps=1 is ~the 1-NFE prediction MF would learn; steps=100 is near-exact
ODE integration. If the envelope wobble shrinks as steps rise (or 1-NFE is steadier),
the wobble is sampling-dynamics / integration error -> MF/Heun/more-steps fix it. If
wobble is flat across step counts and present at s=1.0, it's representation/loss.

Metric: windowed-RMS envelope over the sustained core; "wobble" = std of the
frame-to-frame difference of the log-envelope (octave-agnostic, mean-invariant).
Lower = steadier. Compared against the real sample's own envelope wobble.
"""

from __future__ import annotations

import sys
from pathlib import Path

import soundfile as sf
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from backbone.factory import build_backbone  # noqa: E402
from data.audio_dataset import AudioDirectoryDataset, collate_audio_batch  # noqa: E402
from emb.factory import build_embedding  # noqa: E402
from flow.fm import RectifiedFlow  # noqa: E402

CKPT = ROOT / "runs/fm-oneshots-mars-stft-200k/checkpoints/step_00200000.pt"
OUT = ROOT / "experiments/envelope_diag/out"
N_CANDIDATES = 96
N_PICK = 6
WIN = 1024
HOP = 256


def rms_env(x: torch.Tensor) -> torch.Tensor:
    """Windowed-RMS envelope of a [T] waveform."""
    frames = x.unfold(0, WIN, HOP)  # [n, WIN]
    return frames.pow(2).mean(-1).clamp_min(1e-12).sqrt()


def wobble(x: torch.Tensor) -> float:
    """Std of frame-to-frame log-envelope diff over the sustained core (central 60%)."""
    env = rms_env(x)
    n = env.shape[0]
    core = env[int(0.2 * n) : int(0.8 * n)]
    if core.numel() < 4:
        return float("nan")
    logd = torch.log(core[1:]) - torch.log(core[:-1])
    return float(logd.std())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    print(f"ckpt={CKPT.name} backbone.dim={cfg.backbone.block.dim} "
          f"stft={dict(cfg.backbone.stft)} rms_lift={cfg.data.get('rms_lift', False)}")

    model = build_backbone(cfg.backbone)
    model.load_state_dict(ckpt["model"])
    if ckpt.get("ema") is not None:  # match logged-sample weights
        shadow = ckpt["ema"]["shadow"]
        st = model.state_dict()
        st.update({k: v for k, v in shadow.items() if k in st})
        model.load_state_dict(st)
    model.eval().to(device)

    conditioner = build_embedding(OmegaConf.to_container(cfg.conditioner, resolve=True))

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    for key in ("bucket_pool_multiplier", "augmentations", "rms_lift", "lift_scale", "rms_target"):
        data_cfg.pop(key, None)
    dataset = AudioDirectoryDataset(**data_cfg)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_audio_batch)

    sr = int(cfg.data.sample_rate)
    lift_scale = float(cfg.data.get("lift_scale", 3.0)) if bool(cfg.data.get("rms_lift", False)) else 1.0
    flow = RectifiedFlow()

    # Rank candidates by sustainedness: long + low real-wobble.
    cands = []
    for i, batch in enumerate(loader):
        if i >= N_CANDIDATES:
            break
        wav = batch["audio"][0, 0]
        if wav.shape[0] < sr // 2:  # need >= ~0.5s to judge a sustain
            continue
        cands.append((wobble(wav), batch))
    cands = [c for c in cands if c[0] == c[0]]  # drop nan
    cands.sort(key=lambda c: c[0])
    picks = cands[:N_PICK]
    print(f"\npicked {len(picks)} most-sustained of {len(cands)} candidates\n")

    configs = [("s1.0_n1", 1, 1.0), ("s3.5_n1", 1, 3.5), ("s3.5_n4", 4, 3.5),
               ("s3.5_n25", 25, 3.5), ("s3.5_n100", 100, 3.5), ("s1.0_n25", 25, 1.0)]
    header = f"{'sample':28s} {'real':>7s} " + " ".join(f"{c[0]:>9s}" for c in configs)
    print(header)
    print("-" * len(header))

    rows = []
    for k, (rw, batch) in enumerate(picks):
        audio = batch["audio"].to(device)
        length = audio.shape[-1]
        with torch.no_grad():
            cond = conditioner(audio, sample_rate=sr, audio_lengths=batch["audio_lengths"])
        name = Path(batch["paths"][0]).stem[:26] if "paths" in batch else f"sample{k}"
        sf.write(OUT / f"{k:02d}_{name}_real.wav", audio[0].cpu().T.numpy(), sr)
        line = f"{name:28s} {rw:7.3f} "
        cells = []
        for tag, steps, s in configs:
            with torch.no_grad():
                gen = flow.sample(model, shape=(1, audio.shape[1], length), cond=cond,
                                  steps=steps, method="euler", guidance_scale=s, lift_scale=lift_scale)
            gw = wobble(gen[0, 0].float().cpu())
            cells.append(f"{gw:9.3f}")
            sf.write(OUT / f"{k:02d}_{name}_{tag}.wav", gen[0].clamp(-1, 1).cpu().T.numpy(), sr)
        rows.append((rw, [float(c) for c in cells]))
        print(line + " ".join(cells))

    # Aggregate: mean wobble per config vs real.
    import statistics as st
    real_mean = st.mean(r[0] for r in rows)
    print("-" * len(header))
    agg = [st.mean(r[1][j] for r in rows) for j in range(len(configs))]
    print(f"{'MEAN':28s} {real_mean:7.3f} " + " ".join(f"{a:9.3f}" for a in agg))
    print(f"\nreal envelope wobble = {real_mean:.3f}; ratios gen/real:")
    for (tag, _, _), a in zip(configs, agg, strict=True):
        print(f"  {tag:10s} {a / real_mean:5.2f}x")


if __name__ == "__main__":
    main()
