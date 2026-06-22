"""Probe the maximum train-step batch size that fits on the current GPU.

Mirrors RFTrainer.training_step memory cost: builds the backbone + MATPAC
conditioner from a Hydra experiment config, then runs a forward + backward at
max-length audio (cfg.data.max_seconds @ cfg.data.sample_rate) under AMP.

Doubles the batch size until OOM, then bisects between the last success and the
first failure. Reports the largest batch size that completed a full step.

Example:
    PYTHONPATH=src uv run python scripts/probe_max_batch.py \
        --config-name experiment/fm_oneshots_mars
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))  # so `import matpac` (top-level pkg) resolves

from backbone.factory import build_backbone  # noqa: E402
from emb.factory import build_embedding  # noqa: E402
from flow.fm import EPS, RectifiedFlow  # noqa: E402
from losses.audio import mr_stft_loss  # noqa: E402


def _load_cfg(config_name: str, overrides: list[str] | None = None):
    config_dir = str(REPO_ROOT / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name=config_name, overrides=list(overrides or []))


def _build(cfg, device: torch.device):
    model = build_backbone(cfg.backbone).to(device).train()
    conditioner_cfg = OmegaConf.to_container(cfg.get("conditioner", {"type": "none"}), resolve=True)
    if conditioner_cfg.get("type") == "matpac":
        conditioner_cfg["device"] = str(device)
    conditioner = build_embedding(conditioner_cfg, device=device)
    if conditioner is not None:
        conditioner = conditioner.to(device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    return model, conditioner, optimizer


def _step(model, conditioner, optimizer, cfg, batch_size: int, num_samples: int, device: torch.device) -> None:
    """One full forward + backward + optimizer step at max-length audio."""
    sample_rate = int(cfg.data.sample_rate)
    audio = torch.randn(batch_size, 1, num_samples, device=device)
    audio_lengths = torch.full((batch_size,), num_samples, dtype=torch.long, device=device)

    with torch.no_grad():
        cond = (
            conditioner(audio, sample_rate=sample_rate, audio_lengths=audio_lengths)
            if conditioner is not None
            else None
        )

    method = RectifiedFlow()
    # Match RFTrainer._sample_t (logit_normal default).
    t = torch.randn(batch_size, device=device).sigmoid().clamp(EPS, 1.0 - EPS)
    x_t, t, x1 = method.train_tuple(audio, t=t)

    optimizer.zero_grad(set_to_none=True)
    amp_enabled = bool(cfg.train.get("amp", True)) and device.type == "cuda"
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
        pred = method._predict(model, x_t, t=t, cond=cond, length=audio.shape[-1], with_aux=False)[0]
        loss_cfg = cfg.loss
        total, _ = method.loss(
            pred,
            x1,
            x_t,
            t,
            space=str(loss_cfg.get("loss_space", "v")),
            loss_type=str(loss_cfg.get("primary", "mse")),
        )
        mr_stft_weight = float(loss_cfg.get("mr_stft_weight", 0.0))
        if mr_stft_weight > 0.0:
            total = total + mr_stft_weight * mr_stft_loss(
                pred, x1, log_weight=float(loss_cfg.get("mr_stft_log_weight", 0.0))
            )
    total.backward()
    grad_clip = float(cfg.train.get("grad_clip", 0.0))
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    torch.cuda.synchronize(device) if device.type == "cuda" else None


def _try_batch(cfg, batch_size: int, num_samples: int, device: torch.device) -> tuple[bool, float, str]:
    """Returns (ok, peak_gib, error_message). Locals freed on return; caller must empty_cache."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        model, conditioner, optimizer = _build(cfg, device)
        _step(model, conditioner, optimizer, cfg, batch_size, num_samples, device)
        peak = (
            torch.cuda.max_memory_allocated(device) / (1024**3)
            if device.type == "cuda"
            else 0.0
        )
        return True, peak, ""
    except torch.cuda.OutOfMemoryError as exc:
        return False, 0.0, f"OOM: {exc}"
    except RuntimeError as exc:
        msg = str(exc)
        if "out of memory" in msg.lower():
            return False, 0.0, f"OOM: {msg}"
        raise


def _attempt(cfg, batch_size: int, num_samples: int, device: torch.device) -> tuple[bool, float, str]:
    """Wrap _try_batch + post-return cleanup so each attempt starts from a clean CUDA cache."""
    result = _try_batch(cfg, batch_size, num_samples, device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def probe(config_name: str, start: int, ceiling: int, overrides: list[str] | None = None) -> int:
    cfg = _load_cfg(config_name, overrides)
    result = _probe_cfg(cfg, start, ceiling)
    print(f"Set with:  train.dataloader.batch_size={result}")
    return result


BACKBONES = ["stft_transformer"]


def _swap_backbone(cfg, backbone_name: str):
    """Replace cfg.backbone with the named backbone config + cond_dim=384."""
    backbone_path = REPO_ROOT / "configs" / "backbone" / f"{backbone_name}.yaml"
    new_backbone = OmegaConf.load(backbone_path)
    OmegaConf.set_struct(new_backbone, False)
    new_backbone.conditioning.cond_dim = 384  # match MATPAC's native dim
    # fm_baseline.yaml wires backbone.conditioning.time_scale = ${flow.t_encoding.scale}
    # via interpolation in YAML; after replacing the subtree we set it explicitly
    # (and force-add the key since some backbone yamls don't declare it).
    new_backbone.conditioning.time_scale = float(cfg.flow.t_encoding.scale)
    OmegaConf.set_struct(cfg, False)
    cfg.backbone = new_backbone
    return cfg


def _probe_cfg(cfg, start: int, ceiling: int) -> int:
    """Run the doubling+bisect probe on an already-constructed cfg."""
    device = torch.device(cfg.train.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    num_samples = int(round(float(cfg.data.max_seconds) * float(cfg.data.sample_rate)))
    print(f"Probing on {device} at {num_samples} samples ({cfg.data.max_seconds}s @ {cfg.data.sample_rate}Hz)")
    print(f"AMP: {bool(cfg.train.get('amp', True))} | backbone: {cfg.backbone.get('name', '?')}")

    last_ok, first_fail = 0, None
    bs = max(1, start)
    while bs <= ceiling:
        ok, peak, err = _attempt(cfg, bs, num_samples, device)
        status = f"OK   peak={peak:6.2f} GiB" if ok else err.splitlines()[0]
        print(f"  batch={bs:4d}  {status}")
        if ok:
            last_ok, bs = bs, bs * 2
        else:
            first_fail = bs
            break

    if first_fail is None:
        print(f"\nReached ceiling without OOM. Max tested OK: {last_ok}")
        return last_ok

    lo, hi = last_ok, first_fail
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok, peak, err = _attempt(cfg, mid, num_samples, device)
        status = f"OK   peak={peak:6.2f} GiB" if ok else err.splitlines()[0]
        print(f"  batch={mid:4d}  {status}")
        if ok:
            lo = mid
        else:
            hi = mid

    print(f"\nMax batch size at {cfg.data.max_seconds}s: {lo}")
    return lo


def probe_all_backbones(config_name: str, start: int, ceiling: int, skip: list[str] | None = None) -> dict[str, tuple[int, bool]]:
    """Probe each backbone with MATPAC conditioning (cond_dim forced to 384).

    Returns {name: (max_bs, amp_used)}. If the backbone hits a BFloat16 dtype
    error (torch.stft/istft don't support bf16), automatically retries with
    train.amp=false and notes that in the result.
    """
    skip_set = set(skip or [])
    results: dict[str, tuple[int, bool]] = {}
    for name in BACKBONES:
        if name in skip_set:
            print(f"\n(skipping {name})")
            continue
        print(f"\n{'=' * 60}\nbackbone: {name}\n{'=' * 60}")
        amp_used = True
        max_bs = -1
        try:
            cfg = _load_cfg(config_name)
            cfg = _swap_backbone(cfg, name)
            max_bs = _probe_cfg(cfg, start, ceiling)
        except RuntimeError as exc:
            if "BFloat16" not in str(exc):
                print(f"  ERROR probing {name}: RuntimeError: {exc}")
            else:
                print(f"  bf16-incompatible op ({exc.__class__.__name__}); retrying with AMP off…")
                amp_used = False
                try:
                    cfg = _load_cfg(config_name)
                    cfg = _swap_backbone(cfg, name)
                    OmegaConf.set_struct(cfg.train, False)
                    cfg.train.amp = False
                    max_bs = _probe_cfg(cfg, start, ceiling)
                except Exception as inner:
                    print(f"  ERROR (AMP-off retry) {name}: {type(inner).__name__}: {inner}")
        except Exception as exc:
            print(f"  ERROR probing {name}: {type(exc).__name__}: {exc}")
        results[name] = (max_bs, amp_used)

    print(f"\n{'=' * 60}\nSummary (max batch at 4.0s @ 48kHz on a 24 GiB 4090)\n{'=' * 60}")
    for name, (max_bs, amp) in results.items():
        amp_note = "" if amp else " (AMP OFF — bf16-incompatible STFT/ISTFT)"
        print(f"  {name:24s}  {max_bs}{amp_note}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True, help="e.g. experiment/fm_oneshots_mars")
    parser.add_argument("--start", type=int, default=4)
    parser.add_argument("--ceiling", type=int, default=1024)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override; can be passed multiple times. e.g. --override backbone@backbone=vocos",
    )
    parser.add_argument(
        "--all-backbones",
        action="store_true",
        help="Probe each backbone in configs/backbone/ with cond_dim=384 (MATPAC).",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Skip these backbones in --all-backbones mode. Repeatable.",
    )
    args = parser.parse_args()
    if args.all_backbones:
        probe_all_backbones(args.config_name, args.start, args.ceiling, skip=args.skip)
    else:
        probe(args.config_name, args.start, args.ceiling, overrides=args.override)


if __name__ == "__main__":
    main()
