# MeanFlow Implementation Spec

> **For the implementing agent:** work task-by-task in order; each task ends with a runnable
> verification and a commit. Read `CLAUDE.md` and `docs/fm_training_notes.md` first. Match the
> existing code style (`src/trainer.py`, `src/flow/fm.py`): compact, minimal comments, research
> code not deployment code. Do not add abstractions, registries, or options beyond this spec.

**Goal:** add MeanFlow (iMF-style, x-prediction) training as a sibling of the existing
rectified-flow path, fine-tunable from an RF checkpoint, with **CFG as a conditioning input**
(iMF flexible guidance — ω always conditioned; the guidance interval either a fixed training-time
mask or conditioned on both bounds, switchable by one flag) shared by RF and MF, and a swappable
`du/dt` backend (`dde` finite-difference default, `jvp` optional).

**Architecture:** a `MeanFlow(RectifiedFlow)` method class in `src/flow/mf.py` (math + jump
sampler), an `MFTrainer(RFTrainer)` in `src/trainer.py` (timestep-pair sampling, RF/MF
micro-batch branching, warm start), zero-init gap/guidance/interval embeddings in the Transformer
backbone, and a `configs/flow/mf.yaml` group. The flexible-CFG target lives on `RectifiedFlow`
so `RFTrainer` gets it too.

**Why CFG is conditioned, not baked to a constant.** ω-as-conditioning keeps inference at 1
forward (it is *not* runtime two-pass CFG) while letting you sweep guidance at inference without
retraining — iMF's finding is that the optimal ω shifts with model size, training length and NFE,
so a frozen constant is always wrong on some axis. The **interval is baked into the 1-NFE model**
regardless of mode: the one-step jump `x̂ = z + u_θ(z,0,1)` is the *average* of the guided
instantaneous velocity over the whole `[0,1]` path, and the interval mask (gating the target by
start-time `t`) decides which part of that path deposits guidance into the single jump. So the
interval genuinely shapes 1-NFE output; conditioning on it (both bounds) makes it tunable at
inference, fixed mode bakes one choice. Default is fixed; flip `condition_interval` to make it
tunable.

> **Superseded (2026-06-19):** the original recipe below trained MeanFlow with a *single* head and
> took the `du/dt` tangent from the u-head's h=0 boundary eval. That collapsed (noise on silence;
> `‖h·du/dt‖` blow-up — see `docs/fm_training_notes.md` 2026-06-17). The auxiliary v-head listed as
> out-of-scope here is now **built** as the fix; see §6 for the twin-head recipe that overrides the
> Task 4 `_mf_step` tangent and the "no v-head" scope line.

**Out of scope (do not build):** the Triton flash-attention-JVP kernel itself (only the seam
for it), ConvNeXt-backbone support, in-context conditioning (the AdaLN→token swap iMF made to fit
6 conditions — our fallback only *if* conditioned-interval AdaLN overloads, see Task 1 note), an
auxiliary v-head (← now built, see §6), perceptual embedder losses, annealed `rf_fraction` schedules. The `jvp` mode is
best-effort: `torch.func.jvp` through `torch.stft`/`istft` may lack forward-AD support — if it
errors, leave the mode in place (it is the seam for the kernel port) and note it; `dde` is the
supported default.

---

## 1. Conventions and math (read carefully — papers use the opposite time axis)

Repo convention (`src/flow/fm.py`): `x_t = (1−t)·x0 + t·x1`, `x0` = noise at `t=0`, `x1` = data
at `t=1`. Velocity `v = x1 − x0`. The model predicts **x̂ (clean data)**; the v-space loss
converts via `target_to_v(x̂, x_t, t) = (x̂ − x_t)/(1−t)`. MeanFlow/iMF papers put noise at `t=1`;
every formula below is already mirrored into the repo convention — implement these, not the
papers'.

**Average velocity.** For a jump from current time `t` to destination `s ∈ [t, 1]`, gap
`h = s − t`:

```
u(z_t, t, s) = (1/h) ∫_t^s v(z_τ, τ) dτ
```

The network keeps predicting x̂, now as a field `x̂_θ(z, t, h)`, and u is recovered with the
**existing conversion** (this is what makes the RF warm start exact):

```
u_θ(z, t, h) = (x̂_θ(z, t, h) − z) / (1 − t)        # == target_to_v(x̂_θ, z, t)
```

Boundary checks: at `h = 0`, `u_θ = v̂` (the RF model, unchanged). At `s = 1` (`h = 1 − t`),
displacement `= (1−t)·u = x̂ − z`, so `x̂` is the trajectory endpoint — **1-NFE sampling is one
forward pass returning x̂**.

**MeanFlow identity** (differentiate `h·u = ∫_t^s v dτ` w.r.t. `t` along the trajectory, `s`
fixed; lower-limit Leibniz gives `−v`):

```
v(z_t, t) = u − h·(du/dt)        where du/dt is the total derivative along the trajectory
```

With network arguments `(z, t, h)` and `s` fixed, the trajectory tangent is
`(dz/dt, dt/dt, dh/dt) = (v, 1, −1)` — note `dh/dt = −1`.

**Training loss** (iMF form: network-independent target, stop-grad on du/dt, adaptive weight):

```
V_θ      = u_θ(z_t, t, h) − h · sg[du_θ/dt]
Δ        = V_θ − v_tgt                                  # v_tgt below
per_i    = mean(Δ_i²)                                   # per-sample
loss     = mean_i [ per_i / (sg(per_i) + c)^p ]          # p=1, c=1e-3; p=0 → plain MSE
```

**CFG as conditioning** (iMF flexible guidance; the same construction serves RF and MF). ω is a
per-row conditioning input fed to the network (sampled in training, set at inference), so guidance
stays **one forward pass**. The guided per-sample velocity target is:

```
v_cond = (x1 − x_t)/(1−t)                               # == target_to_v(x1, x_t, t)
v̂_c    = target_to_v(model(x_t, t, cond=c,  ω, t_lo, t_hi))      # no-grad boundary (h=0)
v̂_∅    = target_to_v(model(x_t, t, cond=∅,  ω, t_lo, t_hi))      # no-grad boundary (h=0)
mask    = 1[ t_lo ≤ t ≤ t_hi ]
v_g     = v_cond + (1 − 1/ω) · mask · (v̂_c − v̂_∅)
```

where `∅` is **zeroed cond** (`torch.zeros_like(cond)`) — the same null convention as
`_cfg_dropout`; never `cond=None`. `ω = 1` ⇒ correction vanishes ⇒ `v_g = v_cond`. Dropped-cond
rows self-resolve (cond already zeroed ⇒ `v̂_c = v̂_∅` ⇒ correction 0), so no special-casing.

**Interval, two modes** (one flag `flow.guidance.condition_interval`):

- **fixed** (default): `t_lo, t_hi` are config constants (`guidance.interval = [0.0, 0.8]` by
  default, based on the repository RF sweep); the network is **not** conditioned on the interval
  (only on ω). The fixed mask shapes the targets and bakes one interval into the model. 3 AdaLN
  scalars (t, h, ω).
- **conditioned**: per-row `t_lo ~ U(0, 0.5)`, `t_hi ~ U(0.5, 1)` sampled and fed to the network
  (so both bounds are tunable at inference); mask uses the per-row bounds. 5 AdaLN scalars
  (t, h, ω, t_lo, t_hi) — see the Task 1 AdaLN-overload note.

ω sampled `log-uniform[1, omega_max]` (biased toward ω=1; iMF's power-law). Interval bounds in
repo t-axis (0=noise, 1=data): lowering `t_hi` switches guidance off nearer data, raising `t_lo`
off nearer noise.

**Tangent for the MF JVP/DDE:**

- guidance **on**: `v_tgt = v_g`, tangent = the model's conditioned boundary velocity
  `v̂_c = target_to_v(model(x_t, t, cond=c, ω, t_lo, t_hi), x_t, t)`. Keep the stochastic
  guided velocity as the regression target only; using it as the tangent reintroduces the
  conditional-velocity variance that iMF removes.
- guidance **off** (`flow.guidance` null/disabled): `v_tgt = v_cond`, tangent = the model's own
  boundary (`h=0`) velocity `target_to_v(model(x_t, t, cond))` — iMF's low-variance marginal
  tangent; do **not** use `v_cond` as the tangent.
- `cond is None` (unconditional run): `v_tgt = v_cond`, tangent = boundary call.

**du/dt backends** (the swap the config exposes):

- `dde` (default): central finite difference along the tangent, two **no-grad** forwards, fp32
  arithmetic on the outputs. With scalar `ε = dde_eps` (the t/h sampler guarantees `h ≥ ε`):

  ```
  du/dt ≈ [u_θ(z+εv̄, t+ε, h−ε) − u_θ(z−εv̄, t−ε, h+ε)] / 2ε      # v̄ = tangent
  ```

  Overshooting `t ± ε` slightly past `[EPS, 1−EPS]` is harmless; do not add edge clamps.
  `target_to_v` must preserve the sign of `1−t` when flooring its magnitude near zero so the
  `t+ε > 1` probe remains valid even with `adaptive_p=0`. ω/t_lo/t_hi are held fixed across the
  perturbation (only z, t, h move).
- `jvp`: `torch.func.jvp(u_fn, (z, t, h), (v̄, 1, −1))`. One isolated call site — this is where
  the Triton flash-attention-JVP kernel plugs in later.

**Sampling** (Euler jumps on u; `guidance_scale` is the inference ω, `guidance_t_lo/​t_hi` the
inference interval — all conditioning inputs, single forward):

```
grid = linspace(EPS, 1−EPS, steps+1)
z ← z + (grid[i+1] − grid[i]) · u_θ(z, grid[i], h = grid[i+1] − grid[i] ; ω, t_lo, t_hi)
```

---

## 2. File map

| File | Change |
|---|---|
| `src/backbone/transformer.py` | zero-init `gap_embed` (h), `omega_embed` (ω), `lo_embed`/`hi_embed` (interval); `h, omega, t_lo, t_hi` kwargs |
| `src/flow/fm.py` | `v_to_target` + `guided_velocity_target` on `RectifiedFlow` |
| `src/flow/mf.py` | **new** — `MeanFlow(RectifiedFlow)`: `u_and_dudt`, `mf_loss`, jump `sample`/`generate` |
| `src/flow/factory.py` | register `mean_flow` |
| `src/trainer.py` | `RFTrainer`: `_lift` + `_guidance_inputs` + guided-target hook; **new** `MFTrainer(RFTrainer)` |
| `configs/flow/fm.yaml` | add `guidance: null` |
| `configs/flow/mf.yaml` | **new** flow group |
| `configs/experiment/mf_oneshots_mars_stft.yaml` | **new** — the real run (init_from RF ckpt) |
| `configs/experiment/mf_smoke.yaml` | **new** — CPU smoke |
| `tests/test_mean_flow.py` | **new** |
| `tests/test_backbone.py` | add gap/guidance/interval warm-start tests |

---

## 3. Tasks

### Task 1 — gap / guidance / interval embeddings in the Transformer backbone

**Files:** modify `src/backbone/transformer.py`, test in `tests/test_backbone.py`.

> **AdaLN-overload watch-item (not a blocker).** Conditioned-interval mode sums 5 scalars
> (t, h, ω, t_lo, t_hi) into the AdaLN time path. iMF reported adaLN-zero degrading when ~6
> heterogeneous conditions are summed, which is why it switched to in-context conditioning. If a
> conditioned-interval run shows worse conditioning fidelity than fixed-interval at matched steps
> (recon cosine / retrieval), the fallback is in-context conditioning (out of scope here) or
> `condition_interval: false`. Fixed mode (3 scalars) is unaffected.

- [ ] **Step 1: failing test** — append to `tests/test_backbone.py`:

```python
def _tiny_transformer(gap_embed=False, guidance_embed=False, interval_embed=False):
    from backbone.transformer import Transformer

    return Transformer(
        channels=1,
        stft={"n_fft": 64, "hop_length": 16, "win_length": 64},
        block={"dim": 32, "depth": 1, "heads": 2},
        conditioning={
            "cond_dim": 16, "embed_dim": 16,
            "gap_embed": gap_embed, "guidance_embed": guidance_embed, "interval_embed": interval_embed,
        },
        sample_rate=8000,
    )


def test_aux_embeds_zero_init_are_exact_rf_warm_start():
    torch.manual_seed(0)
    rf = _tiny_transformer()
    mf = _tiny_transformer(gap_embed=True, guidance_embed=True, interval_embed=True)
    missing, unexpected = mf.load_state_dict(rf.state_dict(), strict=False)
    assert not unexpected
    assert all(any(tag in key for tag in ("gap_embed", "omega_embed", "lo_embed", "hi_embed")) for key in missing)

    x = torch.randn(2, 1, 256)
    t = torch.tensor([0.3, 0.7])
    cond = torch.randn(2, 16)
    # every zero-init aux path reproduces the RF model exactly, for any aux input or None
    kw = dict(h=torch.tensor([0.5, 0.2]), omega=torch.tensor([3.0, 5.0]),
              t_lo=torch.tensor([0.1, 0.2]), t_hi=torch.tensor([0.8, 0.9]))
    assert torch.allclose(mf(x, t=t, cond=cond, **kw), rf(x, t=t, cond=cond), atol=1e-6)
    assert torch.allclose(mf(x, t=t, cond=cond), rf(x, t=t, cond=cond), atol=1e-6)


def test_aux_inputs_without_embeds_raise():
    rf = _tiny_transformer()
    x, t, cond = torch.randn(1, 1, 256), torch.tensor([0.5]), torch.randn(1, 16)
    with pytest.raises(ValueError, match="gap_embed"):
        rf(x, t=t, h=torch.tensor([0.5]), cond=cond)
    with pytest.raises(ValueError, match="guidance_embed"):
        rf(x, t=t, omega=torch.tensor([3.0]), cond=cond)
    with pytest.raises(ValueError, match="interval_embed"):
        rf(x, t=t, t_hi=torch.tensor([0.8]), cond=cond)
```

- [ ] **Step 2:** `uv run pytest tests/test_backbone.py -k "aux" -v` — expect FAIL (unexpected kwarg `h`).
- [ ] **Step 3: implement.** Add a module-level helper above `class Transformer` in
`src/backbone/transformer.py`:

```python
def _zero_init_time_embed(cond_dim: int, time_scale: float) -> "TimeEmbedding":
    # Zero-init output ⇒ an RF checkpoint loads (strict=False) as an exact warm start: the
    # extra (h / ω / interval) signal contributes nothing until trained.
    emb = TimeEmbedding(cond_dim, time_scale=time_scale)
    nn.init.zeros_(emb.mlp[-1].weight)
    nn.init.zeros_(emb.mlp[-1].bias)
    return emb
```

In `Transformer.__init__`, after `self.time_embed = ...`:

```python
ts = conditioning.get("time_scale", 1.0)
self.gap_embed = _zero_init_time_embed(self.cond_dim, ts) if conditioning.get("gap_embed", False) else None
self.omega_embed = _zero_init_time_embed(self.cond_dim, conditioning.get("omega_scale", 1.0)) if conditioning.get("guidance_embed", False) else None
self.lo_embed = _zero_init_time_embed(self.cond_dim, ts) if conditioning.get("interval_embed", False) else None
self.hi_embed = _zero_init_time_embed(self.cond_dim, ts) if conditioning.get("interval_embed", False) else None
```

In `forward`, change the signature to
`(self, x, t=None, h=None, cond=None, omega=None, t_lo=None, t_hi=None, length=None)` and replace
the `t_embed = ...` line with:

```python
t_embed = self.time_embed(t) if t is not None else None
if h is not None and self.gap_embed is None:
    raise ValueError("Backbone got `h` but conditioning.gap_embed is false")
if omega is not None and self.omega_embed is None:
    raise ValueError("Backbone got `omega` but conditioning.guidance_embed is false")
if (t_lo is not None or t_hi is not None) and self.lo_embed is None:
    raise ValueError("Backbone got interval bounds but conditioning.interval_embed is false")
if t_embed is not None:
    # all aux signals summed into the time path BEFORE cond_combine's RMS-norm: a separate
    # normalised path would rescale a zero-init branch to unit RMS the moment it wakes up.
    if self.gap_embed is not None:
        t_embed = t_embed + self.gap_embed(torch.zeros_like(t) if h is None else h)
    if self.omega_embed is not None:
        t_embed = t_embed + self.omega_embed(torch.ones_like(t) if omega is None else omega)
    if self.lo_embed is not None:
        t_embed = t_embed + self.lo_embed(torch.zeros_like(t) if t_lo is None else t_lo)
        t_embed = t_embed + self.hi_embed(torch.ones_like(t) if t_hi is None else t_hi)
```

- [ ] **Step 4:** `uv run pytest tests/test_backbone.py -v` — all pass (the `None` defaults keep every existing call site working).
- [ ] **Step 5:** commit `mf: zero-init gap/guidance/interval embeddings in Transformer`.

### Task 2 — flow math: `MeanFlow`, guided-CFG target, factory

**Files:** create `src/flow/mf.py`, modify `src/flow/fm.py`, `src/flow/factory.py`; test in
`tests/test_mean_flow.py`.

- [ ] **Step 1: failing tests** — create `tests/test_mean_flow.py`:

```python
from __future__ import annotations

import pytest
import torch

from flow.fm import EPS, RectifiedFlow
from flow.mf import MeanFlow


class ConstantVelocityModel(torch.nn.Module):
    """x̂ = x + (1−t)·c: a perfectly straight field with velocity c at every (z, t, h)."""

    def __init__(self, c: torch.Tensor):
        super().__init__()
        self.register_buffer("c", c)

    def forward(self, x, t=None, h=None, cond=None, omega=None, t_lo=None, t_hi=None, length=None):
        del h, cond, omega, t_lo, t_hi, length
        return x + (1.0 - t.view(-1, 1, 1)) * self.c


class CurvedModel(torch.nn.Module):
    """Smooth nonlinear (z, t, h)-dependence for dde-vs-jvp agreement checks."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)

    def forward(self, x, t=None, h=None, cond=None, omega=None, t_lo=None, t_hi=None, length=None):
        del cond, omega, t_lo, t_hi, length
        t = t.view(-1, 1, 1)
        h = torch.zeros_like(t) if h is None else h.view(-1, 1, 1)
        mix = torch.tanh(self.proj(x))  # x is [B, 1, 8]; Linear over the length dim
        return x * torch.cos(t + h) + (1.0 - t) * mix * (1.0 + h)


class CondShiftModel(torch.nn.Module):
    """x̂ = x_t + (1−t)·(1 + mean(cond)): boundary v_θ = 1 + mean(cond), so ∅ (zeros) gives 1."""

    def forward(self, x, t=None, h=None, cond=None, omega=None, t_lo=None, t_hi=None, length=None):
        del h, omega, t_lo, t_hi, length
        return x + (1.0 - t.view(-1, 1, 1)) * (1.0 + cond.mean(dim=1).view(-1, 1, 1))


def _pair(batch=4, length=8):
    torch.manual_seed(0)
    c = torch.randn(1, 1, length)
    x0 = torch.randn(batch, 1, length)
    return c, x0, x0 + c  # v_cond = c exactly


def test_constant_field_satisfies_mf_identity():
    flow = MeanFlow()
    c, x0, x1 = _pair()
    model = ConstantVelocityModel(c)
    t = torch.tensor([0.1, 0.4, 0.6, 0.8])
    h = torch.tensor([0.3, 0.2, 0.3, 0.1])
    x_t, t, _ = flow.train_tuple(x1, t=t, noise=x0)
    v_tgt = flow.target_to_v(x1, x_t, t)
    for mode in ("dde", "jvp"):
        u, dudt, x_pred = flow.u_and_dudt(model, x_t, t, h, v_tgt, cond=None, length=8, mode=mode, dde_eps=5e-3)
        V = u - flow._time_like(h, u) * dudt.detach()
        assert torch.allclose(V, v_tgt.expand_as(V), atol=1e-4), mode
        loss, terms = flow.mf_loss(V, v_tgt.expand_as(V), p=1.0, c=1e-3)
        assert loss < 1e-6
        assert "mf_mse" in terms


def test_dde_matches_jvp_on_curved_model():
    flow = MeanFlow()
    torch.manual_seed(1)
    model = CurvedModel()
    x_t = torch.randn(4, 1, 8)
    t = torch.tensor([0.2, 0.4, 0.6, 0.8])
    h = torch.tensor([0.1, 0.2, 0.15, 0.1])
    tangent = torch.randn_like(x_t)
    _, dudt_dde, _ = flow.u_and_dudt(model, x_t, t, h, tangent, None, 8, mode="dde", dde_eps=1e-3)
    _, dudt_jvp, _ = flow.u_and_dudt(model, x_t, t, h, tangent, None, 8, mode="jvp", dde_eps=1e-3)
    assert torch.allclose(dudt_dde, dudt_jvp, atol=1e-2, rtol=1e-2)


def test_u_carries_grad_and_dudt_does_not():
    flow = MeanFlow()
    model = CurvedModel()
    x_t, t, h = torch.randn(2, 1, 8), torch.tensor([0.3, 0.5]), torch.tensor([0.2, 0.2])
    u, dudt, _ = flow.u_and_dudt(model, x_t, t, h, torch.randn_like(x_t), None, 8, mode="dde", dde_eps=5e-3)
    assert u.requires_grad and not dudt.requires_grad


def test_one_step_generate_returns_endpoint_prediction():
    flow = MeanFlow()
    c, _, _ = _pair()
    model = ConstantVelocityModel(c)
    noise = torch.randn(2, 1, 8)
    sample = flow.sample(model, shape=(2, 1, 8), noise=noise, steps=1)
    expected = noise + (1.0 - 2 * EPS) * c
    assert torch.allclose(sample, expected, atol=1e-5)


def test_generate_runs_with_omega_and_interval():
    flow = MeanFlow()
    model = ConstantVelocityModel(torch.zeros(1, 1, 8))  # ignores ω/interval
    out = flow.sample(model, shape=(2, 1, 8), steps=1, guidance_scale=3.0, guidance_t_lo=0.1, guidance_t_hi=0.8)
    assert out.shape == (2, 1, 8)


def test_guided_target_formula_fixed_interval():
    flow = RectifiedFlow()
    model = CondShiftModel()
    x1 = torch.randn(2, 1, 8)
    x_t, t, _ = flow.train_tuple(x1, t=torch.tensor([0.3, 0.9]))
    cond = torch.full((2, 4), 2.0)  # v̂_c = 3, v̂_∅ = 1 everywhere
    omega = torch.tensor([3.0, 3.0])
    v_cond = flow.target_to_v(x1, x_t, t)
    # fixed interval [0, 0.8], model not conditioned on bounds (model_lo/hi=None)
    v_g = flow.guided_velocity_target(model, x1, x_t, t, cond, omega, 8, 0.0, 0.8, None, None)
    assert torch.allclose(v_g[0], v_cond[0] + (1 - 1 / 3) * (3.0 - 1.0), atol=1e-5)  # t=0.3 inside
    assert torch.allclose(v_g[1], v_cond[1], atol=1e-5)  # t=0.9 outside → v_cond


def test_guided_target_conditioned_interval_per_row():
    flow = RectifiedFlow()
    model = CondShiftModel()
    x1 = torch.randn(2, 1, 8)
    x_t, t, _ = flow.train_tuple(x1, t=torch.tensor([0.3, 0.7]))
    cond = torch.full((2, 4), 2.0)
    omega = torch.tensor([3.0, 3.0])
    t_lo, t_hi = torch.tensor([0.0, 0.0]), torch.tensor([0.5, 0.5])  # row0 t=0.3 in; row1 t=0.7 out
    v_cond = flow.target_to_v(x1, x_t, t)
    v_g = flow.guided_velocity_target(model, x1, x_t, t, cond, omega, 8, t_lo, t_hi, t_lo, t_hi)
    assert torch.allclose(v_g[0], v_cond[0] + (1 - 1 / 3) * 2.0, atol=1e-5)
    assert torch.allclose(v_g[1], v_cond[1], atol=1e-5)


def test_guided_target_omega_one_and_unconditional_are_v_cond():
    flow = RectifiedFlow()
    x1 = torch.randn(2, 1, 8)
    x_t, t, _ = flow.train_tuple(x1, t=torch.rand(2))
    v_cond = flow.target_to_v(x1, x_t, t)
    g1 = flow.guided_velocity_target(CondShiftModel(), x1, x_t, t, torch.full((2, 4), 2.0), torch.ones(2), 8, 0.0, 1.0, None, None)
    assert torch.allclose(g1, v_cond, atol=1e-6)  # ω=1 → no guidance
    g2 = flow.guided_velocity_target(CondShiftModel(), x1, x_t, t, None, torch.full((2,), 3.0), 8, 0.0, 1.0, None, None)
    assert torch.allclose(g2, v_cond)  # cond None → v_cond
```

- [ ] **Step 2:** `uv run pytest tests/test_mean_flow.py -v` — expect FAIL (no `flow.mf`).
- [ ] **Step 3: implement.** Append to `src/flow/fm.py` (inside `RectifiedFlow`):

```python
def v_to_target(self, v: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return x_t + (1.0 - self._time_like(t, x_t)) * v

@torch.no_grad()
def guided_velocity_target(
    self,
    model,
    x1: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor | None,
    omega: torch.Tensor,
    length: int,
    interval_lo,
    interval_hi,
    model_t_lo: torch.Tensor | None = None,
    model_t_hi: torch.Tensor | None = None,
    return_boundary: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """iMF flexible CFG (ω as a conditioning input): target = v_cond + (1−1/ω)·1[t∈[lo,hi]]·(sg[v̂_c]−sg[v̂_∅]).
    ω is fed to the network so guidance is tunable in one forward at inference. `interval_lo/hi` are the
    mask bounds (scalars for fixed mode, per-row [B] tensors for conditioned mode). `model_t_lo/hi` are the
    bounds fed to the network when the interval is *conditioned* (None ⇒ network sees ω only, fixed mask).
    ∅ is zeroed cond (the trainer's cond-dropout convention), NOT cond=None. ω=1 ⇒ plain v_cond."""
    v_cond = self.target_to_v(x1, x_t, t)
    if cond is None:
        return v_cond
    v_c = self.target_to_v(model(x_t, t=t, cond=cond, omega=omega, t_lo=model_t_lo, t_hi=model_t_hi, length=length), x_t, t)
    v_u = self.target_to_v(model(x_t, t=t, cond=torch.zeros_like(cond), omega=omega, t_lo=model_t_lo, t_hi=model_t_hi, length=length), x_t, t)
    coeff = 1.0 - 1.0 / self._time_like(omega, v_cond)
    mask = self._time_like(((t >= interval_lo) & (t <= interval_hi)).to(v_cond.dtype), v_cond)
    v_guided = v_cond + coeff * mask * (v_c - v_u)
    return (v_guided, v_c) if return_boundary else v_guided
```

Create `src/flow/mf.py`:

```python
from __future__ import annotations

import torch

from flow.fm import EPS, RectifiedFlow


class MeanFlow(RectifiedFlow):
    """MeanFlow on the x-prediction backbone: u_θ(z, t, h) = (x̂_θ(z, t, h) − z)/(1−t),
    trained with the iMF v-loss V_θ = u_θ − h·sg[du/dt] against a network-independent
    velocity target. Shares interpolant/conversion/lift semantics with RectifiedFlow;
    the trainer owns (t, h) sampling, branching and CFG conditioning."""

    def u_and_dudt(
        self,
        model,
        x_t: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        tangent: torch.Tensor,
        cond: torch.Tensor | None,
        length: int,
        omega: torch.Tensor | None = None,
        t_lo: torch.Tensor | None = None,
        t_hi: torch.Tensor | None = None,
        mode: str = "dde",
        dde_eps: float = 5e-3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(u, du/dt, x̂) at (x_t, t, h). du/dt is the trajectory total derivative with
        tangent (dz, dt, dh) = (tangent, 1, −1); it is detached in both modes. ω/t_lo/t_hi are
        held fixed across the perturbation. 'jvp' is the seam for the flash-attention-JVP kernel."""
        if mode not in {"dde", "jvp"}:
            raise ValueError("flow.mf.dudt must be 'dde' or 'jvp'")

        def u_fn(z, t_, h_):
            x_pred = model(z, t=t_, h=h_, cond=cond, omega=omega, t_lo=t_lo, t_hi=t_hi, length=length)
            return self.target_to_v(x_pred, z, t_), x_pred

        if mode == "jvp":
            (u, x_pred), (dudt, _) = torch.func.jvp(
                u_fn, (x_t, t, h), (tangent, torch.ones_like(t), -torch.ones_like(t))
            )
            return u, dudt.detach(), x_pred

        u, x_pred = u_fn(x_t, t, h)
        eps_t = torch.full_like(t, dde_eps)
        eps_x = self._time_like(eps_t, x_t)
        with torch.no_grad():
            u_plus, _ = u_fn(x_t + eps_x * tangent, t + eps_t, h - eps_t)
            u_minus, _ = u_fn(x_t - eps_x * tangent, t - eps_t, h + eps_t)
            dudt = (u_plus.float() - u_minus.float()) / (2.0 * dde_eps)
        return u, dudt.to(u.dtype), x_pred

    @staticmethod
    def mf_loss(
        V: torch.Tensor, v_tgt: torch.Tensor, p: float = 1.0, c: float = 1e-3
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Adaptive-weighted MSE (MeanFlow): per-sample 1/(sg‖Δ‖²+c)^p; p=0 is plain MSE."""
        delta = V.float() - v_tgt.float()
        per_sample = delta.pow(2).mean(dim=tuple(range(1, delta.ndim)))
        weight = (per_sample.detach() + c).pow(-p)
        loss = (weight * per_sample).mean()
        return loss, {"mf_mse": per_sample.mean().detach()}

    @torch.no_grad()
    def sample(self, model, shape, cond=None, noise=None, steps=1, method="euler",
               guidance_scale=1.0, guidance_t_lo=None, guidance_t_hi=None, lift_scale=1.0):
        return self.generate(model, shape, cond, noise, steps, method,
                             guidance_scale, guidance_t_lo, guidance_t_hi, lift_scale)

    def generate(
        self,
        model,
        shape: tuple[int, ...],
        cond: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        steps: int = 1,
        method: str = "euler",
        guidance_scale: float = 1.0,
        guidance_t_lo: float | None = None,
        guidance_t_hi: float | None = None,
        lift_scale: float = 1.0,
    ) -> torch.Tensor:
        """Euler jumps on the average velocity. CFG is conditioned: `guidance_scale` is the inference
        ω fed to the network (single forward, no uncond pass; ω=1 ⇒ unconditional), and `guidance_t_lo/hi`
        the inference interval bounds (None on a fixed-interval model). `method` is accepted for protocol
        compat and ignored (jumps are first-order in u)."""
        if steps < 1:
            raise ValueError("steps must be >= 1")
        device = next(model.parameters()).device
        x = torch.randn(shape, device=device) if noise is None else noise.to(device=device)
        if tuple(x.shape) != tuple(shape):
            raise ValueError(f"noise shape {tuple(x.shape)} must match sample shape {tuple(shape)}")
        if cond is not None:
            cond = cond.to(device=device, dtype=x.dtype)
        batch, length = shape[0], shape[-1]

        def _col(v):  # broadcast an inference scalar to [B], or None to skip the conditioning input
            return None if v is None else torch.full((batch,), float(v), device=device, dtype=x.dtype)

        omega = None if float(guidance_scale) == 1.0 else _col(guidance_scale)
        t_lo, t_hi = _col(guidance_t_lo), _col(guidance_t_hi)
        grid = torch.linspace(EPS, 1.0 - EPS, steps + 1, device=device, dtype=x.dtype)
        for i in range(steps):
            t = grid[i].expand(batch)
            h = (grid[i + 1] - grid[i]).expand(batch)
            x_pred = model(x, t=t, h=h, cond=cond, omega=omega, t_lo=t_lo, t_hi=t_hi, length=length)
            x = x + (grid[i + 1] - grid[i]) * self.target_to_v(x_pred, x, t)
        return x / lift_scale
```

Register in `src/flow/factory.py`:

```python
from flow.mf import MeanFlow

_METHODS: dict[str, type] = {
    "rectified_flow": RectifiedFlow,
    "mean_flow": MeanFlow,
}
```

- [ ] **Step 4:** `uv run pytest tests/test_mean_flow.py tests/test_flow_matching.py -v` — all pass.
      If the jvp tests fail with a forward-AD "not implemented" error from the toy models'
      ops, fix the toy model, not the code; if `torch.func.jvp` itself fails on stft later,
      that is the documented jvp-mode limitation, not a blocker.
- [ ] **Step 5:** commit `mf: MeanFlow method + guided (ω-conditioned) velocity target`.

### Task 3 — flexible CFG (ω-conditioning) in `RFTrainer`

**Files:** modify `src/trainer.py` (`RFTrainer` only).

- [ ] **Step 1: refactor lift.** In `RFTrainer`, extract the existing rms-lift block out of
`training_step` verbatim into:

```python
def _lift(self, audio: torch.Tensor) -> torch.Tensor:
    if not self.rms_lift:
        return audio
    rms = audio.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-8)
    return self.lift_scale * torch.tanh((self.rms_target / rms) * audio)
```

(keep the existing WavFlow comment with it) and call `audio = self._lift(audio)` at the top of
`training_step`.

- [ ] **Step 2: guidance inputs helper.** Add to `RFTrainer` (`import math` already at the top of
`src/trainer.py`):

```python
def _guidance_inputs(self, audio: torch.Tensor):
    """Per-batch CFG conditioning. Returns (omega, lo, hi, model_lo, model_hi):
    - guidance off → all None.
    - fixed interval → (omega, cfg_lo, cfg_hi, None, None): network sees ω only.
    - conditioned interval → (omega, t_lo, t_hi, t_lo, t_hi): both bounds sampled and fed in."""
    g = self.cfg.flow.get("guidance", None)
    if not (g and g.get("enabled", False)):
        return None, None, None, None, None
    batch, device, dtype = audio.shape[0], audio.device, audio.dtype
    omega = (torch.rand(batch, device=device, dtype=dtype) * math.log(float(g.get("omega_max", 8.0)))).exp()
    if g.get("condition_interval", False):
        t_lo = torch.rand(batch, device=device, dtype=dtype) * 0.5          # U(0, 0.5)
        t_hi = 0.5 + torch.rand(batch, device=device, dtype=dtype) * 0.5    # U(0.5, 1)
        return omega, t_lo, t_hi, t_lo, t_hi
    lo, hi = g.get("interval", [0.0, 1.0])
    return omega, float(lo), float(hi), None, None
```

- [ ] **Step 3: guided target.** Factor the RF body into
`_rf_step(audio, cond, adaptive=False)` and have `training_step` call it with `adaptive=False`.
The prediction is ω-conditioned and the target is the guided velocity when guidance is on:

```python
def training_step(self, audio: torch.Tensor, cond: torch.Tensor | None):
    return self._rf_step(audio, cond, adaptive=False)

def _rf_step(self, audio: torch.Tensor, cond: torch.Tensor | None, adaptive: bool):
    loss_cfg = self.cfg.loss
    audio = self._lift(audio)
    x_t, t, x1 = self.method.train_tuple(audio, t=self._sample_t(audio))
    length = audio.shape[-1]
    omega, lo, hi, model_lo, model_hi = self._guidance_inputs(audio)
    with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp_enabled):
        pred = self.model(x_t, t=t, cond=cond, omega=omega, t_lo=model_lo, t_hi=model_hi, length=length)
        if omega is not None and cond is not None:
            v_g = self.method.guided_velocity_target(
                self.model, x1, x_t, t, cond, omega, length, lo, hi, model_lo, model_hi
            )
            target = self.method.v_to_target(v_g, x_t, t)
        else:
            target = x1
        if adaptive:
            total, terms = self.method.mf_loss(
                self.method.target_to_v(pred, x_t, t),
                self.method.target_to_v(target, x_t, t),
                p=self.adaptive_p, c=self.adaptive_c,
            )
            terms = {"rf_mse": terms["mf_mse"]}
        else:
            total, terms = self.method.loss(
                pred, target, x_t, t,
                space=str(loss_cfg.get("loss_space", "v")),
                loss_type=str(loss_cfg.get("primary", "mse")),
            )
        mr_stft_weight = float(loss_cfg.get("mr_stft_weight", 0.0))
        if mr_stft_weight > 0.0:
            aux = mr_stft_loss(pred, x1, log_weight=float(loss_cfg.get("mr_stft_log_weight", 0.0)))
            total = total + mr_stft_weight * aux
            terms = {**terms, "mr_stft": aux}
    return total, terms
```

(MR-STFT aux stays against the real `x1` — a spectral anchor to data, not the guided field.)
Add `_mr_stft_aux(pred, x1, t)`: when `loss.mr_stft_t_min` is set, subset to rows with
`t >= mr_stft_t_min`; when absent, preserve the existing all-row behavior. `MFTrainer` calls
`_rf_step(..., adaptive=True)`, which converts prediction/target to velocity and applies
`MeanFlow.mf_loss` so the RF anchor and MF batches use the same adaptive weighting.

- [ ] **Step 4:** `uv run pytest tests/test_trainer.py tests/test_fm_scaffold.py -v` — all pass
      (guidance defaults to null; `omega=None` path is byte-for-byte the old behaviour).
- [ ] **Step 5:** commit `rf: optional flexible-CFG (ω-conditioned) training target`.

### Task 4 — `MFTrainer`

**Files:** modify `src/trainer.py` (add below `RFTrainer`, above `FDTrainer`).

- [ ] **Step 1: implement** (mirror `FDTrainer`'s init_from pattern; reuse everything else):

```python
class MFTrainer(RFTrainer):
    """MeanFlow training (iMF v-loss on the x-pred backbone). Micro-batches branch between
    plain RF (h=0, fraction flow.mf.rf_fraction, incl. an initial pure-RF warmup) and the
    MF objective. Warm-startable from an RF checkpoint via train.init_from (the backbone's
    zero-init aux embeddings make the load an exact RF warm start)."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        mf_cfg = _as_dict(self.cfg.flow.get("mf", {}))
        self.rf_fraction = float(mf_cfg.get("rf_fraction", 0.5))
        self.rf_warmup_steps = int(mf_cfg.get("rf_warmup_steps", 0))
        self.dudt_mode = str(mf_cfg.get("dudt", "dde"))
        self.dde_eps = float(mf_cfg.get("dde_eps", 5e-3))
        self.adaptive_p = float(mf_cfg.get("adaptive_p", 1.0))
        self.adaptive_c = float(mf_cfg.get("adaptive_c", 1e-3))

        init_from = self.cfg.train.get("init_from", None)
        if init_from and self.step == 0:
            state = torch.load(Path(init_from).expanduser(), map_location="cpu", weights_only=False)
            missing, unexpected = self.model.load_state_dict(state["model"], strict=False)
            aux = ("gap_embed", "omega_embed", "lo_embed", "hi_embed")
            if unexpected or any(not any(tag in key for tag in aux) for key in missing):
                raise ValueError(f"init_from mismatch: missing={missing} unexpected={unexpected}")
            if self.ema is not None:
                self.ema = EMA(self.model, decay=float(self.cfg.train.ema_decay))
            tqdm.write(f"Warm-started from {init_from}")

    def _sample_t_h(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(t, h): two draws from the RF t-distribution; t = min, destination = max.
        h is floored at dde_eps so the DDE probe at h−ε stays in-range."""
        a, b = self._sample_t(audio), self._sample_t(audio)
        t = torch.minimum(a, b)
        h = (torch.maximum(a, b) - t).clamp_min(self.dde_eps)
        return t, h

    def training_step(self, audio: torch.Tensor, cond: torch.Tensor | None):
        # Whole micro-batches branch (not per-row): keeps the MF extra forwards off the
        # RF batches entirely; grad accumulation averages the mix across the step.
        if self.step < self.rf_warmup_steps or torch.rand(()).item() < self.rf_fraction:
            return self._rf_step(audio, cond, adaptive=True)
        return self._mf_step(audio, cond)

    def _mf_step(self, audio: torch.Tensor, cond: torch.Tensor | None):
        loss_cfg = self.cfg.loss
        audio = self._lift(audio)
        t, h = self._sample_t_h(audio)
        x_t, t, x1 = self.method.train_tuple(audio, t=t)
        length = audio.shape[-1]
        omega, lo, hi, model_lo, model_hi = self._guidance_inputs(audio)
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp_enabled):
            with torch.no_grad():
                if omega is not None and cond is not None:
                    v_tgt, tangent = self.method.guided_velocity_target(
                        self.model, x1, x_t, t, cond, omega, length, lo, hi, model_lo, model_hi,
                        return_boundary=True,
                    )
                else:
                    v_tgt = self.method.target_to_v(x1, x_t, t)
                    # iMF marginal tangent: the model's own boundary (h=0) velocity, not v_cond
                    x_b = self.model(x_t, t=t, cond=cond, omega=omega, t_lo=model_lo, t_hi=model_hi, length=length)
                    tangent = self.method.target_to_v(x_b, x_t, t)
            u, dudt, x_pred = self.method.u_and_dudt(
                self.model, x_t, t, h, tangent, cond, length,
                omega=omega, t_lo=model_lo, t_hi=model_hi, mode=self.dudt_mode, dde_eps=self.dde_eps,
            )
            V = u - self.method._time_like(h, u) * dudt
            total, terms = self.method.mf_loss(V, v_tgt, p=self.adaptive_p, c=self.adaptive_c)
            mr_stft_weight = float(loss_cfg.get("mr_stft_weight", 0.0))
            keep = t >= float(loss_cfg.get("mr_stft_t_min", 0.2))
            if mr_stft_weight > 0.0 and keep.any():
                aux = mr_stft_loss(
                    x_pred[keep], x1[keep],
                    log_weight=float(loss_cfg.get("mr_stft_log_weight", 0.0)),
                )
                total = total + mr_stft_weight * aux
                terms = {**terms, "mr_stft": aux}
        return total, terms

    def sample(self, shape, cond=None, noise=None) -> torch.Tensor:
        s = self.cfg.sampling
        audio = self.method.sample(
            self.model, shape, cond=cond, noise=noise,
            steps=int(s.get("steps", 1)), method=str(s.get("method", "euler")),
            guidance_scale=float(s.get("guidance_scale", 1.0)),
            guidance_t_lo=s.get("guidance_t_lo", None), guidance_t_hi=s.get("guidance_t_hi", None),
            lift_scale=self.lift_scale if self.rms_lift else 1.0,
        )
        peak = audio.abs().amax(dim=tuple(range(1, audio.ndim)), keepdim=True).clamp_min(1e-8)
        return audio / peak
```

Notes for the implementer:
- `validation_step` is inherited: val loss stochastically mixes RF/MF terms. Acceptable —
  the load-bearing validation is the embedding metrics and 1-NFE samples.
- `FDTrainer` composes with MF checkpoints unchanged: it rebuilds the method from the
  checkpoint cfg via the factory and calls `method.generate` (its call omits `guidance_t_lo/hi`,
  which default to None → the conditioned model samples at its `t_lo=0, t_hi=1` full-interval
  defaults; pass them through later if FD-over-conditioned-MF needs a specific interval).

- [ ] **Step 2:** `uv run pytest tests/ -v` — full suite passes.
- [ ] **Step 3:** commit `mf: MFTrainer (RF/MF branching, warmup, init_from, dde/jvp)`.

### Task 5 — configs

**Files:** modify `configs/flow/fm.yaml`; create `configs/flow/mf.yaml`,
`configs/experiment/mf_oneshots_mars_stft.yaml`, `configs/experiment/mf_smoke.yaml`.

- [ ] **Step 1:** append to `configs/flow/fm.yaml`:

```yaml
# Flexible CFG as a conditioning input (RectifiedFlow.guided_velocity_target); null = off (plain RF).
# To enable on a plain RF run: set backbone.conditioning.guidance_embed: true (and interval_embed: true
# if condition_interval) and provide a guidance block like configs/flow/mf.yaml's.
guidance: null
```

- [ ] **Step 2:** create `configs/flow/mf.yaml`:

```yaml
method: mean_flow
t_encoding:
  type: sinusoidal
  scale: 100.0
eps: 1e-5
# (t, h) are two draws from this distribution: t = min, h = max − min (clamped ≥ mf.dde_eps).
t_distribution: logit_normal
logit_mean: 0.4  # mirrors iMF's -0.4 into the repo axis (0=noise, 1=data)
logit_std: 1.0
mf:
  rf_fraction: 0.5      # fraction of micro-batches trained as plain RF (h=0); MeanAudio used 0.75 pre-iMF
  rf_warmup_steps: 0    # pure-RF steps before mixing; 0 when train.init_from is an RF checkpoint
  dudt: dde             # 'dde' (two no-grad forwards, FlashAttention-safe, default) | 'jvp' (triton-kernel seam)
  dde_eps: 0.005
  adaptive_p: 1.0       # adaptive loss weight 1/(sg|Δ|²+c)^p; 0 = plain MSE
  adaptive_c: 1.0e-3
# Flexible CFG. ω is ALWAYS a conditioning input (sampled log-uniform[1, omega_max] in training,
# set via sampling.guidance_scale at inference; ω=1 = unconditional). The interval has two modes:
#   condition_interval: false → fixed [lo,hi] mask, network sees ω only (+0 scalars). One interval baked in.
#   condition_interval: true  → sample t_lo~U(0,0.5), t_hi~U(0.5,1) per row and condition on BOTH bounds
#                               (needs backbone.conditioning.interval_embed: true; tunable at inference via
#                               sampling.guidance_t_lo / guidance_t_hi). +2 AdaLN scalars — see Task 1 note.
# interval is repo t-axis (0=noise, 1=data): lower hi = guidance off nearer data, raise lo = off nearer noise.
guidance:
  enabled: true
  omega_max: 8.0
  condition_interval: false
  interval: [0.0, 0.8]   # measured RF sweep: guidance off near data; ignored in conditioned mode
```

- [ ] **Step 3:** create `configs/experiment/mf_oneshots_mars_stft.yaml` (mirrors
`fm_oneshots_mars_stft.yaml`; differences flagged):

```yaml
# @package _global_
#
# MeanFlow fine-tune of the pretrained STFT-transformer RF model (iMF v-loss, x-pred backbone,
# ω-conditioned CFG, dde du/dt). Warm start: set train.init_from to an RFTrainer checkpoint and
# keep flow.mf.rf_warmup_steps: 0. From scratch instead: drop init_from, set rf_warmup_steps ~50000.
# To make the guidance interval tunable too: flow.guidance.condition_interval: true AND
# backbone.conditioning.interval_embed: true.

defaults:
  - /train/default@train
  - /data/audio_directory@data
  - /flow/mf@flow
  - /backbone/stft_transformer@backbone
  - /eval/preliminary@eval
  - _self_

trainer:
  _target_: trainer.MFTrainer

backbone:
  conditioning:
    time_scale: ${flow.t_encoding.scale}
    gap_embed: true       # MeanFlow gap h
    guidance_embed: true  # ω conditioning
    interval_embed: false # set true together with flow.guidance.condition_interval: true

conditioner:
  type: matpac
  checkpoint_path: /media/NAS/neutone/diff_one_shot/checkpoints/whole-violet-235/last-v1_clean.ckpt
  device: ${train.device}
  use_teacher: false
  encode_batch_size: 0
  compile_encoder: false

loss:
  loss_space: v
  primary: mse
  mr_stft_weight: 1.0
  mr_stft_log_weight: 0.0
  mr_stft_t_min: 0.2  # pMF t<=0.8 mirrored to repo axis: apply only for t>=0.2

optimizer:
  _target_: torch.optim.AdamW
  lr: 1e-4  # fine-tune LR (half the RF pretrain LR)
  weight_decay: 0.01
  betas: [0.9, 0.95]  # β₂=0.95 already matches the TVM stability recommendation

data:
  root: /media/storage/samples/samples_from_mars/one_shots

train:
  run_dir: runs/mf-oneshots-mars-stft
  device: cuda
  init_from: runs/fm-oneshots-mars-stft-b64-200k/checkpoints/step_00200000.pt  # ← adjust to the actual RF ckpt
  max_steps: 150000
  amp: true
  ema_decay: 0.999
  grad_clip: 1.0
  grad_accum_steps: 4
  warmup_steps: 2000
  min_lr_ratio: 0.0
  resume: null
  val_fraction: 0.02
  log_every: 10
  val_every: 1000
  cond_dropout_prob: 0.1  # keep: the ∅ branch of the guided target trains through dropout
  sample_every: 2000
  ckpt_every: 25000
  val_batches: 24
  dataloader:
    batch_size: 64
    num_workers: 8
    pin_memory: true
    drop_last: true
  wandb:
    audio_examples: 10

eval:
  metrics:
    embedding_validation:
      enabled: true
      backend: clap
      distance: mind
      cosine: true
      mind_projections: 256
      cache_real: true
      density_coverage:
        enabled: true
        k: 5

sampling:
  batch_size: 1
  steps: 1               # 1-NFE; quality A/B at steps: 2
  method: euler
  guidance_scale: 1.0    # inference ω (1 = unconditional); raise to sample stronger guidance
  guidance_t_lo: null    # set (with condition_interval/interval_embed) to sweep the interval at inference
  guidance_t_hi: null
```

- [ ] **Step 4:** create `configs/experiment/mf_smoke.yaml` (CPU; uses the same `data/fm-smoke`
fixture as `fm_wavenext_smoke`; guidance left at the flow-group default = fixed-interval, exercising
the ω-conditioning path with a no-op interval):

```yaml
# @package _global_

defaults:
  - /train/default@train
  - /data/audio_directory@data
  - /flow/mf@flow
  - /backbone/stft_transformer@backbone
  - /eval/preliminary@eval
  - _self_

trainer:
  _target_: trainer.MFTrainer

conditioner:
  type: "null"
  embedding_dim: 16

backbone:
  sample_rate: 8000
  stft:
    n_fft: 64
    hop_length: 16
    win_length: 64
  block:
    dim: 32
    depth: 1
    heads: 2
  conditioning:
    cond_dim: 16
    embed_dim: 16
    gap_embed: true
    guidance_embed: true
    interval_embed: false
    time_scale: ${flow.t_encoding.scale}

flow:
  mf:
    rf_warmup_steps: 1  # exercises warmup → mixed branching within the smoke run

loss:
  loss_space: v
  primary: mse
  mr_stft_weight: 0.0

optimizer:
  _target_: torch.optim.AdamW
  lr: 1e-3
  weight_decay: 0.01
  betas: [0.9, 0.95]

data:
  root: data/fm-smoke
  sample_rate: 8000
  min_seconds: 0.032
  max_seconds: 0.032
  channels: 1
  augmentations:
    enabled: false

train:
  run_dir: runs/mf-smoke
  device: cpu
  max_steps: 4
  amp: false
  ema_decay: 0.9
  warmup_steps: 0
  val_fraction: 0.5
  log_every: 1
  val_every: 2
  sample_every: 2
  ckpt_every: 2
  dataloader:
    batch_size: 2
    num_workers: 0
    pin_memory: false
    drop_last: false

sampling:
  batch_size: 1
  steps: 1
  guidance_scale: 1.0
```

- [ ] **Step 5: verify smoke end-to-end:**

```bash
uv run python train.py --config-name experiment/mf_smoke
uv run python train.py --config-name experiment/mf_smoke flow.mf.dudt=jvp train.run_dir=runs/mf-smoke-jvp
# conditioned-interval path (needs interval_embed):
uv run python train.py --config-name experiment/mf_smoke \
  flow.guidance.condition_interval=true backbone.conditioning.interval_embed=true \
  train.run_dir=runs/mf-smoke-cond
```

Expected: all three runs finish 4 steps with finite losses, a checkpoint, and finite WAVs under
the run dir. If the jvp run fails inside `torch.func.jvp` with a forward-AD NotImplementedError on
`stft`/`istft`, record the error verbatim in `docs/fm_training_notes.md` under Live issues
(jvp mode pending the triton kernel) and proceed — dde is the default.

- [ ] **Step 6:** `uv run ruff check . && uv run pytest` — clean.
- [ ] **Step 7:** commit `mf: flow/experiment/smoke configs`.

---

## 4. Acceptance checklist

- [ ] `uv run pytest` and `uv run ruff check .` pass.
- [ ] `mf_smoke` runs on CPU in `dde` mode, in `jvp` mode (or jvp failure documented as above),
      and in conditioned-interval mode.
- [ ] An RF checkpoint loads via `train.init_from` with only aux-embed keys
      (`gap_embed`/`omega_embed`/`lo_embed`/`hi_embed`) missing, and a 1-step sanity run
      (`train.max_steps=1 flow.mf.rf_fraction=1.0 flow.guidance=null train.wandb.enabled=false`)
      reproduces an ordinary RF loss magnitude (warm start intact).
- [ ] Flexible CFG enabled on a plain RF config trains with finite loss — the "RF and MF matching"
      requirement. Override the whole node (per-key `+` onto a null node fails in Hydra):
      `'flow.guidance={enabled:true,omega_max:8.0,condition_interval:false,interval:[0.0,1.0]}' backbone.conditioning.guidance_embed=true`
- [ ] No changes to `convnext.py`, `FDTrainer` behaviour, or existing RF defaults
      (`flow/fm.yaml` gains only `guidance: null`).

## 5. Known follow-ups (not in this spec)

- Port the flash-attention-JVP triton kernel behind `flow.mf.dudt: jvp`; then A/B jvp vs dde
  for ~10k steps (loss curves + 1-NFE samples) before any long run.
- Tune inference ω (`sampling.guidance_scale`) for 1-NFE and 2-NFE before any FD-loss stage;
  if `condition_interval: true`, sweep `guidance_t_lo/guidance_t_hi` too.
- Sweep `mf.rf_fraction` (0.75 / 0.5 / 0.25).
- Ablate the mirrored iMF default `logit_mean: 0.4` against the RF default `0.0`; the former
  biases toward the repo's data end because the repo time axis is reversed from iMF.
- If conditioned-interval AdaLN overload shows up (worse conditioning fidelity than fixed at
  matched steps), evaluate in-context conditioning (the iMF AdaLN→token swap).
- Sweep the pMF-style MR-STFT threshold around the mirrored default `t >= 0.2` (pMF uses
  `t <= 0.8` on its data-to-noise axis; its longer-training `0.6` setting maps to repo `t >= 0.4`).

---

## 6. Revision (2026-06-19) — pMF twin x-pred heads (fixes the single-head collapse)

The single-head recipe (Tasks 1–5) collapsed in training: 1-NFE output went to noise even on
silence, every eval metric 6–18× worse than the RF base. Root cause (`docs/fm_training_notes.md`,
2026-06-17): the `du/dt` tangent came from the **u-head re-evaluated at the h=0 boundary** — an
unsupervised, high-variance eval — driving the MeanFlow `‖J‖²`/`‖h·du/dt‖` blow-up. pMF
(`/tmp/pMF`) fixes this with a directly-supervised instantaneous-velocity head that supplies the
tangent. This revision implements that and supersedes the Task 4 `_mf_step` tangent and the "no
v-head" scope line. **Both heads stay x-prediction** (converted to velocity internally — mandatory
for raw data); the v-head is training-only.

**Backbone (`src/backbone/transformer.py`).** New `block.aux_depth` (default 0). `blocks` stays the
full `depth` ModuleList — `[:depth−aux_depth]` is a shared trunk, the tail is the u-head — so
`out_proj`/`blocks` keep their names and RF checkpoints + existing tests load unchanged. A parallel
`v_blocks` (`aux_depth` blocks) + `v_out_proj` branch off the shared-trunk activation.
`forward(return_aux=True)` returns `(u_spec, v_spec)`; otherwise (and whenever `aux_depth=0`) it
returns the single u-head tensor, so inference/eval never run the v-head. `aux_depth=0` is the RF
model bit-for-bit; MF runs use `aux_depth=6` of `depth=12`.

**Flow (`src/flow/fm.py`).** `target_to_v` floors `|1−t|` at module constant `VELOCITY_CLIP=0.05`
(sign-preserving), matching pMF's `clip(t,0.05,1)` (was `EPS=1e-5`). `_predict` gains `return_aux`,
ISTFT-ing the v-branch alongside the u-branch (composes with `return_spec`). `guided_velocity_target`
drops `return_boundary` (the trainer no longer takes a tangent from it). `mf.py` is unchanged.

**Trainer (`src/trainer.py::MFTrainer`).** `_mf_step` now: compute `v_tgt` (guided or plain) under
no-grad; forward the v-head (`_predict(return_aux=True)`) for `v_c = target_to_v(x̂_v)`; pass
`v_c.detach()` as the `u_and_dudt` tangent (replacing the h=0 boundary eval); loss
`= loss_u + loss_v`, `loss_v = mf_loss(v_c, v_tgt)`, logged as `train/v_mse`. `_sample_t_h` applies
a **per-row r=t** mask — `rf_fraction` of rows get `h=0` (pure FM for both heads) — replacing the
whole-microbatch RF/MF split; `training_step` is just `_mf_step`. `rf_warmup_steps` removed.
`MFTrainer` asserts `aux_depth>0`; the warm-start allow-list gains `v_blocks`/`v_out_proj`.

**Configs.** `backbone/stft_transformer.yaml` adds `block.aux_depth: 0`; the MF experiment configs
set `aux_depth: 6`. `flow/mf.yaml` sampler → `logit_mean: -0.8, logit_std: 0.8` (pMF `(0.8,0.8)`
mirrored to the repo axis) and drops `rf_warmup_steps`.

**Tests.** `tests/test_backbone.py`: `aux_depth=0` RF-equivalence, twin-head shapes under
`return_aux`, RF→v-head warm-start. `tests/test_mean_flow.py`: velocity-clip behaviour
(replaces the old past-endpoint sign test); the `return_boundary` test is removed.
