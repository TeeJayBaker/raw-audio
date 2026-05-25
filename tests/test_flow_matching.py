from __future__ import annotations

import pytest
import torch

from flow.fm import RectifiedFlow


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.seen_cond = []

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        length: int | None = None,
    ) -> torch.Tensor:
        del t, length
        self.seen_cond.append(cond)
        if cond is None:
            return x
        return x + cond[:, :, None]


def test_train_tuple_returns_x_t_t_and_target_only():
    flow = RectifiedFlow()
    x1 = torch.randn(2, 1, 8)
    noise = torch.randn_like(x1)
    t = torch.tensor([0.25, 0.75])

    x_t, sampled_t, target = flow.train_tuple(x1, noise=noise, t=t)

    assert x_t.shape == x1.shape
    assert torch.allclose(sampled_t, t)
    assert target is x1
    assert torch.allclose(x_t, (1.0 - t.view(2, 1, 1)) * noise + t.view(2, 1, 1) * x1)


def test_target_to_v_recovers_rectified_flow_velocity():
    flow = RectifiedFlow()
    x1 = torch.randn(2, 1, 8)
    noise = torch.randn_like(x1)
    t = torch.tensor([0.25, 0.75])
    x_t, sampled_t, target = flow.train_tuple(x1, noise=noise, t=t)

    v = flow.target_to_v(target, x_t, sampled_t)

    assert torch.allclose(v, x1 - noise, atol=1e-6)


@pytest.mark.parametrize("space", ["x", "v"])
@pytest.mark.parametrize("loss_type", ["mse", "l1"])
def test_loss_backpropagates(space: str, loss_type: str):
    flow = RectifiedFlow()
    x1 = torch.randn(2, 1, 8)
    x_t, t, target = flow.train_tuple(x1, t=torch.rand(2))
    pred = torch.randn_like(x1, requires_grad=True)

    loss, terms = flow.loss(pred, target, x_t, t, space=space, loss_type=loss_type)
    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(loss)
    assert set(terms) == {f"{space}_{loss_type}"}


def test_sampler_uses_model_device_and_fixed_noise():
    flow = RectifiedFlow()
    model = ToyModel()
    noise = torch.randn(2, 1, 8)

    sample_a = flow.sample(model, shape=(2, 1, 8), noise=noise, steps=2)
    sample_b = flow.sample(model, shape=(2, 1, 8), noise=noise, steps=2)

    assert torch.allclose(sample_a, noise)
    assert torch.allclose(sample_b, noise)
    assert not torch.allclose(noise[0], noise[1])


def test_sampler_unlifts_output_by_lift_scale():
    flow = RectifiedFlow()
    model = ToyModel()
    noise = torch.randn(2, 1, 8)

    baseline = flow.sample(model, shape=(2, 1, 8), noise=noise, steps=2)
    lifted = flow.sample(model, shape=(2, 1, 8), noise=noise, steps=2, lift_scale=2.0)

    assert torch.allclose(lifted, baseline / 2.0)


def test_sampler_rejects_wrong_noise_shape():
    flow = RectifiedFlow()
    model = ToyModel()

    with pytest.raises(ValueError, match="noise shape"):
        flow.sample(model, shape=(2, 1, 8), noise=torch.randn(1, 1, 8))


@pytest.mark.parametrize("method", ["euler", "heun"])
def test_sampler_methods_return_expected_shape(method: str):
    flow = RectifiedFlow()
    model = ToyModel()

    sample = flow.sample(model, shape=(2, 1, 8), steps=2, method=method)

    assert sample.shape == (2, 1, 8)


def test_sampler_cfg_uses_none_for_unconditional_prediction():
    flow = RectifiedFlow()
    model = ToyModel()
    cond = torch.ones(2, 1)

    sample = flow.sample(model, shape=(2, 1, 8), cond=cond, steps=1, guidance_scale=2.0)

    assert sample.shape == (2, 1, 8)
    assert model.seen_cond[0] is not None
    assert model.seen_cond[1] is None
