"""GPU parity tests for the GLM-5.2 TileLang sparse-MLA kernels.

The full AMD profile uses TP8, leaving eight local attention heads (H8), while
the five-layer MI355X proof of concept used TP4/H16. Both shapes are tested
against the same FP32 PyTorch reference. The invalid slots also make an unsafe
``-1`` gather deterministic by placing infinities immediately before ``kv``.
"""

from dataclasses import dataclass

import pytest
import torch

try:
    import tilelang  # noqa: F401
except ImportError:
    tilelang = None

if tilelang is not None:
    from miles_plugins.models.glm5.ops.sparse_mla import SparseMLA
else:
    SparseMLA = None


@dataclass(frozen=True)
class _Diff:
    relative: float
    maximum: float


def _diff(reference: torch.Tensor, actual: torch.Tensor) -> _Diff:
    reference = reference.float().flatten()
    actual = actual.float().flatten()
    denominator = reference.square().sum() + actual.square().sum()
    relative = 0.0 if denominator == 0 else (1 - 2 * (reference * actual).sum() / denominator).item()
    return _Diff(relative=relative, maximum=(reference - actual).abs().max().item())


def _reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    grad_output: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_ref = q.detach().float().requires_grad_(True)
    kv_ref = kv.detach().float().requires_grad_(True)
    indices_2d = indices.squeeze(1)
    valid = indices_2d != -1
    selected = kv_ref[indices_2d.clamp_min(0).long(), 0]

    scores = torch.einsum("shd,skd->shk", q_ref, selected) * scale
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    has_valid_key = valid.any(dim=-1)
    safe_scores = torch.where(has_valid_key[:, None, None], scores, torch.zeros_like(scores))
    probabilities = torch.softmax(safe_scores, dim=-1)
    probabilities = torch.where(valid.unsqueeze(1), probabilities, 0)
    output = torch.einsum("shk,skd->shd", probabilities, selected[..., :512])
    lse_base_2 = torch.logsumexp(scores[has_valid_key], dim=-1) / torch.log(
        torch.tensor(2.0, device=q.device)
    )
    (output * grad_output.float()).sum().backward()

    assert q_ref.grad is not None
    assert kv_ref.grad is not None
    return output, lse_base_2, q_ref.grad, kv_ref.grad


def _inputs(heads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(20260831 + heads)
    sequence_length, kv_length, topk = 16, 80, 64
    q = torch.randn(sequence_length, heads, 576, device="cuda", dtype=torch.bfloat16)

    kv_storage = torch.empty(kv_length + 1, 1, 576, device="cuda", dtype=torch.bfloat16)
    kv_storage[0].fill_(float("inf"))
    kv_storage[1:].normal_()
    kv = kv_storage[1:]
    assert kv.is_contiguous()

    indices = torch.stack(
        [torch.randperm(kv_length, device="cuda")[:topk] for _ in range(sequence_length)]
    ).to(torch.int32)
    indices[:, -11:] = -1
    indices[-1] = -1
    indices = indices.unsqueeze(1).contiguous()
    grad_output = torch.randn(sequence_length, heads, 512, device="cuda", dtype=torch.bfloat16)
    return q, kv, indices, grad_output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
@pytest.mark.skipif(tilelang is None, reason="TileLang not installed")
@pytest.mark.parametrize("heads", [8, 16], ids=["full-tp8-h8", "poc-tp4-h16"])
def test_forward_backward_matches_reference_with_padded_indices(heads: int) -> None:
    """H8 and H16 must meet the same declared BF16 forward/backward tolerances."""
    q, kv, indices, grad_output = _inputs(heads)
    scale = 256**-0.5  # GLM-5.2 scales the pre-absorption 192+64 QK head.
    ref_output, ref_lse, ref_dq, ref_dkv = _reference(q, kv, indices, grad_output, scale)

    q_actual = q.detach().requires_grad_(True)
    kv_actual = kv.detach().requires_grad_(True)
    output, lse = SparseMLA.apply(q_actual, kv_actual, indices, scale)
    output.backward(grad_output)

    assert q_actual.grad is not None
    assert kv_actual.grad is not None
    for name, tensor in {
        "output": output,
        "lse": lse,
        "dQ": q_actual.grad,
        "dKV": kv_actual.grad,
    }.items():
        assert torch.isfinite(tensor).all(), f"{name} contains NaN or Inf for H{heads}"

    output_diff = _diff(ref_output, output)
    dq_diff = _diff(ref_dq, q_actual.grad)
    dkv_diff = _diff(ref_dkv, kv_actual.grad)
    assert output_diff.relative < 1e-3 and output_diff.maximum < 0.1, output_diff
    torch.testing.assert_close(lse[:-1].float(), ref_lse, rtol=2e-2, atol=5e-2)
    torch.testing.assert_close(output[-1].float(), torch.zeros_like(output[-1].float()))
    assert dq_diff.relative < 5e-2, dq_diff
    assert dkv_diff.relative < 5e-2, dkv_diff
