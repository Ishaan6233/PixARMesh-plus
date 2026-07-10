import torch
import torch.nn.functional as F

from src.data.typing import TokenType
from src.models.loss import _dequantize_token_bins, _layout_auxiliary_losses, causal_lm_loss_with_token_types


POS_OFFSET = 6
NUM_POS = 32
VOCAB = POS_OFFSET + NUM_POS + 8
LAYOUT_TOKENS = 24


def _bbox_bins(offset: int = 0) -> torch.Tensor:
    base = torch.tensor(
        [
            [4, 5, 6],
            [10, 5, 6],
            [10, 11, 6],
            [4, 11, 6],
            [4, 5, 15],
            [10, 5, 15],
            [10, 11, 15],
            [4, 11, 15],
        ],
        dtype=torch.long,
    )
    return (base + offset).clamp(0, NUM_POS - 1).reshape(-1)


def _layout_logits(pred_bins: torch.Tensor, *, seq_len: int = LAYOUT_TOKENS) -> torch.Tensor:
    logits = torch.full((1, seq_len, VOCAB), -20.0)
    for i, pred_bin in enumerate(pred_bins[:seq_len].tolist()):
        logits[0, i, POS_OFFSET + pred_bin] = 20.0
    return logits


def _shift_labels(target_bins: torch.Tensor, *, seq_len: int = LAYOUT_TOKENS) -> torch.Tensor:
    labels = torch.full((1, seq_len), -100, dtype=torch.long)
    labels[0, : target_bins.numel()] = POS_OFFSET + target_bins[:seq_len]
    return labels


def _token_types(*, seq_len: int = LAYOUT_TOKENS) -> torch.Tensor:
    return torch.full((1, seq_len), TokenType.LAYOUT, dtype=torch.long)


def test_ordinal_loss_prefers_near_bins_over_far_bins():
    target = _bbox_bins()
    near_logits = _layout_logits((target + 1).clamp(max=NUM_POS - 1))
    far_logits = _layout_logits((target + 12).clamp(max=NUM_POS - 1))
    labels = _shift_labels(target)
    token_types = _token_types()

    near = _layout_auxiliary_losses(
        near_logits,
        labels,
        token_types,
        pos_token_offset=POS_OFFSET,
        num_pos_tokens=NUM_POS,
        loss_layout_ordinal_sigma=2.0,
    )["loss_layout_ordinal"]
    far = _layout_auxiliary_losses(
        far_logits,
        labels,
        token_types,
        pos_token_offset=POS_OFFSET,
        num_pos_tokens=NUM_POS,
        loss_layout_ordinal_sigma=2.0,
    )["loss_layout_ordinal"]

    assert near.item() < far.item()


def test_expected_coordinate_loss_is_differentiable_through_position_logits():
    target = _bbox_bins()
    labels = torch.full((1, LAYOUT_TOKENS + 1), -100, dtype=torch.long)
    labels[0, 1:] = POS_OFFSET + target
    token_types = torch.full((1, LAYOUT_TOKENS + 1), TokenType.LAYOUT, dtype=torch.long)
    logits = torch.zeros((1, LAYOUT_TOKENS + 1, VOCAB), requires_grad=True)

    loss, _, _, components = causal_lm_loss_with_token_types(
        logits,
        labels,
        vocab_size=VOCAB,
        token_type_ids=token_types,
        num_items_in_batch=torch.tensor(1.0),
        pos_token_offset=POS_OFFSET,
        num_pos_tokens=NUM_POS,
        loss_layout_coord_weight=1.0,
        return_layout_components=True,
    )
    loss.backward()

    assert components["loss_layout_coord"] is not None
    assert logits.grad is not None
    assert logits.grad[..., POS_OFFSET : POS_OFFSET + NUM_POS].abs().sum().item() > 0


def test_center_and_log_size_losses_match_24_token_bbox_math():
    target_bins = _bbox_bins()
    pred_bins = _bbox_bins(offset=2)
    logits = _layout_logits(pred_bins)
    labels = _shift_labels(target_bins)

    aux = _layout_auxiliary_losses(
        logits,
        labels,
        _token_types(),
        pos_token_offset=POS_OFFSET,
        num_pos_tokens=NUM_POS,
        loss_layout_ordinal_sigma=None,
    )

    pred_bbox = _dequantize_token_bins(pred_bins, NUM_POS).view(1, 8, 3)
    target_bbox = _dequantize_token_bins(target_bins, NUM_POS).view(1, 8, 3)
    expected_center = F.smooth_l1_loss(pred_bbox.mean(dim=1), target_bbox.mean(dim=1))
    pred_size = pred_bbox.amax(dim=1) - pred_bbox.amin(dim=1)
    target_size = target_bbox.amax(dim=1) - target_bbox.amin(dim=1)
    expected_size = F.smooth_l1_loss(pred_size.clamp_min(1e-6).log(), target_size.clamp_min(1e-6).log())

    assert torch.allclose(aux["loss_layout_center"], expected_center, atol=1e-5)
    assert torch.allclose(aux["loss_layout_size"], expected_size, atol=1e-5)


def test_unsupported_layout_token_count_skips_geometry_losses():
    labels = _shift_labels(_bbox_bins()[:-1], seq_len=LAYOUT_TOKENS - 1)
    logits = _layout_logits(_bbox_bins()[:-1], seq_len=LAYOUT_TOKENS - 1)
    aux = _layout_auxiliary_losses(
        logits,
        labels,
        _token_types(seq_len=LAYOUT_TOKENS - 1),
        pos_token_offset=POS_OFFSET,
        num_pos_tokens=NUM_POS,
        loss_layout_ordinal_sigma=2.0,
    )

    assert aux["loss_layout_ordinal"].item() == 0.0
    assert aux["loss_layout_coord"].item() == 0.0
    assert aux["loss_layout_center"].item() == 0.0
    assert aux["loss_layout_size"].item() == 0.0


def test_legacy_loss_api_still_returns_three_values():
    labels = torch.tensor([[1, POS_OFFSET + 3, 2]], dtype=torch.long)
    logits = torch.zeros((1, 3, VOCAB))

    result = causal_lm_loss_with_token_types(logits, labels, vocab_size=VOCAB)

    assert isinstance(result, tuple)
    assert len(result) == 3
