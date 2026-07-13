import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass
from transformers.modeling_outputs import CausalLMOutputWithPast
from src.data.typing import TokenType


def fixed_cross_entropy_with_token_types(
    source: torch.Tensor,
    target: torch.Tensor,
    token_type_ids: Optional[torch.Tensor] = None,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    loss_layout_scale: Optional[float] = None,
    **kwargs,
) -> torch.Tensor:
    if num_items_in_batch is not None:
        reduction = "sum" if token_type_ids is None else "none"
    else:
        reduction = "mean"
    loss = nn.functional.cross_entropy(
        source, target, ignore_index=ignore_index, reduction=reduction
    )
    loss_layout = None
    loss_object = None
    if reduction in ("sum", "none"):
        # just in case users pass an int for num_items_in_batch, which could be the case for custom trainer
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        if token_type_ids is not None:
            layout_tokens_mask = token_type_ids == TokenType.LAYOUT
            object_tokens_mask = token_type_ids == TokenType.OBJECT
            valid_layout_tokens = layout_tokens_mask.sum().clip(min=1)
            valid_object_tokens = object_tokens_mask.sum().clip(min=1)
            loss_layout_masked = loss * layout_tokens_mask
            loss_layout_sum = loss_layout_masked.sum()
            loss_layout = loss_layout_sum / valid_layout_tokens
            loss_object = (loss * object_tokens_mask).sum() / valid_object_tokens
            if loss_layout_scale is not None:
                loss_others_masked = loss * ~layout_tokens_mask
                loss = loss_layout_scale * loss_layout_sum + loss_others_masked.sum()
            else:
                loss = loss.sum()
        loss = loss / num_items_in_batch
    return loss, loss_layout, loss_object


def _dequantize_token_bins(bins: torch.Tensor, num_pos_tokens: int) -> torch.Tensor:
    return ((bins.float() + 0.5) / float(num_pos_tokens)) * 2.0 - 1.0


def _zero_like_loss(logits: torch.Tensor) -> torch.Tensor:
    return logits.sum() * 0.0


# d/dx log(x) = 1/x: a clamp floor near 0 makes the log-size loss's gradient
# unbounded for near-degenerate bbox extents (thin objects) even though the
# loss *value* stays small. 1e-2 is ~2.5 dequantized-bin widths (2/num_pos_tokens
# at num_pos_tokens=512), bounding the worst-case per-term gradient to ~1e2 --
# well below the observed healthy CE-dominated grad_norm (~1e4) -- instead of
# the ~1e6-1e7 spikes measured at 1e-6 (outputs/da3/train/mv_layout_loss/D_geometry).
_SIZE_LOG_FLOOR = 1e-2


def _layout_auxiliary_losses(
    logits: torch.Tensor,
    shift_labels: torch.Tensor,
    token_type_ids: Optional[torch.Tensor],
    *,
    pos_token_offset: int,
    num_pos_tokens: int,
    loss_layout_ordinal_sigma: Optional[float] = None,
    loss_layout_geometry_tokens: int = 24,
    ignore_index: int = -100,
) -> dict[str, Optional[torch.Tensor]]:
    """Compute optional geometric losses for EdgeRunner full-layout bbox tokens.

    Only samples with exactly `loss_layout_geometry_tokens` supervised layout
    coordinate tokens are used. This avoids silently reshaping SV/BPT/partial
    layouts into the wrong geometry contract.
    """
    zero = _zero_like_loss(logits)
    out: dict[str, Optional[torch.Tensor]] = {
        "loss_layout_ordinal": None,
        "loss_layout_coord": zero,
        "loss_layout_center": zero,
        "loss_layout_size": zero,
    }
    if token_type_ids is None or num_pos_tokens <= 0 or loss_layout_geometry_tokens <= 0:
        return out

    coord_mask = (
        (token_type_ids == TokenType.LAYOUT)
        & (shift_labels != ignore_index)
        & (shift_labels >= pos_token_offset)
        & (shift_labels < pos_token_offset + num_pos_tokens)
    )
    counts = coord_mask.sum(dim=1)
    valid_rows = torch.nonzero(counts == loss_layout_geometry_tokens, as_tuple=False).flatten()
    if valid_rows.numel() == 0:
        if loss_layout_ordinal_sigma is not None and float(loss_layout_ordinal_sigma) > 0:
            out["loss_layout_ordinal"] = zero
        return out

    label_rows = []
    logit_rows = []
    for row in valid_rows.tolist():
        idx = torch.nonzero(coord_mask[row], as_tuple=False).flatten()
        label_rows.append(shift_labels[row, idx] - pos_token_offset)
        logit_rows.append(logits[row, idx, pos_token_offset : pos_token_offset + num_pos_tokens])

    layout_labels = torch.stack(label_rows, dim=0).long()
    layout_logits = torch.stack(logit_rows, dim=0)
    log_probs = nn.functional.log_softmax(layout_logits, dim=-1)
    probs = log_probs.exp()

    sigma = loss_layout_ordinal_sigma
    if sigma is not None and float(sigma) > 0:
        bin_ids = torch.arange(num_pos_tokens, device=logits.device, dtype=layout_logits.dtype)
        dist = bin_ids.view(1, 1, -1) - layout_labels.to(layout_logits.dtype).unsqueeze(-1)
        soft_targets = torch.exp(-(dist**2) / (2.0 * float(sigma) ** 2))
        soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        out["loss_layout_ordinal"] = -(soft_targets * log_probs).sum(dim=-1).mean()

    bin_centers = _dequantize_token_bins(
        torch.arange(num_pos_tokens, device=logits.device), num_pos_tokens
    ).to(layout_logits.dtype)
    pred_coords = (probs * bin_centers.view(1, 1, -1)).sum(dim=-1)
    target_coords = _dequantize_token_bins(layout_labels, num_pos_tokens).to(pred_coords.dtype)

    out["loss_layout_coord"] = nn.functional.smooth_l1_loss(
        pred_coords, target_coords, reduction="mean"
    )
    pred_bbox = pred_coords.view(-1, 8, 3)
    target_bbox = target_coords.view(-1, 8, 3)
    pred_center = pred_bbox.mean(dim=1)
    target_center = target_bbox.mean(dim=1)
    out["loss_layout_center"] = nn.functional.smooth_l1_loss(
        pred_center, target_center, reduction="mean"
    )
    pred_size = (pred_bbox.amax(dim=1) - pred_bbox.amin(dim=1)).clamp_min(_SIZE_LOG_FLOOR)
    target_size = (target_bbox.amax(dim=1) - target_bbox.amin(dim=1)).clamp_min(_SIZE_LOG_FLOOR)
    out["loss_layout_size"] = nn.functional.smooth_l1_loss(
        pred_size.log(), target_size.log(), reduction="mean"
    )
    return out


def causal_lm_loss_with_token_types(
    logits,
    labels,
    vocab_size: int,
    token_type_ids: Optional[torch.Tensor] = None,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    loss_layout_scale: Optional[float] = None,
    pos_token_offset: int = 0,
    num_pos_tokens: int = 0,
    loss_layout_ordinal_sigma: Optional[float] = None,
    loss_layout_ordinal_weight: float = 0.0,
    loss_layout_coord_weight: float = 0.0,
    loss_layout_center_weight: float = 0.0,
    loss_layout_size_weight: float = 0.0,
    loss_layout_geometry_tokens: int = 24,
    return_layout_components: bool = False,
    **kwargs,
) -> (
    tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    | tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        dict[str, Optional[torch.Tensor]],
    ]
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    shift_logits = logits

    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()
    else:
        shift_labels = shift_labels.contiguous()
    shift_labels_2d = shift_labels.to(logits.device)

    if token_type_ids is not None:
        token_type_ids = nn.functional.pad(
            token_type_ids,
            (0, 1),
            value=TokenType.PADDING,
        )
        token_type_ids = token_type_ids[..., 1:].contiguous().to(logits.device)
    token_type_ids_2d = token_type_ids

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    token_type_ids_flat = token_type_ids_2d.view(-1) if token_type_ids_2d is not None else None
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    loss, loss_layout, loss_object = fixed_cross_entropy_with_token_types(
        logits,
        shift_labels,
        token_type_ids=token_type_ids_flat,
        num_items_in_batch=num_items_in_batch,
        ignore_index=ignore_index,
        loss_layout_scale=loss_layout_scale,
        **kwargs,
    )

    components: dict[str, Optional[torch.Tensor]] = {
        "loss_layout_token": loss_layout,
        "loss_layout_ordinal": None,
        "loss_layout_coord": None,
        "loss_layout_center": None,
        "loss_layout_size": None,
    }
    aux_requested = (
        (loss_layout_ordinal_sigma is not None and float(loss_layout_ordinal_weight) != 0.0)
        or float(loss_layout_coord_weight) != 0.0
        or float(loss_layout_center_weight) != 0.0
        or float(loss_layout_size_weight) != 0.0
    )
    if aux_requested:
        aux = _layout_auxiliary_losses(
            shift_logits,
            shift_labels_2d,
            token_type_ids_2d,
            pos_token_offset=pos_token_offset,
            num_pos_tokens=num_pos_tokens,
            loss_layout_ordinal_sigma=loss_layout_ordinal_sigma,
            loss_layout_geometry_tokens=loss_layout_geometry_tokens,
            ignore_index=ignore_index,
        )
        components.update(aux)
        if components["loss_layout_ordinal"] is not None:
            loss = loss + float(loss_layout_ordinal_weight) * components["loss_layout_ordinal"]
        if components["loss_layout_coord"] is not None:
            loss = loss + float(loss_layout_coord_weight) * components["loss_layout_coord"]
        if components["loss_layout_center"] is not None:
            loss = loss + float(loss_layout_center_weight) * components["loss_layout_center"]
        if components["loss_layout_size"] is not None:
            loss = loss + float(loss_layout_size_weight) * components["loss_layout_size"]

    if return_layout_components:
        return loss, loss_layout, loss_object, components
    return loss, loss_layout, loss_object


@dataclass
class CustomCausalLMOutputWithTokenTypes(CausalLMOutputWithPast):
    loss_layout: Optional[torch.Tensor] = None
    loss_object: Optional[torch.Tensor] = None
    loss_layout_token: Optional[torch.Tensor] = None
    loss_layout_ordinal: Optional[torch.Tensor] = None
    loss_layout_coord: Optional[torch.Tensor] = None
    loss_layout_center: Optional[torch.Tensor] = None
    loss_layout_size: Optional[torch.Tensor] = None
