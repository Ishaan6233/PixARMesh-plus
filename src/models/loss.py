import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass
from transformers.modeling_outputs import CausalLMOutputWithPast
from src.data.typing import TokenType


def _ordinal_softce(
    logits: torch.Tensor,   # (N, V) float
    labels: torch.Tensor,   # (N,) long — may contain ignore_index
    sigma: float,
    ignore_index: int = -100,
) -> torch.Tensor:          # (N,) float — per-token soft CE, 0 where labels==ignore_index
    """Gaussian soft-label cross-entropy for ordinal token sequences.

    Replaces the one-hot label for token i with a normalised Gaussian spread:
        p_soft(k | true=i) ∝ exp(-(k-i)² / (2σ²))
    This teaches the model that adjacent bins are nearly correct, giving a gradient
    signal that scales with the magnitude of the coordinate error — unlike hard CE
    which treats every wrong bin equally.  σ=2 bins corresponds to ~0.4% of 512 bins.
    """
    N, V = logits.shape
    device = logits.device
    window = max(1, int(3.0 * sigma + 0.5))

    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)  # (N, V)
    valid = (labels != ignore_index)   # (N,) bool
    safe_labels = labels.clone()
    safe_labels[~valid] = 0            # avoid out-of-bounds index

    offsets = torch.arange(-window, window + 1, device=device, dtype=torch.float32)
    gauss_w = torch.exp(-0.5 * (offsets / sigma) ** 2)
    gauss_w = gauss_w / gauss_w.sum()   # normalise to probability distribution

    # Vectorised: gather all neighbour log-probs in one call, weight-sum in one call.
    all_offsets = offsets.long()                                                      # (2W+1,)
    neighbors = (safe_labels.unsqueeze(1) + all_offsets.unsqueeze(0)).clamp(0, V - 1)  # (N, 2W+1)
    log_p = log_probs.gather(1, neighbors)                                            # (N, 2W+1)
    loss = -(log_p * gauss_w.unsqueeze(0)).sum(1)                                     # (N,)

    return (loss * valid.float()).to(logits.dtype)


def fixed_cross_entropy_with_token_types(
    source: torch.Tensor,
    target: torch.Tensor,
    token_type_ids: Optional[torch.Tensor] = None,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    loss_layout_scale: Optional[float] = None,
    loss_layout_ordinal_sigma: Optional[float] = None,
    **kwargs,
) -> torch.Tensor:
    # Always compute per-token losses; needed for optional ordinal replacement and
    # loss_layout/loss_object decomposition.  Reduction is applied manually below.
    loss = nn.functional.cross_entropy(
        source, target, ignore_index=ignore_index, reduction="none"
    )  # (N,)

    # Ordinal label smoothing: replace hard CE with Gaussian soft-label CE for layout tokens.
    # Applied before the layout/object decomposition so loss_layout reflects the smoothed value.
    if loss_layout_ordinal_sigma is not None and token_type_ids is not None:
        layout_pos = token_type_ids == TokenType.LAYOUT
        if layout_pos.any():
            ord_loss = _ordinal_softce(source, target, loss_layout_ordinal_sigma, ignore_index)
            loss = torch.where(layout_pos, ord_loss.to(loss.dtype), loss)

    loss_layout = None
    loss_object = None
    if token_type_ids is not None:
        layout_tokens_mask = token_type_ids == TokenType.LAYOUT
        object_tokens_mask = token_type_ids == TokenType.OBJECT
        valid_layout_tokens = layout_tokens_mask.sum().clip(min=1)
        valid_object_tokens = object_tokens_mask.sum().clip(min=1)
        loss_layout_sum = (loss * layout_tokens_mask).sum()
        loss_layout = loss_layout_sum / valid_layout_tokens
        loss_object = (loss * object_tokens_mask).sum() / valid_object_tokens
        if loss_layout_scale is not None:
            loss = loss_layout_scale * loss_layout_sum + (loss * ~layout_tokens_mask).sum()
        else:
            loss = loss.sum()
    else:
        loss = loss.sum()

    if num_items_in_batch is not None:
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    else:
        # Normalise by the number of non-ignored tokens (equivalent to reduction="mean")
        valid_n = (target != ignore_index).sum().clip(min=1).to(loss.device)
        loss = loss / valid_n

    return loss, loss_layout, loss_object


def causal_lm_loss_with_token_types(
    logits,
    labels,
    vocab_size: int,
    token_type_ids: Optional[torch.Tensor] = None,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    loss_layout_scale: Optional[float] = None,
    loss_layout_ordinal_sigma: Optional[float] = None,
    **kwargs,
) -> torch.Tensor:
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()

    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    if token_type_ids is not None:
        token_type_ids = nn.functional.pad(
            token_type_ids,
            (0, 1),
            value=TokenType.PADDING,
        )
        token_type_ids = token_type_ids[..., 1:].contiguous()
        token_type_ids = token_type_ids.view(-1)
        token_type_ids = token_type_ids.to(logits.device)

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    loss, loss_layout, loss_object = fixed_cross_entropy_with_token_types(
        logits,
        shift_labels,
        token_type_ids=token_type_ids,
        num_items_in_batch=num_items_in_batch,
        ignore_index=ignore_index,
        loss_layout_scale=loss_layout_scale,
        loss_layout_ordinal_sigma=loss_layout_ordinal_sigma,
        **kwargs,
    )
    return loss, loss_layout, loss_object


@dataclass
class CustomCausalLMOutputWithTokenTypes(CausalLMOutputWithPast):
    loss_layout: Optional[torch.Tensor] = None
    loss_object: Optional[torch.Tensor] = None
