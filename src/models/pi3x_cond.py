"""Pi3X frozen geometry backbone."""

import torch
from .frozen_geo_encoder import FrozenGeoEncoder, register_geo_encoder


@register_geo_encoder("pi3x")
class Pi3XFrozenEncoder(FrozenGeoEncoder):
    """Frozen Pi3X wrapper.

    Pi3X and DiNOv2 share identical ImageNet normalization, so pixel_values
    produced by the DiNOv2 preprocessor are passed directly to Pi3X.encode()
    without double-normalization (Pi3X.forward() normalises; encode() does not).
    """

    def __init__(self, ckpt_path: str, greedy_anchor: bool = True, disable_multimodal: bool = True):
        super().__init__()
        from src.pi3x.models.pi3x import Pi3X

        self.pi3x = Pi3X.from_pretrained(ckpt_path)
        self.pi3x.requires_grad_(False)
        if disable_multimodal:
            self.pi3x.disable_multimodal()
        self.greedy_anchor = greedy_anchor

    @classmethod
    def from_model_cfg(cls, model_cfg) -> "Pi3XFrozenEncoder":
        # Read from ModelConfig's pi3x_* fields directly — no shim needed.
        return cls(
            ckpt_path=model_cfg.pi3x_ckpt_path,
            greedy_anchor=model_cfg.pi3x_greedy_anchor,
            disable_multimodal=model_cfg.pi3x_disable_multimodal,
        )

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> dict:
        """
        Args:
            pixel_values: (B, 3, H, W) DiNOv2-normalised images (ImageNet mean/std).
        Returns:
            dict with 'local_points' (B, H, W, 3) and 'conf' (B, H, W, 1) in camera frame.
        """
        B, C, H, W = pixel_values.shape
        # Reshape to (B, N=1, C, H, W) — Pi3X encode expects (B, N, C, H, W)
        imgs = pixel_values.unsqueeze(1)
        patch_h, patch_w = H // 14, W // 14

        # encode() receives pre-normalised images (same ImageNet stats → no double-normalisation)
        hidden, _, _, _, _ = self.pi3x.encode(imgs, with_prior=False)
        hidden = hidden.reshape(B, 1, -1, self.pi3x.dec_embed_dim)
        hidden, pos = self.pi3x.decode(hidden, 1, H, W, None, None)
        out = self.pi3x.forward_head(hidden, pos, B, 1, H, W, patch_h, patch_w)

        local_points = out["local_points"][:, 0]   # (B, H, W, 3)
        conf         = out["conf"][:, 0]            # (B, H, W, 1)
        return {
            "local_points": local_points.to(pixel_values.dtype),
            "conf":         conf.to(pixel_values.dtype),
        }

    @torch.no_grad()
    def forward_multiview(self, pixel_values: torch.Tensor) -> dict:
        """Greedy-anchor multi-view forward.

        Runs Pi3X independently on each view, selects the anchor with highest
        mean confidence (greedy seed), returns that view's geometry.

        Args:
            pixel_values: (B, N, 3, H, W)
        Returns:
            dict with 'local_points' (B, H, W, 3) and 'conf' (B, H, W, 1).
        """
        B, N, C, H, W = pixel_values.shape
        if N == 1:
            return self.forward(pixel_values[:, 0])

        patch_h, patch_w = H // 14, W // 14
        all_lp, all_cf, mean_confs = [], [], []

        for n in range(N):
            view = pixel_values[:, n:n + 1]
            hidden, _, _, _, _ = self.pi3x.encode(view, with_prior=False)
            hidden = hidden.reshape(B, 1, -1, self.pi3x.dec_embed_dim)
            hidden, pos = self.pi3x.decode(hidden, 1, H, W, None, None)
            out = self.pi3x.forward_head(hidden, pos, B, 1, H, W, patch_h, patch_w)
            lp = out["local_points"][:, 0]
            cf = out["conf"][:, 0]
            all_lp.append(lp)
            all_cf.append(cf)
            mean_confs.append(cf.mean(dim=[1, 2, 3]))

        # Greedy seed: pick the view with highest mean confidence per batch item
        anchor_inds = torch.stack(mean_confs, dim=1).argmax(dim=1)   # (B,)

        lp_stack = torch.stack(all_lp, dim=1)   # (B, N, H, W, 3)
        cf_stack = torch.stack(all_cf, dim=1)   # (B, N, H, W, 1)
        idx = anchor_inds.view(B, 1, 1, 1, 1)
        local_points = lp_stack.gather(1, idx.expand(B, 1, H, W, 3)).squeeze(1)
        conf         = cf_stack.gather(1, idx.expand(B, 1, H, W, 1)).squeeze(1)
        return {
            "local_points": local_points.to(pixel_values.dtype),
            "conf":         conf.to(pixel_values.dtype),
        }
