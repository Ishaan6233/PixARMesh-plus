# MV Layout Geometry Loss Method

## Motivation

Pure CE over 512 coordinate bins is metric-blind: one bin off and 250 bins off
cost the same. In the geometry-loss ablations, `loss_layout` plateaued around
1.75-1.78 while geometric error stayed large. Soft ordinal targets, following
Diaz and Marathe's SORD objective (CVPR 2019), and soft-argmax regression,
following Integral Human Pose Regression (Sun et al., ECCV 2018,
arXiv:1711.08229), address exactly this failure mode by making nearby bins and
metric coordinates visible to the loss.

The `s1c` run had the observed object extent in its own conditioning prefix via
the AABB token, yet predicted size at chance. The evidence was present at the
input, but pure CE gave no direct gradient pressure to read it. The log-size
term adds that pressure.

MV observes about 64% more object surface than SV, but the view-usage probe
showed restricted-view controls beating all-views. The extent along the viewing
ray is the signal SV cannot see, and a size loss pays gradient for exactly that
MV-only evidence.

Log-space size is relative error, matching the standard box-parameterization
intuition used since Faster R-CNN (arXiv:1506.01497). Disentangled center and
size terms also improve 3D boxes in MonoDIS (ICCV 2019, arXiv:1905.12365), and
ATISS (NeurIPS 2021, arXiv:2110.03675) avoids uniform-bin CE for 3D-FRONT
layout attributes.

The honest caveat is that the headline mesh decode uses GT layout and GT mask,
so these losses reach downstream CD/F through the warm-started trunk. The direct
measurable win is Stage 1 layout metrics.

## Method

The method keeps the token CE layout loss and adds four geometric terms for rows
with the full 24 supervised bbox coordinate tokens. The implementation lives in
`src/models/loss.py:57-138`, with the weighted addition in
`src/models/loss.py:213-238`.

- `loss_layout_ordinal`: soft-CE over coordinate bins using Gaussian ordinal
  targets with sigma 2.
- `loss_layout_coord`: Smooth L1 from soft-argmax dequantized coordinates to
  target dequantized coordinates.
- `loss_layout_center`: Smooth L1 on the mean of the eight bbox corners.
- `loss_layout_size`: Smooth L1 on log box size from the min/max span of the
  eight corners.

Rows without exactly 24 valid layout coordinate tokens are skipped for geometry
terms so SV, BPT, or partial-layout samples cannot be silently reshaped into the
wrong contract.

## Config Keys And Defaults

The Trellis2-MV stage configs inherit these defaults from
`configs/edgerunner_3d_front_trellis2_mv.yaml`:

- `model.loss_layout_ordinal_sigma: 2.0`
- `model.loss_layout_ordinal_weight: 1.0`
- `model.loss_layout_coord_weight: 1.0`
- `model.loss_layout_center_weight: 0.5`
- `model.loss_layout_size_weight: 0.5`

The shared `ModelConfig` dataclass defaults stay CE-only so SV and other model
families do not inherit MV-specific behavior unless their Hydra config opts in.

## Control Protocol

Use `+experiment=mv_layout_loss_ce` as the zero-weight control overlay. It sets
the ordinal sigma to `null` and all four geometry weights to `0.0`, overriding
the Trellis2-MV defaults while keeping the same training config otherwise.

## Gates

Stage 1 layout acceptance:

- Size L2 must beat chance `0.164`. The `s1c` checkpoint had the AABB token in
  its prefix and still sized at chance under pure CE.
- Center L2 should beat the prior best `0.229`.

Downstream mesh acceptance:

- Compare Stage 2 against the frozen SV baseline: `avg_cd 0.008452` over
  `432` objects.
- Use `--align-sample-points 5000`.
- Coverage must be at least 95% before trusting CD/F.
