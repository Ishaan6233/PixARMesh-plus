# MV Layout Ordinal/Coordinate Loss Method

## Motivation

Pure CE over 512 coordinate bins is metric-blind: one bin off and 250 bins off
cost the same. Soft ordinal targets, following Diaz and Marathe's SORD objective
(CVPR 2019), and soft-argmax regression, following Integral Human Pose Regression
(Sun et al., ECCV 2018, arXiv:1711.08229), address this failure mode by making
nearby bins and metric coordinates visible to the loss.

Earlier center and log-size auxiliary losses were removed because they are
mathematically redundant when all 24 corner coordinates are correct. Center and
size remain evaluation metrics, but they are no longer separate optimization
terms.

The honest caveat is that the headline mesh decode uses GT layout and GT mask,
so these losses reach downstream CD/F through the warm-started trunk. The direct
measurable win is Stage 1 layout metrics.

## Method

The method keeps the token CE layout loss and adds two auxiliary terms for rows
with the full 24 supervised bbox coordinate tokens. The implementation lives in
`src/models/loss.py`.

- `loss_layout_ordinal`: soft-CE over coordinate bins using Gaussian ordinal
  targets with sigma 2.
- `loss_layout_coord`: Smooth L1 from soft-argmax dequantized coordinates to
  target dequantized coordinates.

Rows without exactly 24 valid layout coordinate tokens are skipped for geometry
terms so SV, BPT, or partial-layout samples cannot be silently reshaped into the
wrong contract.

## Config Keys And Defaults

The Trellis2-MV stage configs inherit these defaults from
`configs/edgerunner_3d_front_trellis2_mv.yaml`:

- `model.loss_layout_ordinal_sigma: 2.0`
- `model.loss_layout_ordinal_weight: 1.0`
- `model.loss_layout_coord_weight: 1.0`

The shared `ModelConfig` dataclass defaults stay CE-only so SV and other model
families do not inherit MV-specific behavior unless their Hydra config opts in.

## Control Protocol

Use `+experiment=mv_layout_loss_ce` as the zero-weight control overlay. It sets
the ordinal sigma to `null` and both auxiliary weights to `0.0`, overriding
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
