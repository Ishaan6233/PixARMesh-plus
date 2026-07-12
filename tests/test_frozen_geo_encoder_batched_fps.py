import pytest
import torch

from src.models.frozen_geo_encoder import (
    _adaptive_filter_cap,
    _batched_fps_fixed_k,
    _batched_pool_fps,
    _cap_points,
    adaptive_fps_voxelize,
    fps_centroid_seeded,
)

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="pytorch3d sample_farthest_points requires CUDA"
)


@cuda_only
def test_batched_fps_fixed_k_matches_per_item_dense_and_sparse():
    device = torch.device("cuda")
    torch.manual_seed(0)
    dense = torch.randn(500, 3, device=device)   # length > K: real FPS needed
    sparse = torch.randn(5, 3, device=device)    # length < K: pad-with-last-point
    exact = torch.randn(32, 3, device=device)    # length == K: passthrough
    K = 32

    batched = _batched_fps_fixed_k([dense, sparse, exact], K, device, torch.float32)
    assert batched.shape == (3, K, 3)

    ref_dense = fps_centroid_seeded(dense.unsqueeze(0), K).squeeze(0)
    ref_sparse = fps_centroid_seeded(sparse.unsqueeze(0), K).squeeze(0)
    ref_exact = fps_centroid_seeded(exact.unsqueeze(0), K).squeeze(0)

    assert torch.equal(batched[0], ref_dense)
    assert torch.equal(batched[1], ref_sparse)
    assert torch.equal(batched[2], ref_exact)
    # Sparse item's padding repeats its own last (real) point, not a neighbor's data.
    assert torch.equal(batched[1, 5:], sparse[-1:].expand(K - 5, 3))


@cuda_only
def test_batched_fps_fixed_k_empty_item_returns_zeros():
    device = torch.device("cuda")
    empty = torch.zeros(0, 3, device=device)
    dense = torch.randn(200, 3, device=device)
    K = 16
    batched = _batched_fps_fixed_k([empty, dense], K, device, torch.float32)
    assert torch.equal(batched[0], torch.zeros(K, 3, device=device))


@cuda_only
def test_batched_pool_fps_sparse_passes_through_unchanged():
    device = torch.device("cuda")
    torch.manual_seed(1)
    sparse = torch.randn(50, 3, device=device)  # length <= K: no FPS, no padding
    K = 8192
    out = _batched_pool_fps([sparse], K, device)
    assert out[0].shape == (50, 3)
    assert torch.equal(out[0], sparse)  # identical order and values, not FPS-reordered


@cuda_only
def test_batched_pool_fps_dense_matches_direct_fps_and_mixed_batch_is_index_correct():
    device = torch.device("cuda")
    torch.manual_seed(2)
    dense_a = torch.randn(9000, 3, device=device)
    sparse = torch.randn(100, 3, device=device)
    dense_b = torch.randn(20000, 3, device=device)
    K = 8192

    out = _batched_pool_fps([dense_a, sparse, dense_b], K, device)
    assert out[0].shape == (K, 3)
    assert out[1].shape == (100, 3)
    assert out[2].shape == (K, 3)
    assert torch.equal(out[1], sparse)

    # Dense items' selection must match calling fps_centroid_seeded directly on
    # that SAME item alone (i.e. batching must not mix candidates across items).
    ref_a = fps_centroid_seeded(dense_a.unsqueeze(0), K).squeeze(0)
    ref_b = fps_centroid_seeded(dense_b.unsqueeze(0), K).squeeze(0)
    assert torch.equal(out[0], ref_a)
    assert torch.equal(out[2], ref_b)


@cuda_only
@pytest.mark.parametrize("with_conf", [True, False])
def test_adaptive_filter_cap_matches_single_item_adaptive_fps_voxelize(with_conf):
    device = torch.device("cuda")
    torch.manual_seed(3)
    pts = torch.randn(2000, 3, device=device)
    conf = torch.rand(2000, device=device) if with_conf else None
    n_voxels, conf_threshold, max_in = 256, 0.3, 65536

    filtered = _adaptive_filter_cap(pts, conf, n_voxels, conf_threshold, max_in)

    # Replicate adaptive_fps_voxelize's single-item filter+cap step manually (the
    # piece _adaptive_filter_cap factors out so the FPS itself can be batched).
    if conf is None:
        expected = _cap_points(pts, max_in)
    else:
        keep = conf >= conf_threshold
        pts_keep = pts[keep]
        if pts_keep.shape[0] < max(n_voxels, 4):
            pts_keep = pts
        expected = _cap_points(pts_keep, max_in)
    assert torch.equal(filtered, expected)

    # And feeding the filtered candidates through fps_centroid_seeded must equal
    # calling the original single-item adaptive_fps_voxelize end-to-end.
    direct = adaptive_fps_voxelize(
        pts.unsqueeze(0), conf.unsqueeze(0) if conf is not None else None,
        n_voxels, conf_threshold, max_in,
    ).squeeze(0)
    via_split = fps_centroid_seeded(filtered.unsqueeze(0), n_voxels).squeeze(0)
    assert torch.equal(direct, via_split)


@cuda_only
def test_adaptive_filter_cap_reverts_to_unfiltered_when_too_few_pass_threshold():
    device = torch.device("cuda")
    pts = torch.randn(100, 3, device=device)
    conf = torch.zeros(100, device=device)  # nothing passes threshold
    conf[0] = 1.0  # exactly one point passes
    n_voxels = 50  # max(n_voxels, 4) = 50 > 1 kept point -> must revert to all 100

    filtered = _adaptive_filter_cap(pts, conf, n_voxels, conf_threshold=0.5, max_in=65536)
    assert filtered.shape[0] == 100
    assert torch.equal(filtered, pts)
