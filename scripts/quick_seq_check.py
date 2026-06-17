"""
Quick sequence-length sanity check for a BPT checkpoint.
Run against intermediate checkpoints to verify the float32 retrain is on track.

Usage:
    python scripts/quick_seq_check.py <checkpoint_path> [--num-samples 4]

Healthy training signals (approximate):
    step ~5k:  mean_tokens > 300   (up from ~135 in the bf16 run)
    step ~10k: mean_tokens > 800
    step ~25k: mean_tokens > 2000  (target: ~3930)
"""
import sys
sys.path.insert(0, '/work/spatial/Ishaan/PixARMesh+')
import argparse, torch, numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint')
parser.add_argument('--num-samples', type=int, default=4)
parser.add_argument('--max-tokens', type=int, default=1000)
args = parser.parse_args()

print(f"Loading {args.checkpoint} ...")
from src.utils.inference import prepare_model_for_inference
model, _, _ = prepare_model_for_inference(is_bpt=True, checkpoint=args.checkpoint)
model.eval().cuda()

print(f"Model param dtype: {next(model.parameters()).dtype}")

from x_transformers.autoregressive_wrapper import top_k, top_p
from functools import partial

def joint_filter(logits, k=50, p=0.95):
    logits = top_k(logits, k=k)
    logits = top_p(logits, thres=p)
    return logits

INDICATOR_ID = 5121
EOS_ID = 5120
lengths = []

for i in range(args.num_samples):
    torch.manual_seed(i * 7 + 42)
    B = 1
    # Simple unit-sphere point cloud (in-distribution enough for a sanity check)
    theta = torch.rand(B, 4096) * 2 * np.pi
    phi   = torch.acos(2 * torch.rand(B, 4096) - 1)
    x_ = torch.sin(phi)*torch.cos(theta); y_ = torch.sin(phi)*torch.sin(theta); z_ = torch.cos(phi)
    pts = torch.stack([x_,y_,z_],-1)
    cond_pcs    = torch.cat([pts, pts], -1).cuda().bfloat16()
    cond_pcs_2d = torch.zeros(B, 4096, 2, device='cuda', dtype=torch.bfloat16)
    ctx_pcs     = cond_pcs.clone()
    ctx_pcs_2d  = cond_pcs_2d.clone()
    pixel_values = torch.zeros(B, 3, 364, 476, device='cuda', dtype=torch.bfloat16)

    layout_seq = torch.tensor([[0]*18 + [INDICATOR_ID]], device='cuda', dtype=torch.long)

    with torch.no_grad():
        cond_embeds = model.get_inputs_with_cond(
            input_ids=layout_seq,
            cond_pcs=cond_pcs, cond_pcs_2d=cond_pcs_2d,
            ctx_pcs=ctx_pcs, ctx_pcs_2d=ctx_pcs_2d,
            pixel_values=pixel_values,
        )
        has_nan = cond_embeds.isnan().any().item()
        
        codes = model.generate(
            inputs=layout_seq,
            cond_embeds=cond_embeds,
            filter_logits_fn=partial(joint_filter, k=50, p=0.95),
            temperature=0.5,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            tqdm_position=i,
        )
    
    # Count generated mesh tokens (exclude layout+indicator prefix, EOS/-1 suffix)
    seq = codes[0].tolist()
    prefix_len = 19  # 18 layout + 1 indicator
    mesh_tokens = [t for t in seq[prefix_len:] if t != EOS_ID and t != -1]
    n_tokens = len(mesh_tokens)
    n_faces  = n_tokens // 2  # rough: ~2 tokens per face
    lengths.append(n_tokens)
    print(f"  sample {i}: cond_nan={has_nan}, mesh_tokens={n_tokens}, approx_faces={n_faces}")

print(f"\nMean mesh tokens: {np.mean(lengths):.0f} (target ~3930; bf16 baseline ~135)")
print(f"Min/max: {min(lengths)}/{max(lengths)}")
if np.mean(lengths) > 500:
    print("✓ LOOKS HEALTHY — float32 training is producing longer sequences")
elif np.mean(lengths) > 200:
    print("~ IMPROVING — longer than bf16 baseline, keep monitoring")
else:
    print("✗ STILL SHORT — may still be bf16 issue or early in training")
