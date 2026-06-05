---
pipeline_tag: image-to-3d
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
license: cc-by-nc-4.0
---

# $\pi^3$: Permutation-Equivariant Visual Geometry Learning

This repository contains the weights for **Pi3X**, an enhanced version of the $\pi^3$ model introduced in the paper [$\pi^3$: Permutation-Equivariant Visual Geometry Learning](https://huggingface.co/papers/2507.13347).

$\pi^3$ is a feed-forward neural network for visual geometry reconstruction that eliminates the need for a fixed reference view. It employs a fully permutation-equivariant architecture to predict affine-invariant camera poses and scale-invariant local point maps from an unordered set of images, making it robust to input ordering and achieving state-of-the-art performance.

- **Project Page:** [yyfz.github.io/pi3/](https://yyfz.github.io/pi3/)
- **GitHub Repository:** [github.com/yyfz/Pi3](https://github.com/yyfz/Pi3)
- **Demo:** [Hugging Face Space](https://huggingface.co/spaces/yyfz233/Pi3)

## Pi3X Engineering Update
Pi3X is an enhanced version focusing on flexibility and reconstruction quality:
* **Smoother Reconstruction:** Uses a Convolutional Head to reduce grid-like artifacts.
* **Flexible Conditioning:** Supports optional injection of camera poses, intrinsics, and depth.
* **Reliable Confidence:** Predicts continuous quality levels for better noise filtering.
* **Metric Scale:** Supports approximate metric scale reconstruction.

## Sample Usage

To use this model, you need to clone the [official repository](https://github.com/yyfz/Pi3) and install the dependencies.

```python
import torch
from pi3.models.pi3x import Pi3X            # new version (Recommended)
from pi3.utils.basic import load_images_as_tensor 

# --- Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()

# --- Load Data ---
# Load a sequence of N images into a tensor (N, 3, H, W)
# pixel values in the range [0, 1]
imgs = load_images_as_tensor('path/to/your/data', interval=10).to(device)

# --- Inference ---
print("Running model inference...")
# Use mixed precision for better performance on compatible GPUs
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

with torch.no_grad():
    with torch.amp.autocast('cuda', dtype=dtype):
        # Add a batch dimension -> (1, N, 3, H, W)
        results = model(imgs[None])

print("Reconstruction complete!")
# Access outputs: results['points'], results['camera_poses'] and results['local_points'].
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{wang2025pi,
  title={$\pi^3$: Permutation-Equivariant Visual Geometry Learning},
  author={Wang, Yifan and Zhou, Jianjun and Zhu, Haoyi and Chang, Wenzheng and Zhou, Yang and Li, Zizun and Chen, Junyi and Pang, Jiangmiao and Shen, Chunhua and He, Tong},
  journal={arXiv preprint arXiv:2507.13347},
  year={2025}
}
```

## License
- **Code**: BSD 3-Clause
- **Model Weights**: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (Strictly Non-Commercial)