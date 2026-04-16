# Explore — Residual Connections + Label Smoothing + Cosine Annealing

**Rationale:** Add skip connections (ResNet-style) to combat vanishing gradients in a short 3-epoch run. Label smoothing (ε=0.1) regularizes without adding parameters. The residual path requires a 1x1 projection conv when channel counts change. Est. params ~660K, safely under 1M. Batch 256 keeps training within 175s.

## Architecture
- Input: 32x32 RGB (CIFAR-10, 10 classes)
- **Residual block 1**: 
  - Main: Conv2d(3→32, 3x3, pad=1) → BN → ReLU → Conv2d(32→32, 3x3, pad=1) → BN
  - Skip: identity (same dims)
  - → ReLU → MaxPool(2)
- **Residual block 2**:
  - Main: Conv2d(32→64, 3x3, pad=1) → BN → ReLU → Conv2d(64→64, 3x3, pad=1) → BN
  - Skip: Conv2d(32→64, 1x1) (projection)
  - → ReLU → MaxPool(2)
- **Residual block 3**:
  - Main: Conv2d(64→128, 3x3, pad=1) → BN → ReLU → Conv2d(128→128, 3x3, pad=1) → BN
  - Skip: Conv2d(64→128, 1x1) (projection)
  - → ReLU → MaxPool(2)
- GlobalAvgPool(4x4→1x1) → Linear(128→10)  ← replaces large FC head
- Est. params: ~370K

## Training
- Optimizer: SGD, lr=0.05, momentum=0.9, weight_decay=1e-4
- Scheduler: CosineAnnealingLR(T_max=epochs)
- Epochs: 3
- Batch size: 256
- Loss: CrossEntropyLoss with label_smoothing=0.1

## Augmentation
- RandomHorizontalFlip
- RandomCrop(32, padding=4)
- Normalize mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616)