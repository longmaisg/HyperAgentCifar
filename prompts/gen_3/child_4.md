# Exploration — Lightweight Residual Network with Batch 256

## Rationale
Every Gen 1–2 child used a plain conv stack. Adding a single residual shortcut in the first block gives the gradient a direct path back and tends to accelerate early-epoch accuracy gains — exactly what matters in a 2-epoch budget. The architecture is kept deliberately narrow (32→48→96 instead of 32→64→128) so the residual block's extra conv layer does not inflate parameter count or per-epoch time beyond safe limits. Batch size 256 keeps epoch time well under 80 s. SGD + OneCycleLR is the schedule that has shown the best single-epoch progress across generations.

## Architecture
- Input: 32×32 RGB (CIFAR-10, 10 classes)
- Stem: Conv2d(3→32, 3×3, pad=1) → BN → ReLU
- Residual block (stride-1, same width):
  - Branch: Conv2d(32→32, 3×3, pad=1) → BN → ReLU → Conv2d(32→32, 3×3, pad=1) → BN
  - Skip: identity
  - Merge: add → ReLU → MaxPool(2×2)   [output: 32×16×16]
- Conv block 2: Conv2d(32→64, 3×3, pad=1) → BN → ReLU → MaxPool(2)   [output: 64×8×8]
- Conv block 3: Conv2d(64→96, 3×3, pad=1) → BN → ReLU → MaxPool(2)   [output: 96×4×4]
- Flatten (96×4×4 = 1536) → Linear(1536→256) → ReLU → Dropout(0.5) → Linear(256→10)
- Total params ≈ 730 k (well under 1 M)

## Training
- Optimizer: SGD, lr=0.01 (initial), momentum=0.9, weight_decay=1e-4
- Scheduler: OneCycleLR(max_lr=0.05, epochs=2, steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy='cos')
- Epochs: 2
- Batch size: 256
- Loss: CrossEntropyLoss

## Augmentation
- RandomHorizontalFlip
- RandomCrop(32, padding=4)
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)