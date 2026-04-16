# Exploration — Residual Skip Connection + Label Smoothing

**Novel:** Add a residual skip from conv block 1 output to conv block 2 output (1×1 projection to match channels). This gives the gradient a shorter path and often improves accuracy without adding meaningful parameters. Pair with label smoothing (ε=0.1) to reduce overconfidence on 1-epoch training.

## Time Budget
- Hard wall-clock limit: 60 seconds.
- After epoch 1, set remaining_epochs = max(1, floor(45 / epoch1_time)).
- If epoch1_time > 40s, stop immediately after epoch 1.

## Architecture
- Input: 32x32 RGB (CIFAR-10, 10 classes)
- **Block 1:** Conv2d(3→32, 3x3, pad=1) → BN → ReLU → MaxPool(2) → 16x16  [output: x1]
- **Block 2:** Conv2d(32→32, 3x3, pad=1) → BN → ReLU → MaxPool(2) → 8x8  [output: x2]
- **Residual:** shortcut = MaxPool(2)(x1)  — same spatial size, same channels; add to x2: out = x2 + shortcut
- Flatten → Linear(32*8*8 → 64) → ReLU → Dropout(0.25) → Linear(64 → 10)
- ~50k parameters (skip connection adds no parameters since channels match)

**Implementation note:** `shortcut = F.max_pool2d(x1, 2)` then `x2 = x2 + shortcut` before flatten.

## Training
- Optimizer: SGD, lr=0.05, momentum=0.9, weight_decay=1e-4
- Scheduler: CosineAnnealingLR(T_max=total_epochs)
- Epochs: dynamic
- Batch size: 256
- **Loss: CrossEntropyLoss(label_smoothing=0.1)**

## Augmentation
- RandomCrop(32, padding=4)
- RandomHorizontalFlip
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)