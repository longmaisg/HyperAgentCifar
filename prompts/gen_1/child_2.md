# Mutation — Narrower Architecture (faster forward/backward pass)

**Change:** Reduce width of conv block 2 from 64→32 channels and shrink FC from 128→64. Fewer FLOPs per step means more epochs fit in the time window.

## Time Budget
- Hard wall-clock limit: 60 seconds.
- After epoch 1, set remaining_epochs = max(1, floor(45 / epoch1_time)).
- If epoch1_time > 40s, stop immediately after epoch 1.

## Architecture
- Input: 32x32 RGB (CIFAR-10, 10 classes)
- Conv block 1: Conv2d(3→32, 3x3, pad=1) → BN → ReLU → MaxPool(2)  → 16x16
- **Conv block 2: Conv2d(32→32, 3x3, pad=1) → BN → ReLU → MaxPool(2)** → 8x8
- Flatten → **Linear(32*8*8 → 64)** → ReLU → Dropout(0.25) → Linear(64 → 10)
- ~approx 50k parameters (well under budget)

## Training
- Optimizer: SGD, lr=0.05, momentum=0.9, weight_decay=1e-4
- Scheduler: CosineAnnealingLR(T_max = total_epochs)
- Epochs: dynamic
- Batch size: 256
- Loss: CrossEntropyLoss

## Augmentation
- RandomHorizontalFlip
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)

---