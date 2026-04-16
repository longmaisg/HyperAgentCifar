# Mutation — Larger Batch + Aggressive Time Guard

**Change:** Batch size 256→512 to halve per-epoch time. Also tighten the time guard so the process never approaches the kill wall.

## Time Budget
- Hard wall-clock limit: 60 seconds.
- After epoch 1, record epoch1_time. Set remaining_epochs = max(1, floor(40 / epoch1_time)).
- If epoch1_time > 35s, stop after epoch 1 and report results immediately.

## Architecture (unchanged)
- Input: 32x32 RGB (CIFAR-10, 10 classes)
- Conv block 1: Conv2d(3→32, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 2: Conv2d(32→64, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Flatten → Linear(64*8*8 → 128) → ReLU → Dropout(0.3) → Linear(128 → 10)

## Training
- Optimizer: SGD, lr=0.05, momentum=0.9, weight_decay=1e-4
- Scheduler: CosineAnnealingLR(T_max = total_epochs)
- Epochs: dynamic (see time budget)
- **Batch size: 512** (key mutation — fewer steps per epoch, faster wall-clock)
- Loss: CrossEntropyLoss

## Augmentation
- RandomHorizontalFlip
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)

---