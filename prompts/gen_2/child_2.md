# Mutation — Wider Middle Block on Second-Best Architecture

**Change:** Take the generation-1 Child 0 base (32→64 conv) and reduce to 32→48 to trade some capacity for speed. Also drop FC from 128→96 to stay fast. Goal: faster epoch1 than Child 0 (45.6s) while keeping more width than Child 2.

## Time Budget
- Hard wall-clock limit: 60 seconds.
- After epoch 1, set remaining_epochs = max(1, floor(45 / epoch1_time)).
- If epoch1_time > 40s, stop immediately after epoch 1.

## Architecture
- Input: 32x32 RGB (CIFAR-10, 10 classes)
- Conv block 1: Conv2d(3→32, 3x3, pad=1) → BN → ReLU → MaxPool(2) → 16x16
- **Conv block 2: Conv2d(32→48, 3x3, pad=1) → BN → ReLU → MaxPool(2) → 8x8**
- Flatten → **Linear(48*8*8 → 96)** → ReLU → Dropout(0.25) → Linear(96 → 10)
- ~220k parameters (well under budget)

## Training
- Optimizer: SGD, lr=0.05, momentum=0.9, weight_decay=1e-4
- Scheduler: CosineAnnealingLR(T_max=total_epochs)
- Epochs: dynamic
- Batch size: 256
- Loss: CrossEntropyLoss

## Augmentation
- RandomHorizontalFlip
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)

---