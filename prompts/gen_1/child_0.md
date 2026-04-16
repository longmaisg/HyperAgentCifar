# Elitism — Exact Copy of Seed

## Time Budget
- Hard wall-clock limit: 60 seconds. Process is killed at 60s.
- Target: complete at least 2 epochs within 55 seconds.
- After epoch 1, compute elapsed time and set remaining_epochs = max(1, floor(45 / epoch1_time)).

## Architecture (keep small and fast)
- Input: 32x32 RGB (CIFAR-10, 10 classes)
- Conv block 1: Conv2d(3→32, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 2: Conv2d(32→64, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Flatten → Linear(64*8*8 → 128) → ReLU → Dropout(0.3) → Linear(128 → 10)

## Training
- Optimizer: SGD, lr=0.05, momentum=0.9, weight_decay=1e-4
- Scheduler: CosineAnnealingLR
- Epochs: dynamic (measure epoch 1, then adjust)
- Batch size: 256
- Loss: CrossEntropyLoss

## Augmentation
- RandomHorizontalFlip
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)

---