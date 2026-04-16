# Crossover — Gen 0 Architecture + Cosine Annealing + AdamW

**Rationale:** Keep the proven 3-block conv architecture but swap training strategy: replace SGD+StepLR with AdamW+CosineAnnealingLR. AdamW converges faster on short schedules; cosine annealing provides a smooth decay without needing to tune step_size. Batch 256 keeps timing safe.

## Architecture
- Input: 32x32 RGB (CIFAR-10, 10 classes)
- Conv block 1: Conv2d(3→32, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 2: Conv2d(32→64, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 3: Conv2d(64→128, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Flatten → Linear(2048→256) → ReLU → Dropout(0.5) → Linear(256→10)

## Training
- Optimizer: AdamW, lr=3e-3, weight_decay=1e-4  ← **changed**
- Scheduler: CosineAnnealingLR(T_max=epochs)  ← **changed**
- Epochs: 3
- Batch size: 256  ← **changed** (safety margin for time budget)
- Loss: CrossEntropyLoss

## Augmentation
- RandomHorizontalFlip
- RandomCrop(32, padding=4)
- Normalize mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616)

---