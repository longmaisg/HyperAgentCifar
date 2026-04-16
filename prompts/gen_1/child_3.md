# Crossover — Seed Architecture + Adam + Cosine Annealing

## Crossover
Top-1 architecture kept intact; training strategy replaced with Adam optimizer and cosine LR schedule

## Architecture
- Input: 32x32 RGB (CIFAR-10, 10 classes)
- Conv block 1: Conv2d(3→32, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 2: Conv2d(32→64, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 3: Conv2d(64→128, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Flatten → Linear(128*4*4 → 256) → ReLU → Dropout(0.5) → Linear(256 → 10)

## Training
- Optimizer: Adam, lr=3e-4, weight_decay=1e-4
- Scheduler: CosineAnnealingLR(T_max=10, eta_min=1e-6)
- Epochs: 10
- Batch size: 128
- Loss: CrossEntropyLoss

## Augmentation
- RandomHorizontalFlip
- RandomCrop(32, padding=4)
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)

---