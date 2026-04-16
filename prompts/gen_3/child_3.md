# Crossover — Gen 2 Best Architecture + OneCycleLR + Batch 256 + Label Smoothing

## Rationale
Combines the strongest elements seen so far: the Gen 2 Child 0 3-conv architecture (highest accuracy) with OneCycleLR's fast-convergence schedule (Child 1 Gen 2). Batch size is raised to 256 for speed. Label smoothing (ε=0.1) is added as a zero-cost regulariser that tends to improve calibration and generalisation in short training runs without adding any compute. Dropout stays at 0.5.

## Architecture
- Input: 32×32 RGB (CIFAR-10, 10 classes)
- Conv block 1: Conv2d(3→32, 3×3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 2: Conv2d(32→64, 3×3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 3: Conv2d(64→128, 3×3, pad=1) → BN → ReLU → MaxPool(2)
- Flatten → Linear(2048→256) → ReLU → Dropout(0.5) → Linear(256→10)

## Training
- Optimizer: SGD, lr=0.01 (initial), momentum=0.9, weight_decay=1e-4
- Scheduler: OneCycleLR(max_lr=0.05, epochs=2, steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy='cos')
- Epochs: 2
- Batch size: 256
- Loss: CrossEntropyLoss with label_smoothing=0.1

## Augmentation
- RandomHorizontalFlip
- RandomCrop(32, padding=4)
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)

---