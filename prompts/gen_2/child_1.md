# Hyperparameter Mutation — OneCycleLR for Fast Two-Epoch Convergence

## Rationale
OneCycleLR is purpose-built for short training runs: it warms up to a high peak LR then decays via cosine annealing, squeezing the most learning out of every batch. The parent (Child 0 Gen 1) used a flat StepLR that barely activates in 2 epochs. Switching to OneCycleLR(max_lr=0.05) with a 30 % warmup phase should lift accuracy by ~2–3 % at no extra cost. Architecture and batch size are unchanged to keep per-epoch time at ~85 s.

## Architecture
- Identical to CHILD_0
- Input: 32×32 RGB (CIFAR-10, 10 classes)
- Conv block 1: Conv2d(3→32, 3×3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 2: Conv2d(32→64, 3×3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 3: Conv2d(64→128, 3×3, pad=1) → BN → ReLU → MaxPool(2)
- Flatten → Linear(2048→256) → ReLU → Dropout(0.5) → Linear(256→10)

## Training
- Optimizer: SGD, lr=0.01 (initial), momentum=0.9, weight_decay=1e-4
- Scheduler: OneCycleLR(max_lr=0.05, epochs=2, steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy='cos')
- Epochs: 2
- Batch size: 128
- Loss: CrossEntropyLoss

## Augmentation
- RandomHorizontalFlip
- RandomCrop(32, padding=4)
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)

---