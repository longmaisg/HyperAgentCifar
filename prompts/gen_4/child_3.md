# Crossover — Top-1 Architecture + Top-2 Scheduler + AMP

## Rationale
Top-1 (standard 32→64→128, StepLR) holds the best raw accuracy but is perpetually killed. Top-2 (OneCycleLR, batch=256) converges faster per epoch but was also killed. Crossing their strengths: keep the full-width architecture for accuracy capacity, adopt OneCycleLR + batch=256 for faster convergence per step, and add AMP to slash per-epoch wall time. Together these three changes should deliver top-1-level accuracy while fitting within the 175 s budget.

## Architecture
- Input: 32×32 RGB (CIFAR-10, 10 classes)
- Conv block 1: Conv2d(3→32, 3×3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 2: Conv2d(32→64, 3×3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 3: Conv2d(64→128, 3×3, pad=1) → BN → ReLU → MaxPool(2)
- Flatten → Linear(2048→256) → ReLU → Dropout(0.5) → Linear(256→10)

## Training
- Optimizer: SGD, lr=0.01 (initial), momentum=0.9, weight_decay=1e-4
- Scheduler: OneCycleLR(max_lr=0.1, epochs=2, steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy='cos')
- Epochs: 2
- Batch size: 256
- Loss: CrossEntropyLoss
- **Mixed precision: `torch.cuda.amp.autocast()` + `GradScaler`** (same pattern as Child 1)

## Augmentation
- RandomHorizontalFlip
- RandomCrop(32, padding=4)
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)

---