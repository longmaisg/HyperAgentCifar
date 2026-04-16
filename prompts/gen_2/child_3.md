# Crossover — Top-1 Architecture + Adam + CosineAnnealingLR + ColorJitter

## Rationale
Top-1 (Child 0) architecture with top-3 (Child 3 Gen 1) training strategy, but fixed to 2 epochs to avoid the timeout. Adam converges faster than SGD in the first few epochs; CosineAnnealingLR provides smooth decay over the short run. ColorJitter adds cheap regularisation that improves generalisation. Batch size drops to 128 (vs. Child 3 Gen 1's 256) because the batch size change did not help wall-clock time in Gen 1; keeping it at 128 avoids the generalisation penalty of large batches in just 2 epochs.

## Architecture
- Input: 32×32 RGB (CIFAR-10, 10 classes)
- Conv block 1: Conv2d(3→32, 3×3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 2: Conv2d(32→64, 3×3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 3: Conv2d(64→128, 3×3, pad=1) → BN → ReLU → MaxPool(2)
- Flatten → Linear(2048→256) → ReLU → Dropout(0.4) → Linear(256→10)

## Training
- Optimizer: Adam, lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4
- Scheduler: CosineAnnealingLR(T_max=2, eta_min=1e-5)
- Epochs: 2
- Batch size: 128
- Loss: CrossEntropyLoss

## Augmentation
- RandomHorizontalFlip
- RandomCrop(32, padding=4)
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)

---