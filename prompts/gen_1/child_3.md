# Crossover — Top-1 Architecture + Adam + Cosine Annealing

Rationale: Keep the proven 3-block conv architecture but swap the training strategy entirely. Adam with a low lr converges more reliably in few epochs than SGD on CIFAR-10. CosineAnnealingLR provides smooth decay that avoids the step-jump artifacts of StepLR, extracting more accuracy from 2 epochs.

## Architecture
- Input: 32x32 RGB (CIFAR-10, 10 classes)
- Conv block 1: Conv2d(3→32, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 2: Conv2d(32→64, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 3: Conv2d(64→128, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Flatten → Linear(2048→256) → ReLU → Dropout(0.4) → Linear(256→10)

## Training
- Optimizer: Adam, lr=0.001, weight_decay=1e-4, betas=(0.9, 0.999)
- Scheduler: CosineAnnealingLR(T_max=epochs, eta_min=1e-5)
- Epochs: 3
- Batch size: 256
- Loss: CrossEntropyLoss

## Augmentation
- RandomHorizontalFlip
- RandomCrop(32, padding=4)
- ColorJitter(brightness=0.2, contrast=0.2)
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)

---