# Hyperparameter Mutation — Higher LR + Larger Batch

Rationale: The parent was killed at 182s. Increasing batch size to 256 cuts per-epoch time by ~30%; raising lr to 0.05 compensates for fewer gradient steps so accuracy does not regress. Momentum Nesterov improves convergence at higher lr.

## Architecture
- Identical to CHILD_0
- Input: 32x32 RGB (CIFAR-10, 10 classes)
- Conv block 1: Conv2d(3→32, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 2: Conv2d(32→64, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 3: Conv2d(64→128, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Flatten → Linear(2048→256) → ReLU → Dropout(0.5) → Linear(256→10)

## Training
- Optimizer: SGD, lr=0.05, momentum=0.9, nesterov=True, weight_decay=1e-4
- Scheduler: StepLR(step_size=3, gamma=0.5)
- Epochs: 3
- Batch size: 256
- Loss: CrossEntropyLoss

## Augmentation
- RandomHorizontalFlip
- RandomCrop(32, padding=4)
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)

---