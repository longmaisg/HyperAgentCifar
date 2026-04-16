# Mutant — Higher LR + Reduced Dropout

## Mutation
Single hyperparameter change: lr 0.01 → 0.05, dropout 0.5 → 0.3

## Architecture
- Input: 32x32 RGB (CIFAR-10, 10 classes)
- Conv block 1: Conv2d(3→32, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 2: Conv2d(32→64, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 3: Conv2d(64→128, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Flatten → Linear(128*4*4 → 256) → ReLU → Dropout(0.3) → Linear(256 → 10)

## Training
- Optimizer: SGD, lr=0.05, momentum=0.9, weight_decay=1e-4
- Scheduler: StepLR(step_size=5, gamma=0.5)
- Epochs: 10
- Batch size: 128
- Loss: CrossEntropyLoss

## Augmentation
- RandomHorizontalFlip
- RandomCrop(32, padding=4)
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)

---