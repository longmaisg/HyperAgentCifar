# Mutation — Batch Size 256 + LR Scaling to Cut Epoch Wall-Clock Time

## Rationale
All Gen 2 runs exceeded 175 s despite training completing in ~166 s; overhead (validation, I/O) pushed total past the limit. The single fastest lever is halving the number of optimizer steps per epoch by doubling batch size to 256. LR is scaled linearly (0.01 → 0.02) to maintain effective gradient signal per parameter update. Everything else is identical to the Gen 2 best so accuracy regression is minimal and the time saving should absorb the overhead.

## Architecture
- Input: 32×32 RGB (CIFAR-10, 10 classes)
- Conv block 1: Conv2d(3→32, 3×3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 2: Conv2d(32→64, 3×3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 3: Conv2d(64→128, 3×3, pad=1) → BN → ReLU → MaxPool(2)
- Flatten → Linear(2048→256) → ReLU → Dropout(0.5) → Linear(256→10)

## Training
- Optimizer: SGD, lr=0.02, momentum=0.9, weight_decay=1e-4
- Scheduler: StepLR(step_size=5, gamma=0.5)
- Epochs: 2
- Batch size: 256
- Loss: CrossEntropyLoss

## Augmentation
- RandomHorizontalFlip
- RandomCrop(32, padding=4)
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)

---