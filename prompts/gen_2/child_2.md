# Architectural Mutation — Wider Channels + GlobalAvgPool (No Large FC)

## Rationale
Top-2 (Child 1 Gen 1) used the same architecture as top-1 but with a higher LR. Instead, mutate the architecture: double channel widths (32→64→128→256) for more representational power, then replace the expensive Flatten + Linear(2048→256) block with AdaptiveAvgPool(1). This removes 524 K parameters from the FC layer, dropping total params to ~374 K while making the spatial features richer. Keep the Gen 1 top-2 training settings (SGD, lr=0.05, Nesterov) but cap at 2 epochs to stay inside the time budget.

## Architecture
- Input: 32×32 RGB (CIFAR-10, 10 classes)
- Conv block 1: Conv2d(3→64, 3×3, pad=1) → BN → ReLU → MaxPool(2) → 16×16
- Conv block 2: Conv2d(64→128, 3×3, pad=1) → BN → ReLU → MaxPool(2) → 8×8
- Conv block 3: Conv2d(128→256, 3×3, pad=1) → BN → ReLU → AdaptiveAvgPool(1) → 1×1
- Flatten → Linear(256→10)
- Parameters: ~374 K

## Training
- Optimizer: SGD, lr=0.05, momentum=0.9, nesterov=True, weight_decay=1e-4
- Scheduler: StepLR(step_size=3, gamma=0.5)
- Epochs: 2
- Batch size: 256
- Loss: CrossEntropyLoss

## Augmentation
- RandomHorizontalFlip
- RandomCrop(32, padding=4)
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)

---