# Architectural Mutation — Wider Channels + Deeper FC

Rationale: Double the channel count in block 1 (3→64) and keep subsequent widths at 128/128. Wider early filters capture more low-level features. Replace first FC layer with a narrower bottleneck (512→256) to stay under 1M params. Remove one MaxPool level to preserve spatial resolution for deeper features.

## Architecture
- Input: 32x32 RGB (CIFAR-10, 10 classes)
- Conv block 1: Conv2d(3→64, 3x3, pad=1) → BN → ReLU → MaxPool(2)  [16x16]
- Conv block 2: Conv2d(64→128, 3x3, pad=1) → BN → ReLU → MaxPool(2) [8x8]
- Conv block 3: Conv2d(128→128, 3x3, pad=1) → BN → ReLU → MaxPool(2) [4x4]
- Flatten → Linear(128*4*4=2048→512) → ReLU → Dropout(0.4) → Linear(512→10)
- Estimated params: ~820K (under 1M)

## Training
- Optimizer: SGD, lr=0.01, momentum=0.9, weight_decay=1e-4
- Scheduler: StepLR(step_size=5, gamma=0.5)
- Epochs: 2
- Batch size: 256
- Loss: CrossEntropyLoss

## Augmentation
- RandomHorizontalFlip
- RandomCrop(32, padding=4)
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)

---