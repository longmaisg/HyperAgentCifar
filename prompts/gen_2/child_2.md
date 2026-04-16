# Mutant — Wider Channels (top-2 base + architectural width increase)

## Mutation
Base: Child 1 (lr=0.05, dropout=0.3). Architectural change: increase channel widths from 32→64→128 to 48→96→192, keep FC at 256. This widens all conv layers, increasing representational capacity while staying under 1M params.

## Rationale
Child 1's higher lr may benefit from a wider network that has more gradient signal to propagate. Wider channels also improve feature diversity without adding depth (which risks vanishing gradients at low epoch counts).

## Architecture
- Input: 32x32 RGB (CIFAR-10, 10 classes)
- Conv block 1: Conv2d(3→48, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 2: Conv2d(48→96, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 3: Conv2d(96→192, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Flatten → Linear(192*4*4 → 256) → ReLU → Dropout(0.3) → Linear(256 → 10)

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

## Estimated params
- Conv layers: 3*48*9 + 48*96*9 + 96*192*9 ≈ 1,296 + 41,472 + 165,888 = 208,656
- BN layers: ~672
- FC: 192*16*256 + 256*10 = 786,432 + 2,560 = 788,992
- Total: ~998,320 (just under 1M)

---