# Explorer — Mini-ResNet with Global Average Pooling + Cosine Annealing + Label Smoothing

## Novel strategy
Three simultaneous innovations: (1) residual skip connections for gradient flow, (2) global average pooling replacing the large FC bottleneck, (3) cosine annealing LR schedule + label smoothing (ε=0.1) for regularization. This is architecturally distinct from all prior children.

## Rationale
Residual connections address vanishing gradients in deeper networks and are well-proven on CIFAR-10. GAP dramatically cuts parameter count (no 2048-dim flatten), allowing us to spend the budget on conv depth instead. Label smoothing prevents overconfident predictions in only 10 epochs.

## Architecture
- Input: 32x32 RGB (CIFAR-10, 10 classes)
- Stem: Conv2d(3→32, 3x3, pad=1) → BN → ReLU  [32x32]
- Res block 1: [Conv2d(32→32, 3x3, pad=1) → BN → ReLU → Conv2d(32→32, 3x3, pad=1) → BN] + skip → ReLU → MaxPool(2)  [16x16]
- Res block 2: [Conv2d(32→64, 3x3, pad=1) → BN → ReLU → Conv2d(64→64, 3x3, pad=1) → BN] + skip(1x1 proj 32→64) → ReLU → MaxPool(2)  [8x8]
- Res block 3: [Conv2d(64→128, 3x3, pad=1) → BN → ReLU → Conv2d(128→128, 3x3, pad=1) → BN] + skip(1x1 proj 64→128) → ReLU → MaxPool(2)  [4x4]
- GlobalAvgPool(4x4 → 1x1) → Flatten(128) → Linear(128 → 10)

## Training
- Optimizer: SGD, lr=0.1, momentum=0.9, weight_decay=1e-4
- Scheduler: CosineAnnealingLR(T_max=10, eta_min=1e-4)
- Epochs: 10
- Batch size: 128
- Loss: CrossEntropyLoss with label_smoothing=0.1

## Augmentation
- RandomHorizontalFlip
- RandomCrop(32, padding=4)
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)

## Estimated params
- Stem: 3*32*9 = 864
- Res block 1: 32*32*9*2 = 18,432 + BN
- Res block 2: 32*64*9 + 64*64*9 + 1x1(32→64) = 18,432 + 36,864 + 2,048 = 57,344
- Res block 3: 64*128*9 + 128*128*9 + 1x1(64→128) = 73,728 + 147,456 + 8,192 = 229,376
- GAP + FC: 128*10 = 1,280
- BN overhead: ~2,000
- Total: ~309,296 (well under budget, leaving room to scale up in future generations)