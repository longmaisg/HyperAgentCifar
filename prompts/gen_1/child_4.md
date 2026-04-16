# Explorer — Lightweight ResNet with Global Average Pooling

## Novel Elements
Residual connections for gradient flow; Global Average Pooling eliminates large FC layer; ~308k params; label smoothing regularization

## Architecture
- Input: 32x32 RGB (CIFAR-10, 10 classes)
- Stem: Conv2d(3→32, 3x3, pad=1) → BN → ReLU
- ResBlock 1: [Conv2d(32→32, 3x3, pad=1) → BN → ReLU → Conv2d(32→32, 3x3, pad=1) → BN] + identity → ReLU
- ResBlock 2: [Conv2d(32→64, 3x3, stride=2, pad=1) → BN → ReLU → Conv2d(64→64, 3x3, pad=1) → BN] + shortcut(Conv2d(32→64,1x1,stride=2)) → ReLU
- ResBlock 3: [Conv2d(64→128, 3x3, stride=2, pad=1) → BN → ReLU → Conv2d(128→128, 3x3, pad=1) → BN] + shortcut(Conv2d(64→128,1x1,stride=2)) → ReLU
- AdaptiveAvgPool2d(1) → Flatten → Linear(128 → 10)

## Training
- Optimizer: SGD, lr=0.05, momentum=0.9, weight_decay=1e-4, nesterov=True
- Scheduler: CosineAnnealingLR(T_max=10, eta_min=1e-5)
- Epochs: 10
- Batch size: 128
- Loss: CrossEntropyLoss(label_smoothing=0.1)

## Augmentation
- RandomHorizontalFlip
- RandomCrop(32, padding=4)
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)