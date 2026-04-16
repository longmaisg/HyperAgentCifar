# Exploration — Depthwise Separable Convs + Label Smoothing

**Novel idea:** Replace standard convolutions with depthwise-separable blocks (MobileNet style) to cut FLOPs by ~8–9x while keeping receptive field. Add a third conv stage (cheap at this width) and use label smoothing for better generalization per epoch.

## Time Budget
- Hard wall-clock limit: 60 seconds.
- After epoch 1, set remaining_epochs = max(2, floor(50 / epoch1_time)).
- DS convs are ~4x cheaper than standard convs → expect 4+ epochs feasible.

## Architecture
- Input: 32x32 RGB (CIFAR-10, 10 classes)
- **DS block 1:** DepthwiseConv2d(3→3, 3x3, pad=1) + PointwiseConv2d(3→32, 1x1) → BN → ReLU → MaxPool(2)  →16x16
- **DS block 2:** DepthwiseConv2d(32→32, 3x3, pad=1) + PointwiseConv2d(32→64, 1x1) → BN → ReLU → MaxPool(2) → 8x8
- **DS block 3:** DepthwiseConv2d(64→64, 3x3, pad=1) + PointwiseConv2d(64→64, 1x1) → BN → ReLU → MaxPool(2) → 4x4
- AdaptiveAvgPool2d(1) → Flatten → Linear(64 → 10)
- No large FC layer; global average pooling keeps params tiny (~30k total)

## Training
- Optimizer: SGD, lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True
- Scheduler: CosineAnnealingLR(T_max = total_epochs)
- Epochs: dynamic (see time budget)
- Batch size: 512
- **Loss: CrossEntropyLoss with label_smoothing=0.1**

## Augmentation
- RandomHorizontalFlip
- RandomCrop(32, padding=4)
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)