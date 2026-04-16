# Novel Exploration — Residual Blocks + Global Average Pooling

Rationale: Replace flat conv stack with three residual blocks (skip connections stabilize training and allow higher lr). Use Global Average Pooling instead of a large FC layer — this cuts ~500K params, leaving room for wider channels (32→64→128) and produces a regularization effect similar to Dropout. Result: ~290K params, faster forward pass, and better gradient flow. Train with OneCycleLR for aggressive warm-up in limited epochs.

## Architecture
```
Input (3, 32, 32)
│
ResBlock1: [Conv(3→32,3x3,pad=1)→BN→ReLU→Conv(32→32,3x3,pad=1)→BN]
           + shortcut Conv(3→32,1x1) → ReLU → MaxPool(2)   [16x16]
│
ResBlock2: [Conv(32→64,3x3,pad=1)→BN→ReLU→Conv(64→64,3x3,pad=1)→BN]
           + shortcut Conv(32→64,1x1) → ReLU → MaxPool(2)  [8x8]
│
ResBlock3: [Conv(64→128,3x3,pad=1)→BN→ReLU→Conv(128→128,3x3,pad=1)→BN]
           + shortcut Conv(64→128,1x1) → ReLU → MaxPool(2) [4x4]
│
GlobalAvgPool → (128,)
│
Dropout(0.3) → Linear(128→10)
```
- Estimated params: ~290K (well under 1M)

## Training
- Optimizer: SGD, lr=0.1 (peak), momentum=0.9, weight_decay=1e-4, nesterov=True
- Scheduler: OneCycleLR(max_lr=0.1, pct_start=0.3, epochs=3, steps_per_epoch=len(train_loader))
- Epochs: 3
- Batch size: 256
- Loss: CrossEntropyLoss(label_smoothing=0.1)

## Augmentation
- RandomHorizontalFlip
- RandomCrop(32, padding=4)
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)