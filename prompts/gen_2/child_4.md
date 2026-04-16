# Novel — Residual Connections (ResNet-Lite) + OneCycleLR

## Rationale
Skip connections allow gradients to flow cleanly through the network, reducing the accuracy penalty of a short training run. A lightweight ResNet-lite with three residual blocks and AdaptiveAvgPool costs only ~308 K parameters — well under the 1 M limit — and avoids the large flat FC layer that dominated the Gen 1 architecture. OneCycleLR (max_lr=0.1) drives aggressive learning in epoch 1 then decays smoothly, which pairs well with residual nets. The stem + three residual blocks keep per-epoch time comparable to Gen 1 (~80–85 s), so 2 epochs comfortably fits under 175 s.

## Architecture
- Stem: Conv2d(3→32, 3×3, pad=1) → BN → ReLU → MaxPool(2) → 16×16
- ResBlock 1 (32→32, stride=1): [Conv(32→32,3×3,p=1)→BN→ReLU→Conv(32→32,3×3,p=1)→BN] + identity → ReLU; then MaxPool(2) → 8×8
- ResBlock 2 (32→64, projection): [Conv(32→64,3×3,p=1)→BN→ReLU→Conv(64→64,3×3,p=1)→BN] + Conv1×1(32→64); → ReLU; then MaxPool(2) → 4×4
- ResBlock 3 (64→128, projection): [Conv(64→128,3×3,p=1)→BN→ReLU→Conv(128→128,3×3,p=1)→BN] + Conv1×1(64→128); → ReLU
- AdaptiveAvgPool(1) → Flatten → Linear(128→10)
- Parameters: ~308 K

## Training
- Optimizer: SGD, lr=0.01 (initial), momentum=0.9, weight_decay=1e-4
- Scheduler: OneCycleLR(max_lr=0.1, epochs=2, steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy='cos')
- Epochs: 2
- Batch size: 128
- Loss: CrossEntropyLoss

## Augmentation
- RandomHorizontalFlip
- RandomCrop(32, padding=4)
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)