# Mutation — Narrower Channels (32→64→96) to Reduce Compute + OneCycleLR

## Rationale
Gen 3 Child 2 (top-2) used OneCycleLR with batch=256 but was still killed at ~86 s/epoch. The third conv block (64→128) dominates compute: its 128 output channels create a 4×4×128=2048-dim flatten feeding a large FC. Shrinking that block to 96 channels cuts the final feature map to 4×4×96=1536 and reduces the most expensive conv op by 25 %. Expected epoch saving: ~8–12 s, bringing 2-epoch total under 175 s. OneCycleLR with max_lr=0.1 and batch=256 are retained from top-2 since they drove faster convergence per step.

## Architecture
- Input: 32×32 RGB (CIFAR-10, 10 classes)
- Conv block 1: Conv2d(3→32, 3×3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 2: Conv2d(32→64, 3×3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 3: Conv2d(64→**96**, 3×3, pad=1) → BN → ReLU → MaxPool(2)
- Flatten → Linear(**1536**→256) → ReLU → Dropout(0.5) → Linear(256→10)

## Parameter count (estimated)
~468 K — well under 1 M limit.

## Training
- Optimizer: SGD, lr=0.01 (initial), momentum=0.9, weight_decay=1e-4
- Scheduler: OneCycleLR(max_lr=0.1, epochs=2, steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy='cos')
- Epochs: 2
- Batch size: 256
- Loss: CrossEntropyLoss

## Augmentation
- RandomHorizontalFlip
- RandomCrop(32, padding=4)
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)

---