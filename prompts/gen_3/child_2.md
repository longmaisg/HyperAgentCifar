# Mutation — OneCycleLR with Aggressive Peak LR + Batch 256

## Rationale
Child 1 Gen 2 used OneCycleLR(max_lr=0.05) but still got killed, partly because batch=128 kept epoch time at ~86 s. Switching to batch=256 shaves ~15–20 s per epoch. Simultaneously raising max_lr to 0.1 (SGD with momentum can tolerate this under OneCycleLR's cosine shape) extracts more signal from each step. The 30 % warmup fraction is retained; the aggressive peak compensates for fewer steps per epoch.

## Architecture
- Input: 32×32 RGB (CIFAR-10, 10 classes)
- Conv block 1: Conv2d(3→32, 3×3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 2: Conv2d(32→64, 3×3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 3: Conv2d(64→128, 3×3, pad=1) → BN → ReLU → MaxPool(2)
- Flatten → Linear(2048→256) → ReLU → Dropout(0.5) → Linear(256→10)

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