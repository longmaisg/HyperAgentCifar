# Crossover — Seed Architecture + Adam + Fixed Short Schedule

**Strategy:** Keep the seed's 2-block CNN but replace SGD+cosine with Adam and a fixed 3-epoch plan so there is no dynamic scheduling logic that could misfire.

## Time Budget
- Hard wall-clock limit: 60 seconds.
- Run exactly 3 epochs. If epoch 1 takes >18s, drop to 2 epochs.
- No dynamic calculation — fixed epoch count avoids scheduling bugs.

## Architecture (same as seed)
- Input: 32x32 RGB (CIFAR-10, 10 classes)
- Conv block 1: Conv2d(3→32, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 2: Conv2d(32→64, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Flatten → Linear(64*8*8 → 128) → ReLU → Dropout(0.3) → Linear(128 → 10)

## Training (mutated strategy)
- **Optimizer: Adam, lr=3e-3, weight_decay=1e-4**
- **Scheduler: OneCycleLR(max_lr=3e-3, steps_per_epoch=len(trainloader), epochs=3)**
- **Epochs: 3 (fixed)**
- Batch size: 512
- Loss: CrossEntropyLoss

## Augmentation
- RandomHorizontalFlip
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)

---