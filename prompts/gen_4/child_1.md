# Mutation — AMP Mixed Precision to Break the Time Budget

## Rationale
Every Gen 3 run was killed at ~182 s despite epoch compute finishing in ~85 s each. The single highest-leverage lever that touches nothing else is `torch.cuda.amp` (autocast + GradScaler). On a CUDA GPU, FP16 arithmetic roughly halves tensor-core latency for conv layers; in practice this yields 30–50 % wall-clock reduction per epoch, comfortably absorbing the ~5–10 s validation/I/O overhead. All hyperparameters are held identical to the Gen 3 top-1 so any accuracy change is attributable solely to AMP numerics (empirically negligible for this architecture).

## Architecture
- Input: 32×32 RGB (CIFAR-10, 10 classes)
- Conv block 1: Conv2d(3→32, 3×3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 2: Conv2d(32→64, 3×3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 3: Conv2d(64→128, 3×3, pad=1) → BN → ReLU → MaxPool(2)
- Flatten → Linear(2048→256) → ReLU → Dropout(0.5) → Linear(256→10)

## Training
- Optimizer: SGD, lr=0.01, momentum=0.9, weight_decay=1e-4
- Scheduler: StepLR(step_size=5, gamma=0.5)
- Epochs: 2
- Batch size: 128
- Loss: CrossEntropyLoss
- **Mixed precision: `torch.cuda.amp.autocast()` around forward+loss; `torch.cuda.amp.GradScaler` wrapping optimizer step**

## Augmentation
- RandomHorizontalFlip
- RandomCrop(32, padding=4)
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)

## Implementation note
```python
scaler = torch.cuda.amp.GradScaler()
for x, y in loader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        loss = criterion(model(x), y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---