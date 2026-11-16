import torch
import torch.nn as nn
from siamese_model import SiameseResNet18

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")

device = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0))

model = SiameseResNet18(pretrained=True).to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler("cuda")

batch_size = 16
pre = torch.randn(batch_size, 3, 128, 128, device=device)
post = torch.randn(batch_size, 3, 128, 128, device=device)
labels = torch.randint(0, 4, (batch_size,), device=device)

optimizer.zero_grad(set_to_none=True)

with torch.amp.autocast("cuda"):
    outputs = model(pre, post)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

print("Batch size:", batch_size)
print("Output shape:", tuple(outputs.shape))
print("Loss:", round(loss.item(), 4))
print("Peak GPU memory:", round(
    torch.cuda.max_memory_allocated() / 1024**3, 2
), "GB")
print("PRETRAINED GPU TRAINING TEST PASSED")
