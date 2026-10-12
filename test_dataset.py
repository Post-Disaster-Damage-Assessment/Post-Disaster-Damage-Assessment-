from pathlib import Path
from dataset import find_samples, XView2

root = Path("data/train")
samples = find_samples(root)

print("Total paired samples:", len(samples))
print("First sample:", samples[0])

train_ds = XView2(samples[:10], train=True, crop=512)
pre, post, mask = train_ds[0]

print("Pre shape :", pre.shape)
print("Post shape:", post.shape)
print("Mask shape:", mask.shape)
print("Pre dtype :", pre.dtype)
print("Post dtype:", post.dtype)
print("Mask dtype:", mask.dtype)
print("Mask classes:", mask.unique().tolist())
