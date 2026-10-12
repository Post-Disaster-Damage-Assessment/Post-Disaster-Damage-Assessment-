"""python infer.py --pre a_pre_disaster.png --post a_post_disaster.png --ckpt best.pt"""
import argparse, cv2, numpy as np, torch
from dataset import MEAN, STD, CLASS_NAMES
from model import SiameseUNet

COLORS = np.array([[0, 0, 0], [0, 200, 0], [255, 220, 0], [255, 130, 0], [220, 0, 0]], np.uint8)

def load(p):
    im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    t = ((im.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)
    return im, torch.from_numpy(t)[None]

ap = argparse.ArgumentParser()
ap.add_argument("--pre"); ap.add_argument("--post")
ap.add_argument("--ckpt", default="best.pt"); ap.add_argument("--out", default="damage_overlay.png")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = SiameseUNet(pretrained=False).to(dev)
model.load_state_dict(torch.load(a.ckpt, map_location=dev)); model.eval()

_, pre = load(a.pre); post_img, post = load(a.post)
with torch.no_grad():
    prob = torch.softmax(model(pre.to(dev), post.to(dev)), 1)
    # flip TTA
    prob += torch.softmax(model(pre.flip(-1).to(dev), post.flip(-1).to(dev)), 1).flip(-1)
pred = prob.argmax(1)[0].cpu().numpy()

overlay = post_img.copy(); m = pred > 0
overlay[m] = (0.4 * post_img[m] + 0.6 * COLORS[pred[m]]).astype(np.uint8)
cv2.imwrite(a.out, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
n = max(m.sum(), 1)
for c in range(1, 5):
    print(f"{CLASS_NAMES[c]:>10}: {(pred == c).sum() / n * 100:5.1f}% of building area")
print("saved", a.out)
