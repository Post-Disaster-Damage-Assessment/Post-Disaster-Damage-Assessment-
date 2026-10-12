"""xView2 dataset: rasterises building polygons from the label JSONs.

Mask classes (post-disaster): 0 background, 1 no-damage, 2 minor, 3 major,
4 destroyed, 255 = un-classified (ignored in loss/metrics).
"""
import json, random
from pathlib import Path
import cv2, numpy as np, torch
from shapely import wkt
from torch.utils.data import Dataset

DAMAGE = {"no-damage": 1, "minor-damage": 2, "major-damage": 3,
          "destroyed": 4, "un-classified": 255}
CLASS_NAMES = ["background", "no-damage", "minor", "major", "destroyed"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def rasterize(json_path, h, w):
    with open(json_path) as f:
        feats = json.load(f)["features"]["xy"]
    mask = np.zeros((h, w), np.uint8)
    for ft in feats:
        cls = DAMAGE.get(ft["properties"].get("subtype", "no-damage"), 1)
        geom = wkt.loads(ft["wkt"])
        for p in (geom.geoms if hasattr(geom, "geoms") else [geom]):
            if p.is_empty:
                continue
            cv2.fillPoly(mask, [np.round(np.array(p.exterior.coords)).astype(np.int32)], int(cls))
    return mask


def find_samples(root):
    root = Path(root)
    out = []
    for pre in sorted((root / "images").glob("*_pre_disaster.png")):
        post = pre.with_name(pre.name.replace("_pre_", "_post_"))
        lab = root / "labels" / post.with_suffix(".json").name
        if post.exists() and lab.exists():
            out.append((pre, post, lab))
    return out


def split_samples(samples, val_frac=0.1, seed=0):
    s = samples[:]
    random.Random(seed).shuffle(s)
    n = int(len(s) * val_frac)
    return s[n:], s[:n]


class XView2(Dataset):
    def __init__(self, samples, train=True, crop=512):
        self.samples, self.train, self.crop = samples, train, crop

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        pre_p, post_p, lab_p = self.samples[i]
        pre = cv2.cvtColor(cv2.imread(str(pre_p)), cv2.COLOR_BGR2RGB)
        post = cv2.cvtColor(cv2.imread(str(post_p)), cv2.COLOR_BGR2RGB)
        h, w = post.shape[:2]
        mask = rasterize(lab_p, h, w)
        if self.train:
            c = self.crop
            y, x = random.randint(0, h - c), random.randint(0, w - c)
            pre, post, mask = pre[y:y+c, x:x+c], post[y:y+c, x:x+c], mask[y:y+c, x:x+c]
            if random.random() < .5: pre, post, mask = pre[:, ::-1], post[:, ::-1], mask[:, ::-1]
            if random.random() < .5: pre, post, mask = pre[::-1], post[::-1], mask[::-1]
            k = random.randint(0, 3)
            pre, post, mask = np.rot90(pre, k), np.rot90(post, k), np.rot90(mask, k)
        def prep(a):
            a = (np.ascontiguousarray(a).astype(np.float32) / 255 - MEAN) / STD
            return torch.from_numpy(a.transpose(2, 0, 1))
        return prep(pre), prep(post), torch.from_numpy(np.ascontiguousarray(mask)).long()
