from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from util.crop import center_crop_arr


def pil_to_tensor(image):
    array = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(torch.float32).div_(255.0)
    return array * 2.0 - 1.0


class PairedImageDataset(Dataset):
    def __init__(self, root, img_size):
        self.root = Path(root)
        self.lq_dir = self.root / "lq"
        self.hq_dir = self.root / "hq"
        if not self.lq_dir.is_dir() or not self.hq_dir.is_dir():
            raise FileNotFoundError("Expected paired dataset folders: <root>/lq and <root>/hq")

        self.samples = []
        for hq_path in sorted(self.hq_dir.rglob("*")):
            if not hq_path.is_file():
                continue
            rel_path = hq_path.relative_to(self.hq_dir)
            lq_path = self.lq_dir / rel_path
            if lq_path.is_file():
                self.samples.append((lq_path, hq_path))
        if not self.samples:
            raise RuntimeError(f"No paired files found under {self.root}")
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        image = Image.open(path).convert("RGB")
        image = center_crop_arr(image, self.img_size)
        return pil_to_tensor(image)

    def __getitem__(self, index):
        lq_path, hq_path = self.samples[index]
        lq = self._load_image(lq_path)
        hq = self._load_image(hq_path)
        return lq, hq
