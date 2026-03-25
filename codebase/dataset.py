import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class ImageItem:
    img_id: str
    path: str
    category: str

class ImageDataset:
    def __init__(self, root_dir: str = "../database"):
        self.root_dir = root_dir
        self.items: List[ImageItem] = self._scan()

    def _scan(self) -> List[ImageItem]:
        items = []
        for class_name in sorted(os.listdir(self.root_dir)):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            exts = (".jpg", ".jpeg", ".png")
            all_files = sorted([
                f for f in os.listdir(class_dir)
                if f.lower().endswith(exts)
            ])

            for fname in all_files:
                img_id = os.path.splitext(fname)[0]
                path = os.path.join(class_dir, fname)
                items.append(ImageItem(img_id, path, class_name))

        print(f"[INFO] Loaded {len(items)} images from {len(set(i.category for i in items))} classes.")
        return items

    def __len__(self):
        return len(self.items)

    def get(self, index: int) -> ImageItem:
        return self.items[index]

    def load_image(self, index: int) -> np.ndarray:
        """Load BGR image with OpenCV."""
        item = self.items[index]
        img = cv2.imread(item.path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Cannot read image: {item.path}")
        return img
