import cv2
import numpy as np

def _ensure_sift():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=0, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6), False
    if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "SIFT_create"):
        return cv2.xfeatures2d.SIFT_create(nfeatures=0, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6), False
    return cv2.ORB_create(nfeatures=1000), True

class SIFTExtractor:
    def __init__(self, rootsift: bool = True):
        self.det, self.is_orb = _ensure_sift()
        self.rootsift = bool(rootsift)

    def extract(self, img_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        _, desc = self.det.detectAndCompute(gray, None)
        if desc is None:
            return np.zeros((0, 128 if not self.is_orb else 32), np.float32)
        desc = desc.astype(np.float32)
        if not self.is_orb and self.rootsift:
            desc /= (np.sum(desc, axis=1, keepdims=True) + 1e-12)  # L1
            desc = np.sqrt(desc, dtype=np.float32)                 # RootSIFT
        return desc
