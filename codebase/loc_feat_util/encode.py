import numpy as np
import cv2

def assign_descriptors(desc: np.ndarray, centers: np.ndarray) -> np.ndarray:
    if desc.size == 0:
        return np.zeros((0,), np.int32)
    a2 = np.sum(desc*desc, axis=1, keepdims=True)         # [M,1]
    b2 = np.sum(centers*centers, axis=1, keepdims=True).T # [1,K]
    ab = desc @ centers.T                                 # [M,K]
    d2 = a2 + b2 - 2.0*ab
    return np.argmin(d2, axis=1).astype(np.int32)

def bow_hist(assign_idx: np.ndarray, K: int) -> np.ndarray:
    h = np.bincount(assign_idx, minlength=K).astype(np.float32)
    s = h.sum()
    if s > 0:
        h /= s
    return h

def idf_from_histograms(H: np.ndarray) -> np.ndarray:
    N, K = H.shape
    df = (H > 0).sum(axis=0).astype(np.float32)
    return (np.log((N+1.0)/(df+1.0)) + 1.0).astype(np.float32)

def bow_pyramid(img_rgb, extractor, centers, idf=None, levels=2):
    K = centers.shape[0]
    H, W, _ = img_rgb.shape
    parts = []
    for L in range(1, levels + 1):
        h_step, w_step = H // L, W // L
        for i in range(L):
            for j in range(L):
                y0 = i * h_step
                y1 = (i + 1) * h_step if i < L - 1 else H
                x0 = j * w_step
                x1 = (j + 1) * w_step if j < L - 1 else W
                patch = img_rgb[y0:y1, x0:x1, :]
                D = extractor.extract(patch)
                if D.size == 0:
                    h = np.zeros((K,), np.float32)
                else:
                    a = assign_descriptors(D, centers)
                    h = bow_hist(a, K)
                    if idf is not None:
                        h = h * idf
                        h = h / (np.linalg.norm(h) + 1e-12)
                parts.append(h.astype(np.float32))
    v = np.concatenate(parts, axis=0)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v
