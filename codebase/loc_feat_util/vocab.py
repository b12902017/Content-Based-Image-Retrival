import os, json, cv2, numpy as np

def kmeans_opencv(X: np.ndarray, K: int, attempts: int = 5, seed: int = 71) -> np.ndarray:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-3)
    flags = cv2.KMEANS_PP_CENTERS
    _c, _lab, centers = cv2.kmeans(X.astype(np.float32), K, None, criteria, attempts, flags)
    return centers

def save_codebook(out_dir: str, centers: np.ndarray, idf: np.ndarray | None, meta: dict):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "centers.npy"), centers)
    if idf is not None:
        np.save(os.path.join(out_dir, "idf.npy"), idf)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

def load_codebook(vocab_dir: str):
    centers = np.load(os.path.join(vocab_dir, "centers.npy"))
    idf = None
    p = os.path.join(vocab_dir, "idf.npy")
    if os.path.exists(p): idf = np.load(p)
    with open(os.path.join(vocab_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    return centers, idf, meta
