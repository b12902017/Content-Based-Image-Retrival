import os, json, time, argparse
import numpy as np
from ..dataset import ImageDataset
from .sift import SIFTExtractor
from .encode import assign_descriptors, bow_hist, idf_from_histograms
from .vocab import kmeans_opencv, save_codebook

def main():
    ap = argparse.ArgumentParser("Make a SIFT/RootSIFT codebook (K-Means)")
    ap.add_argument("--root", default="database", help="image root to sample descriptors from")
    ap.add_argument("--out_root", default="features/codebook")
    ap.add_argument("--k", type=int, default=800)
    ap.add_argument("--max_per_image", type=int, default=200)
    ap.add_argument("--max_total", type=int, default=200000)
    ap.add_argument("--rootsift", default=True)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--idf_root", default="database", help="optional: dataset root for computing IDF")
    args = ap.parse_args()

    ds = ImageDataset(args.root)
    ext = SIFTExtractor(rootsift=args.rootsift)

    # sample descriptors
    rng = np.random.default_rng(args.seed)
    bank = []
    for i in range(len(ds)):
        D = ext.extract(ds.load_image(i))
        if D.size == 0: continue
        if D.shape[0] > args.max_per_image:
            idx = rng.choice(D.shape[0], size=args.max_per_image, replace=False)
            D = D[idx]
        bank.append(D)
    if not bank:
        raise RuntimeError("No descriptors found. Check images/SIFT installation.")
    X = np.concatenate(bank, axis=0).astype(np.float32)
    if X.shape[0] > args.max_total:
        idx = rng.choice(X.shape[0], size=args.max_total, replace=False)
        X = X[idx]

    print(f"[INFO] kmeans on {X.shape[0]}x{X.shape[1]} → K={args.k}")
    centers = kmeans_opencv(X, args.k, attempts=5, seed=args.seed)

    idf = None
    if args.idf_root:
        ds_idf = ImageDataset(args.idf_root)
        H_all = []
        for i in range(len(ds_idf)):
            D = ext.extract(ds_idf.load_image(i))
            a = assign_descriptors(D, centers)
            H_all.append(bow_hist(a, args.k))
        H_all = np.stack(H_all, axis=0)
        idf = idf_from_histograms(H_all)

    name = f"siftbow_k{args.k}" + ("_root" if args.rootsift else "")
    out_dir = os.path.join(args.out_root, name)
    meta = {
        "type": "sift_bow",
        "k": int(args.k),
        "rootsift": bool(args.rootsift),
        "desc_dim": int(X.shape[1]),
        "samples": int(X.shape[0]),
        "seed": int(args.seed),
        "trained_on": os.path.abspath(args.root),
        "idf_from": (os.path.abspath(args.idf_root) if args.idf_root else None),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_codebook(out_dir, centers, idf, meta)
    print(f"[OK] codebook saved → {out_dir}")

if __name__ == "__main__":
    main()
