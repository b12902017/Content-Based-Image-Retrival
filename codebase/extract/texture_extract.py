# cbir/extract/texture_extract.py
import os, json, time, argparse
import numpy as np

from ..dataset import ImageDataset
from ..util import save_feature_pack
from ..feature.texture import LBPHist, GradHist, GaborTexture, LawsTexture

def build_method(method: str, args: argparse.Namespace):
    m = method.lower()
    if m == "lbp":
        feat = LBPHist(grid=args.grid)
        metric = "chi2"
        name = f"lbp_p8r1_g{args.grid}"
        return feat, metric, name
    if m == "gradhist":
        feat = GradHist(bins=args.bins, grid=args.grid, soft=True)
        metric = "chi2"
        name = f"gradhist_b{args.bins}_g{args.grid}"
        return feat, metric, name
    if m == "gabor":
        feat = GaborTexture(scales=args.scales, orientations=args.orientations,
                            ksize=args.ksize, sigma0=args.sigma0, gamma=args.gamma, psi=args.psi)
        metric = "cosine"
        name = f"gabor_s{args.scales}_o{args.orientations}_k{args.ksize}"
        return feat, metric, name
    if m == "laws":
        feat = LawsTexture(subset="9", normalize="l2")
        metric = "cosine"
        name = "laws9"
        return feat, metric, name
    raise ValueError(f"Unknown method {method}")

def main():
    p = argparse.ArgumentParser("Extract texture features")
    p.add_argument("--root", default="database")
    p.add_argument("--out_root", default="features")
    p.add_argument("--method", required=True, choices=["lbp", "gradhist", "gabor", "laws"])
    p.add_argument("--grid", type=int, default=3, help="grid split per side (1=global)")
    p.add_argument("--bins", type=int, default=9, help="for gradhist only")
    p.add_argument("--scales", type=int, default=4)
    p.add_argument("--orientations", type=int, default=6)
    p.add_argument("--ksize", type=int, default=21)
    p.add_argument("--sigma0", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--psi", type=float, default=0.0)
    args = p.parse_args()

    method, rec_metric, shortname = build_method(args.method, args)
    ds = ImageDataset(args.root)
    N = len(ds)
    print(f"[INFO] dataset: {N} images / {len(set(x.category for x in ds.items))} classes")

    feats = []
    t0 = time.time()
    for i in range(N):
        img = ds.load_image(i)
        vec = method.extract(img)
        feats.append(vec)
        if (i + 1) % 50 == 0:
            print(f"  extracted {i+1}/{N}")

    X = np.stack(feats, axis=0).astype(np.float32)
    out_dir = os.path.join(args.out_root, shortname)

    ids = [it.img_id for it in ds.items]
    labels = [it.category for it in ds.items]
    meta = {
        "method": args.method,
        "params": {},
        "metric_default": rec_metric,
        "N": int(N),
        "D": int(X.shape[1]),
        "root": os.path.abspath(args.root),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # --- populate params depending on method ---
    if args.method == "lbp":
        meta["params"].update({
            "grid": args.grid,
        })
    elif args.method == "gradhist":
        meta["params"].update({
            "grid": args.grid,
            "bins": args.bins,
        })
    elif args.method == "gabor":
        meta["params"].update({
            "scales": args.scales,
            "orientations": args.orientations,
            "ksize": args.ksize,
            "sigma0": args.sigma0,
            "gamma": args.gamma,
            "psi": args.psi,
        })
    elif args.method == "laws":
        meta["params"].update({
            "subset": "9-filter",
            "normalize": "l2",
        })
    
    save_feature_pack(out_dir, X, ids, labels, meta)
    print(f"[OK] saved → {out_dir}  shape={X.shape}  rec_metric={rec_metric}  time={time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
