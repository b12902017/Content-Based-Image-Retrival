import os, json, argparse, time
import numpy as np
from ..dataset import ImageDataset
from ..util import save_feature_pack
from ..feature.color import HSVHist, GridMomentsHSV, ChromaticityRGHist

def build_method(method: str, args: argparse.Namespace):
    m = method.lower()
    if m == "hsv_hist":
        return HSVHist(h_bins=args.h_bins, s_bins=args.s_bins, v_bins=args.v_bins), "cosine"
    if m == "grid_moments_hsv":
        return GridMomentsHSV(grid=args.grid), "l2"
    if m == "chrom_hist":
        return ChromaticityRGHist(r_bins=args.r_bins, g_bins=args.g_bins), "cosine"
    raise ValueError(f"Unknown method {method}")

def make_feat_dir_name(method: str, args: argparse.Namespace):
    parts = [method]
    if method == "hsv_hist":
        parts += [f"h{args.h_bins}", f"s{args.s_bins}", f"v{args.v_bins}"]
    elif method == "grid_moments_hsv":
        parts += [f"g{args.grid}"]
    elif method == "chrom_hist":
        parts += [f"r{args.r_bins}", f"g{args.g_bins}"]
    if args.tag:
        parts.append(args.tag)
    return "_".join(parts)

def main():
    p = argparse.ArgumentParser("Extract color features")
    p.add_argument("--root", default="database")
    p.add_argument("--out_root", default="features", help="where to store features")
    p.add_argument("--method", required=True,
                   choices=["hsv_hist","grid_moments_hsv","chrom_hist"])
    # p.add_argument("--max_side", type=int, default=0, help="resize so max(H,W)<=max_side (0=off)")
    p.add_argument("--tag", default="", help="suffix for feature folder name")

    # params
    p.add_argument("--h_bins", type=int, default=16)
    p.add_argument("--s_bins", type=int, default=4)
    p.add_argument("--v_bins", type=int, default=4)
    p.add_argument("--grid", type=int, default=3)
    p.add_argument("--r_bins", type=int, default=32)
    p.add_argument("--g_bins", type=int, default=32)
    args = p.parse_args()

    method, rec_metric = build_method(args.method, args)
    ds = ImageDataset(args.root)
    N = len(ds)
    print(f"[INFO] dataset: {N} images / {len(set(x.category for x in ds.items))} classes")

    feats = []
    t0 = time.time()
    for i in range(N):
        img = ds.load_image(i)
        vec = method.extract(img)
        feats.append(vec)
        if (i+1) % 50 == 0:
            print(f"  extracted {i+1}/{N}")

    X = np.stack(feats, axis=0).astype(np.float32)
    feat_dirname = make_feat_dir_name(args.method, args)
    out_dir = os.path.join(args.out_root, feat_dirname)

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

    if args.method == "hsv_hist":
        meta["params"].update({
            "h_bins": args.h_bins,
            "s_bins": args.s_bins,
            "v_bins": args.v_bins
        })
    elif args.method == "grid_moments_hsv":
        meta["params"].update({
            "grid": args.grid,
        })
    elif args.method == "chrom_hist":
        meta["params"].update({
            "r_bins": args.r_bins,
            "g_bins": args.g_bins
        })

    save_feature_pack(out_dir, X, ids, labels, meta)
    print(f"[OK] saved → {out_dir}  shape={X.shape}  rec_metric={rec_metric}  time={time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
