import os, argparse, json
import numpy as np

def make_rp_matrix(d_in: int, d_out: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    R = rng.normal(loc=0.0, scale=1.0/np.sqrt(d_out), size=(d_in, d_out)).astype(np.float32)
    return R

def l2norm_rows(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return (X / n).astype(np.float32)

def save_pack(out_dir, X, ids, labels, base_meta, extra_params, metric_default, D_new):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "matrix.npy"), X.astype(np.float32))
    with open(os.path.join(out_dir, "ids.json"), "w") as f: json.dump(ids, f)
    with open(os.path.join(out_dir, "labels.json"), "w") as f: json.dump(labels, f)
    meta = dict(base_meta)
    meta["method"] = (meta.get("method", "") + "_rp").strip("_")
    params = dict(meta.get("params", {}))
    params.update(extra_params)
    meta["params"] = params
    meta["metric_default"] = metric_default
    meta["D"] = int(D_new)
    with open(os.path.join(out_dir, "meta.json"), "w") as f: json.dump(meta, f, indent=2)

def main():
    ap = argparse.ArgumentParser("Random Projection to 1/4 dim (floating + binarized)")
    ap.add_argument("--feat_dir", required=True, help="features/<name> with matrix.npy")
    ap.add_argument("--out_dim", type=int, default=None, help="target dim (default: 1/4 of input)")
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--save_proj", action="store_true", help="also save R.npy")
    args = ap.parse_args()

    X = np.load(os.path.join(args.feat_dir, "matrix.npy")).astype(np.float32)
    with open(os.path.join(args.feat_dir, "ids.json"), "r") as f: ids = json.load(f)
    with open(os.path.join(args.feat_dir, "labels.json"), "r") as f: labels = json.load(f)
    with open(os.path.join(args.feat_dir, "meta.json"), "r") as f: meta = json.load(f)

    N, D_in = X.shape
    d_out = args.out_dim if args.out_dim is not None else max(1, int(round(D_in * 0.25)))
    print(f"[INFO] RP {N}x{D_in} → {N}x{d_out} (seed={args.seed})")

    R = make_rp_matrix(D_in, d_out, seed=args.seed)          # [D_in, d_out]
    Xp = X @ R                                               # float projection
    Xp_fl = l2norm_rows(Xp)                                  # floating version (cosine-ready)

    # Binarized: sign of projected values → {-1,+1}, then L2 to use cosine
    Xp_bi = np.sign(Xp, dtype=np.float32)
    Xp_bi[Xp_bi == 0] = 1.0
    Xp_bi = l2norm_rows(Xp_bi)

    base = os.path.basename(args.feat_dir.rstrip(os.sep))
    parent = os.path.dirname(args.feat_dir.rstrip(os.sep))
    out_fl = os.path.join(parent, f"{base}_rp{d_out}_fl")
    out_bi = os.path.join(parent, f"{base}_rp{d_out}_bi")

    save_pack(
        out_fl, Xp_fl, ids, labels, meta,
        extra_params={"rp_seed": int(args.seed), "rp_d_in": int(D_in), "rp_d_out": int(d_out), "rp_type": "gaussian", "rp_variant": "float"},
        metric_default="cosine", D_new=d_out
    )
    print(f"[OK] floating RP saved → {out_fl}")

    save_pack(
        out_bi, Xp_bi, ids, labels, meta,
        extra_params={"rp_seed": int(args.seed), "rp_d_in": int(D_in), "rp_d_out": int(d_out), "rp_type": "gaussian", "rp_variant": "binary_sign_(-1,+1)"},
        metric_default="cosine", D_new=d_out
    )
    print(f"[OK] binarized RP saved → {out_bi}")

    if args.save_proj:
        np.save(os.path.join(out_fl, "R.npy"), R)
        np.save(os.path.join(out_bi, "R.npy"), R)
        print("[OK] saved projection matrix R.npy (same for fl/bi)")

if __name__ == "__main__":
    main()
