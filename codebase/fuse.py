# cbir/tools/fuse.py
import os, json, argparse, numpy as np
from .util import DIST, eval_map
import random
from .eval import visualize_topk_for_queries

def reciprocal_rank_fusion(rank_list, k=60, weights=None, exclude_self=True):
    M = len(rank_list)
    if weights is None:
        weights = np.ones(M, dtype=np.float32)
    else:
        weights = np.asarray(weights, dtype=np.float32)
        weights = np.maximum(weights, 0)
        weights /= np.sum(weights) if np.sum(weights) > 0 else 1

    print(rank_list[0])
    print(rank_list[1])
    print(rank_list[2])

    N = rank_list[0].shape[0]
    scores = np.zeros((N, N), dtype=np.float32)
    rows = np.arange(N)[:, None]
    for w, R in zip(weights, rank_list):
        P = np.empty_like(R, dtype=np.int32)
        P[rows, R] = np.arange(N, dtype=np.int32)[None, :]  # invert permutation
        # RRF: 1 / (k + rank_pos + 1)   (+1 to make ranks 1-based)
        scores += w * (1.0 / (k + (P + 1).astype(np.float32)))
    if exclude_self:
        np.fill_diagonal(scores, -np.inf)
    fused_ranks = np.argsort(-scores, axis=1)
    print(fused_ranks)
    return fused_ranks

def compute_ranks(X, metric, include_self=False):
    D = DIST[metric](X, X).astype(np.float32)
    if not include_self:
        np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1)

def main():
    ap = argparse.ArgumentParser("Fuse multiple feature dirs using RRF")
    ap.add_argument("--feat_dirs", nargs="+", required=True)
    ap.add_argument("--metrics", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--k_rrf", type=int, default=60)
    ap.add_argument("--include_self", action="store_true")
    ap.add_argument("--eval_map", default=True)
    ap.add_argument("--viz_n", type=int, default=0, help="# of random queries to visualize (0=off)")
    ap.add_argument("--viz_k", type=int, default=5, help="top-k shown per query")
    ap.add_argument("--viz_seed", type=int, default=71, help="rng seed for sampling queries")
    ap.add_argument("--viz_show", action="store_true", help="also pop up matplotlib windows")
    args = ap.parse_args()

    assert len(args.feat_dirs) == len(args.metrics), "metrics and feat_dirs length mismatch"

    X0 = np.load(os.path.join(args.feat_dirs[0], "matrix.npy"))
    with open(os.path.join(args.feat_dirs[0], "labels.json")) as f:
        labels = json.load(f)
    with open(os.path.join(args.feat_dirs[0], "ids.json")) as f:
        ids0 = json.load(f)

    rank_list = []
    for d, metric in zip(args.feat_dirs, args.metrics):
        X = np.load(os.path.join(d, "matrix.npy"))
        with open(os.path.join(d, "ids.json")) as f:
            ids = json.load(f)
        if ids != ids0:
            raise ValueError(f"ID mismatch between {args.feat_dirs[0]} and {d}")
        print(f"[INFO] computing ranks for {d} using {metric}")
        R = compute_ranks(X, metric, include_self=args.include_self)
        rank_list.append(R)

    fused_ranks = reciprocal_rank_fusion(rank_list, k=args.k_rrf, weights=args.weights)

    if args.eval_map:
        mAP, per, ap_list = eval_map(labels, fused_ranks)
        print(f"[mAP] {mAP:.4f}")

    out_dir = "runs/fused_rrf"
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "ranks_fused.npy"), fused_ranks)
    with open(os.path.join(out_dir, "ids.json"), "w") as f:
        json.dump(ids0, f)
    print(f"[OK] fused ranks saved → {out_dir}")

    if args.viz_n > 0:
        random.seed(args.viz_seed)
        N = X.shape[0]
        qs = random.sample(range(N), k=min(args.viz_n, N))
        viz_dir = os.path.join(out_dir, "viz")
        ds_root = "database"
        visualize_topk_for_queries(
            root=ds_root,
            ids=ids,
            labels=labels,
            ranks=fused_ranks,
            query_indices=qs,
            k=args.viz_k,
            save_dir=viz_dir,
            show=args.viz_show,
            scores=None,
            metric=metric,
            as_similarity=True,
        )
        print(f"[VIZ] wrote {len(qs)} figures to {viz_dir}")

if __name__ == "__main__":
    main()
