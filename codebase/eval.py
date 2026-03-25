# eval.py
import os, json, argparse, time, random
import numpy as np
from .util import DIST, eval_map, visualize_topk_for_queries

def main():
    p = argparse.ArgumentParser("Leave-one-out evaluation")
    p.add_argument("--feat_dir", required=True)
    p.add_argument("--metric", default="", choices=["","cosine","l2","l1","chi2"])
    p.add_argument("--include_self", action="store_true")
    p.add_argument("--save_ranks", action="store_true")

    # ---- visualization options ----
    p.add_argument("--viz_n", type=int, default=0, help="# of random queries to visualize (0=off)")
    p.add_argument("--viz_k", type=int, default=5, help="top-k shown per query")
    p.add_argument("--viz_seed", type=int, default=71, help="rng seed for sampling queries")
    p.add_argument("--viz_show", action="store_true", help="also pop up matplotlib windows")
    args = p.parse_args()

    X = np.load(os.path.join(args.feat_dir, "matrix.npy"))
    labels = json.load(open(os.path.join(args.feat_dir, "labels.json")))
    ids = json.load(open(os.path.join(args.feat_dir, "ids.json")))
    meta = json.load(open(os.path.join(args.feat_dir, "meta.json")))

    metric = args.metric or meta.get("metric_default","cosine")
    if metric not in DIST:
        raise ValueError(f"Unknown metric {metric}")
    print(f"[INFO] metric={metric}  N={X.shape[0]}  D={X.shape[1]}")

    t0 = time.time()
    D = DIST[metric](X, X)
    print("[INFO] computed distance matrix")
    if not args.include_self:
        np.fill_diagonal(D, np.inf)
    ranks = np.argsort(D, axis=1)

    MAP, per_cls, ap_list = eval_map(labels, ranks)
    dt = time.time() - t0

    out_dir = os.path.join("runs", os.path.basename(args.feat_dir.rstrip("/")))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({
            "feat_dir": args.feat_dir,
            "metric": metric,
            "include_self": bool(args.include_self),
            "N": int(X.shape[0]),
            "D": int(X.shape[1]),
            "MAP": MAP,
            "per_class": per_cls,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "compute_time_sec": dt,
        }, f, indent=2)

    if args.save_ranks:
        with open(os.path.join(out_dir, "ranks.jsonl"), "w") as f:
            for i in range(len(ids)):
                f.write(json.dumps({
                    "query_id": ids[i],
                    "query_label": labels[i],
                    "rank_ids": [ids[j] for j in ranks[i].tolist()],
                    "rank_labels": [labels[j] for j in ranks[i].tolist()]
                }) + "\n")

    print(f"[EVAL] MAP={MAP:.4f}  time={dt:.2f}s")
    print("  Top2:", sorted(per_cls.items(), key=lambda x: x[1], reverse=True)[:2])
    print("  Worst2:", sorted(per_cls.items(), key=lambda x: x[1])[:2])
    print("  MAP per class:", sorted(per_cls.items(), key=lambda x: x[1], reverse=True))

    if args.viz_n > 0:
        random.seed(args.viz_seed)
        N = X.shape[0]
        qs = random.sample(range(N), k=min(args.viz_n, N))
        viz_dir = os.path.join(out_dir, "viz")
        ds_root = meta.get("root", "database")
        visualize_topk_for_queries(
            root=ds_root,
            ids=ids,
            labels=labels,
            ranks=ranks,
            query_indices=qs,
            k=args.viz_k,
            save_dir=viz_dir,
            show=args.viz_show,
            scores=D,
            metric=metric,
            as_similarity=True,
        )
        print(f"[VIZ] wrote {len(qs)} figures to {viz_dir}")

if __name__ == "__main__":
    main()
