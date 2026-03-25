import os, json, time, argparse
import numpy as np
from ..dataset import ImageDataset
from ..util import save_feature_pack, DIST   # DIST has chi2/cosine/etc.
from ..loc_feat_util.sift import SIFTExtractor
from ..loc_feat_util.encode import assign_descriptors, bow_hist, bow_pyramid
from ..loc_feat_util.vocab import load_codebook

def main():
    ap = argparse.ArgumentParser("Encode BoW features using a trained codebook")
    ap.add_argument("--root", default="database")
    ap.add_argument("--out_root", default="features")
    ap.add_argument("--vocab_dir", required=True, help="path to features/vocabs/siftbow_kXXX[_root]")
    ap.add_argument("--rootsift", default=True, help="should match the codebook (check vocab meta)")
    ap.add_argument("--tfidf", default=True, help="apply IDF if idf.npy exists")
    ap.add_argument("--precompute_rank", action="store_true", help="also save pairwise distance+rank")
    ap.add_argument("--metric", default="cosine", help="distance for precompute_rank (chi2|cosine|l2)")
    ap.add_argument("--pyramid", action="store_true", help="use spatial pyramid (1x1 + 2x2 by default)")
    ap.add_argument("--pyr_levels", type=int, default=2)

    args = ap.parse_args()

    # load codebook
    centers, idf, vocab_meta = load_codebook(args.vocab_dir)
    K = centers.shape[0]
    print(f"[INFO] loaded vocab K={K} (rootsift={vocab_meta.get('rootsift')})")

    # encode dataset
    ds = ImageDataset(args.root)
    ext = SIFTExtractor(rootsift=args.rootsift)
    levels = args.pyr_levels if args.pyramid else 1
    H = []
    for i in range(len(ds)):
        img = ds.load_image(i)
        if args.pyramid:
            h = bow_pyramid(img, ext, centers, idf=(idf if args.tfidf else None), levels=levels)
        else:
            D = ext.extract(ds.load_image(i))
            a = assign_descriptors(D, centers)
            h = bow_hist(a, K)
            if args.tfidf and idf is not None:
                h = h * idf
                h = h / (np.linalg.norm(h) + 1e-12)
        H.append(h)
        if (i+1) % 50 == 0:
            print(f"  encoded {i+1}/{len(ds)}")
    X = np.stack(H, axis=0).astype(np.float32)

    # save feature pack
    out_name = f"siftbow_k{K}_p{args.pyr_levels}" + ("_root" if args.rootsift else "")
    feat_dir = os.path.join(args.out_root, out_name)
    ids = [it.img_id for it in ds.items]
    labels = [it.category for it in ds.items]
    meta = {
        "method": "sift_bow",
        "params": {
            "k": int(K),
            "rootsift": bool(args.rootsift),
            "tfidf": bool(args.tfidf),
            "vocab_dir": os.path.abspath(args.vocab_dir),
            "pyramid": bool(args.pyramid),
            "pyr_levels": levels,
        },
        "metric_default": "cosine",
        "N": int(len(ds)),
        "D": int(X.shape[1]),
        "root": os.path.abspath(args.root),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_feature_pack(feat_dir, X, ids, labels, meta)
    print(f"[OK] saved → {feat_dir}  shape={X.shape}")

    # optional: precompute pairwise distances + ranks (handy for quick browsing)
    if args.precompute_rank:
        metric = args.metric.lower()
        if metric not in DIST:
            raise ValueError(f"Unknown metric {metric}. Available: {list(DIST.keys())}")
        print(f"[RANK] computing pairwise {metric}…")
        D = DIST[metric](X, X)         # [N,N]
        np.fill_diagonal(D, np.inf)    # exclude self by default
        ranks = np.argsort(D, axis=1)  # nearest-first
        np.save(os.path.join(feat_dir, f"pairwise_{metric}.npy"), D.astype(np.float32))
        np.save(os.path.join(feat_dir, f"ranks_{metric}.npy"), ranks.astype(np.int32))
        print(f"[RANK] saved pairwise_{metric}.npy and ranks_{metric}.npy")

if __name__ == "__main__":
    main()
