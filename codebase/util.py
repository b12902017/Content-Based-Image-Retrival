import os, json
import numpy as np
import cv2

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def l2_normalize(x: np.ndarray, eps=1e-12) -> np.ndarray:
    n = np.linalg.norm(x) + eps
    return (x / n).astype(np.float32)

'''
def resize_if_needed(img: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / m
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
'''

def save_feature_pack(out_dir, X, ids, labels, meta):
    ensure_dir(out_dir)
    np.save(os.path.join(out_dir, "matrix.npy"), X.astype(np.float32))
    with open(os.path.join(out_dir, "ids.json"), "w") as f: json.dump(ids, f)
    with open(os.path.join(out_dir, "labels.json"), "w") as f: json.dump(labels, f)
    with open(os.path.join(out_dir, "meta.json"), "w") as f: json.dump(meta, f, indent=2)

def l2(A, B):
    A2 = (A*A).sum(1, keepdims=True)
    B2 = (B*B).sum(1, keepdims=True).T
    D2 = np.maximum(A2 + B2 - 2*A@B.T, 0.0)
    return np.sqrt(D2, dtype=np.float64)

def cosine(A, B, eps=1e-12):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    return 1.0 - (An @ Bn.T)

def l1(A, B):
    return np.abs(A[:,None,:] - B[None,:,:]).sum(-1)

def chi2(A, B, eps=1e-12):
    A_ = A[:, None, :]   # [N,1,D]
    B_ = B[None, :, :]   # [1,M,D]
    num = (A_ - B_) ** 2
    den = (A_ + B_) + eps
    D = 0.5 * np.sum(num / den, axis=-1)  # [N,M]
    return D

DIST = {"cosine": cosine, "l2": l2, "l1": l1, "chi2": chi2}

def average_precision(ranked_labels):
    r = np.asarray(ranked_labels, dtype=np.int32)
    pos = r.sum()
    if pos == 0: return 0.0
    c = np.cumsum(r)
    prec = c / (np.arange(len(r)) + 1)
    ap = (prec * r).sum() / pos
    return float(ap)

def eval_map(labels, ranks):
    labels = np.asarray(labels)
    N = len(labels)
    ap_list = []
    for i in range(N):
        same = (labels[ranks[i]] == labels[i]).astype(np.int32)
        ap_list.append(average_precision(same))
    MAP = float(np.mean(ap_list))
    per = {}
    for c in sorted(set(labels.tolist())):
        idx = [i for i in range(N) if labels[i]==c]
        per[c] = float(np.mean([ap_list[i] for i in idx])) if idx else 0.0
    return MAP, per, ap_list

import os, glob, textwrap
import cv2
import numpy as np
import matplotlib.pyplot as plt

ALLOWED_EXTS = (".jpg", ".jpeg", ".png")

def id_to_short(x: str, n=18) -> str:
    return x if len(x) <= n else x[:n-3] + "..."

def find_image_path(root: str, label: str, img_id: str) -> str:
    base = os.path.join(root, label)
    for ext in ALLOWED_EXTS:
        p = os.path.join(base, img_id + ext)
        if os.path.exists(p):
            return p
    hits = []
    for ext in ALLOWED_EXTS:
        hits.extend(glob.glob(os.path.join(base, img_id + "*" + ext)))
    if hits:
        return sorted(hits)[0]
    raise FileNotFoundError(f"Cannot locate image for id={img_id} under {base}")

def _wrap(txt: str, width: int) -> str:
    return textwrap.fill(txt, width=width, break_long_words=True, break_on_hyphens=False)

def _score_text(d: float, metric: str, as_similarity: bool) -> (str, float):
    metric = metric.lower()
    if as_similarity and metric == "cosine":
        s = 1.0 - float(d)            # cosine sim in [−1,1], often ~[0,1]
        return f"sim={s:.3f}", s
    # distances (l2, l1, cosine distance)
    return f"d={float(d):.3f}", float(d)

def visualize_topk_for_queries(
    root, ids, labels, ranks, query_indices, k=5,
    save_dir=None, show=False,
    figsize_unit=3.0, title_width=22, title_fs=10, title_pad=6,
    scores=None, metric="cosine", as_similarity=True
):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for qi in query_indices:
        qid, qlab = ids[qi], labels[qi]
        qpath = find_image_path(root, qlab, qid)

        fig, axes = plt.subplots(
            1, k + 1,
            figsize=(figsize_unit * (k + 1), figsize_unit * 1.05),
            constrained_layout=True
        )

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        # Query
        ax0 = axes[0]
        qimg = cv2.cvtColor(cv2.imread(qpath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        ax0.imshow(qimg); ax0.set_axis_off()
        q_title = _wrap(f"QUERY\n{id_to_short(qid)} [{qlab}]", title_width)
        ax0.set_title(q_title, fontsize=title_fs, pad=title_pad)

        # Results
        topk = ranks[qi][:k]
        for j, idx in enumerate(topk, start=1):
            rid, rlab = ids[idx], labels[idx]
            rpath = find_image_path(root, rlab, rid)
            rimg = cv2.cvtColor(cv2.imread(rpath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            ax = axes[j]
            ax.imshow(rimg); ax.set_axis_off()

            hit = (rlab == qlab)
            # score text
            if scores is not None:
                stxt, sval = _score_text(scores[qi, idx], metric, as_similarity)
            else:
                stxt, sval = ("", np.nan)

            mark = "✓" if hit else "✗"
            title = _wrap(f"{mark} {id_to_short(rid)} [{rlab}]\n{stxt}", title_width)
            ax.set_title(title.strip(), fontsize=title_fs, pad=title_pad)

        if save_dir:
            outp = os.path.join(save_dir, f"viz_{qid}.png")
            fig.savefig(outp, dpi=140, bbox_inches="tight", pad_inches=0.2)
            print(f"[VIZ] saved {outp}")
        if show:
            plt.show()
        plt.close(fig)
