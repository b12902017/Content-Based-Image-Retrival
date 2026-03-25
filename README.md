# Image Retrieval Project

This project relies heavily on OpenCV APIs for image processing and feature extraction, and credits OpenCV for its robust feature implementations.

---

## 1. Overview

Implemented feature families:

* **Color:** HSV 3D histogram, Grid HSV moments, Chromaticity r–g histogram
* **Texture/Shape:** LBP (uniform), Gradient orientation histogram, Gabor bank, Laws’ energy measures
* **Local:** SIFT/RootSIFT with BoW (K=800), optional Spatial Pyramid extension
* **Random Projection (RP):** Reduce dimension with floating & binarized variants

Similarity metrics: **cosine** (default) or **χ²** for histogram-like features. \
Evaluation: **leave-one-out retrieval**, reporting **MAP** overall and per category.

---

## 2. Project Structure

```
project/
├── codebase/
|   ├── extract
│   │   ├── color_extract.py
│   │   ├── texture_extract.py
│   │   └── loc_feat_extract.py
│   ├── feature
│   │   ├── color.py
│   │   └── texture.py
│   ├── loc_feat_util
│   │   ├── encode.py
│   │   ├── make_codebook.py
│   │   ├── sift.py
│   │   └── vocab.py
|   ├── dataset.py
|   ├── eval.py
|   ├── random_projection.py
|   ├── fuse.py
|   └── util.py
├── database/
|   ├── category1/
|   ├── category2/
|   ...
├── features/
├── runs/
└── README.md
```

Each feature directory stores a consistent schema (matrix.npy, ids.json, labels.json, meta.json). \
Each run directory stores evaluation summary and (optional) visualization

---

## 3. Environment

```
Python >= 3.9
OpenCV >= 4.5
NumPy
```

---

## 4. Feature Extraction and MAP Evaluation

Methods, parameters, and feature variants are configurable through argparse for flexible experimentation.

### Color

```bash
python -m codebase.extract.color_extract --method hsv_hist
python -m codebase.eval --feat_dir features/hsv_hist_h16_s4_v4
```

### Texture/Shape

```bash
python -m codebase.extract.texture_extract --method lbp
python -m codebase.eval --feat_dir features/lbp_p8r1_g3
```

### Local (SIFT/RootSIFT + BoW)

```bash
# Make codebooks
python -m codebase.loc_feat_util.make_codebook
# Extract 
python -m codebase.extract.loc_feat_extract --vocab_dir features/codebook/siftbow_k800_root
# Evaluate
python -m codebase.eval --feat_dir features/siftbow_k800_root
```

---

## 5. Random Projection

```bash
python -m codebase.random_projection --feat_dir features/siftbow_k800_root
```

Gaussian RP reduces 800→200 dimensions:

* Floating RP: multiply by Gaussian matrix and L2-normalize.
* Binary RP: sign(xR) → {−1,+1}, then L2-normalize.

Both evaluated with cosine similarity. \
RP can work on different features or output dimensions through ```argparse```.

---

## 7. Methods Summary

| Category | Feature                  | Key Steps             | Dim  | Metric |
| -------- | ------------------------ | --------------------- | ---- | ------ |
| Color    | HSV 3D histogram         | 16×4×4 bins, L2       | 256  | cosine |
| Color    | Grid HSV moments         | 3×3 mean/std          | 54   | cosine |
| Color    | Chromaticity r–g         | 32×32 bins            | 1024 | cosine |
| Texture  | LBP (u2, 3×3)            | 59-bin per cell       | 531  | χ²     |
| Texture  | Gradient hist (HOG-like) | 9-bin, soft vote      | 81   | χ²     |
| Texture  | Gabor (4×6)              | mean/std per response | 48   | cosine |
| Texture  | Laws (9 filters)         | mean/std              | 18   | cosine |
| Local    | RootSIFT + BoW           | K=800                 | 800  | cosine |
| Local    | +Spatial Pyramid         | multi-level BoW       | >800 | cosine |
| Local    | +RP (¼-dim)              | 800→200, float/bin    | 200  | cosine |

---

## 8. Results

The following table is the MAP of the IR on the dataset provided by this course (Cognitive Computing, 2025 Fall).

| Method             | MAP   |
| ------------------ | ----- |
| HSV Histogram      | 0.173 |
| HSV Moment         | 0.179 |
| Chromaticity RG    | 0.147 |
| LBP                | 0.183 |
| Gradient Histogram | 0.204 |
| Gabor's Texture    | 0.107 |
| Law's Texture      | 0.127 |
| SIFT BoW           | 0.287 |
| RP (BoW, 200 fl)   | 0.238 |
| RP (BoW, 200 bi)   | 0.192 |

---

## 8.5. Feature Fusion (RRF)

The repository also includes a fusion utility for combining ranks from multiple feature types using Reciprocal Rank Fusion (RRF).
Run it as:
```
python -m codebase.fuse \
  --feat_dir features/siftbow_k800_root features/gradhist_b9_g3/ features/grid_moments_hsv_g3/ \ 
  --metrics cosine chi2 cosine \
  --weights 0.6 0.2 0.2
```
This will compute individual rank matrices for each feature directory, apply RRF with the given weights, and output fused ranks under ```runs/fused_rrf```.\
Again different feature fusions with different weights and result visualization are suppoted.

---

## 9. Notes

* Use fixed random seeds for reproducibility.
* Extract features offline for faster evaluation.
* Ensure histograms are non-negative when using χ².
* Falls back to ORB if SIFT is unavailable in OpenCV build.
