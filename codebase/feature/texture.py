# cbir/feature/texture.py
import numpy as np
import cv2

def _uniform_lut_256():
    lut = np.empty(256, dtype=np.uint8)
    idx = 0
    for p in range(256):
        b = ((p >> np.arange(8)) & 1).astype(np.uint8)
        # circular transitions 0->1 or 1->0
        t = np.sum(b[:-1] != b[1:]) + (b[0] != b[-1])
        if t <= 2:
            lut[p] = idx
            idx += 1
        else:
            lut[p] = 58
    return lut

_LBP_LUT = _uniform_lut_256()  # 59 bins total (0..58)

class LBPHist:
    def __init__(self, grid: int = 3, normalize: str = "l2"):
        assert grid >= 1
        self.grid = int(grid)
        self.normalize = normalize

    @staticmethod
    def _lbp_uniform_8u1(gray: np.ndarray) -> np.ndarray:
        g = gray.astype(np.uint8)
        c = g[1:-1, 1:-1]
        n0 = g[0:-2, 1:-1]      # up
        n1 = g[0:-2, 2:  ]      # up-right
        n2 = g[1:-1, 2:  ]      # right
        n3 = g[2:  , 2:  ]      # down-right
        n4 = g[2:  , 1:-1]      # down
        n5 = g[2:  , 0:-2]      # down-left
        n6 = g[1:-1, 0:-2]      # left
        n7 = g[0:-2, 0:-2]      # up-left

        # binary pattern (>= center)
        code = ((n0 >= c).astype(np.uint8) << 7) \
             | ((n1 >= c).astype(np.uint8) << 6) \
             | ((n2 >= c).astype(np.uint8) << 5) \
             | ((n3 >= c).astype(np.uint8) << 4) \
             | ((n4 >= c).astype(np.uint8) << 3) \
             | ((n5 >= c).astype(np.uint8) << 2) \
             | ((n6 >= c).astype(np.uint8) << 1) \
             | ((n7 >= c).astype(np.uint8) << 0)
        
        return _LBP_LUT[code]

    def _hist(self, patch_codes: np.ndarray) -> np.ndarray:
        # 59 bins (0..58)
        hist = np.bincount(patch_codes.ravel(), minlength=59).astype(np.float32)
        s = hist.sum()
        if s > 0:
            hist /= s
        return hist

    def extract(self, img_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        codes = self._lbp_uniform_8u1(gray)  # shape ~ (H-2,W-2)
        H, W = codes.shape
        g = self.grid
        h_step, w_step = H // g, W // g
        feats = []
        for i in range(g):
            for j in range(g):
                y0, y1 = i * h_step, (i + 1) * h_step if i < g - 1 else H
                x0, x1 = j * w_step, (j + 1) * w_step if j < g - 1 else W
                feats.append(self._hist(codes[y0:y1, x0:x1]))
        v = np.concatenate(feats, axis=0)
        if self.normalize == "l2":
            n = np.linalg.norm(v) + 1e-12
            v = (v / n).astype(np.float32)
        return v


class GradHist:
    def __init__(self, bins: int = 9, grid: int = 3, normalize: str = "l2", soft: bool = True):
        assert bins >= 2 and grid >= 1
        self.bins = int(bins)
        self.grid = int(grid)
        self.normalize = normalize
        self.soft = bool(soft)

    def _grad(self, gray: np.ndarray):
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        ang = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0  # [0,180)
        return mag, ang

    def _hist_cell(self, mag: np.ndarray, ang: np.ndarray) -> np.ndarray:
        B = self.bins
        if self.soft:
            # soft assignment into two neighboring bins
            bin_f = (ang / 180.0) * B  # [0,B)
            b0 = np.floor(bin_f).astype(np.int32) % B
            b1 = (b0 + 1) % B
            w1 = bin_f - b0
            w0 = 1.0 - w1
            H = np.zeros(B, dtype=np.float32)
            for b, w in [(b0, w0), (b1, w1)]:
                add = np.bincount(b.ravel(), weights=(mag * w).ravel(), minlength=B).astype(np.float32)
                H += add
        else:
            # hard assignment
            b = np.floor((ang / 180.0) * B).astype(np.int32)
            b = np.clip(b, 0, B - 1)
            H = np.bincount(b.ravel(), weights=mag.ravel(), minlength=B).astype(np.float32)
        s = np.linalg.norm(H) + 1e-12
        H = (H / s).astype(np.float32)
        return H

    def extract(self, img_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        mag, ang = self._grad(gray)
        Hh, Ww = gray.shape
        g = self.grid
        h_step, w_step = Hh // g, Ww // g
        feats = []
        for i in range(g):
            for j in range(g):
                y0, y1 = i * h_step, (i + 1) * h_step if i < g - 1 else Hh
                x0, x1 = j * w_step, (j + 1) * w_step if j < g - 1 else Ww
                feats.append(self._hist_cell(mag[y0:y1, x0:x1], ang[y0:y1, x0:x1]))
        v = np.concatenate(feats, axis=0)
        if self.normalize == "l2":
            n = np.linalg.norm(v) + 1e-12
            v = (v / n).astype(np.float32)
        elif self.normalize == "l1":
            s = np.sum(np.abs(v)) + 1e-12
            v = (v / s).astype(np.float32)
        return v


class GaborTexture:
    def __init__(self, scales: int = 4, orientations: int = 6,
                 ksize: int = 21, sigma0: float = 2.0, gamma: float = 0.5, psi: float = 0.0):
        assert scales >= 1 and orientations >= 1
        self.scales = int(scales)
        self.orientations = int(orientations)
        self.ksize = int(ksize)
        self.sigma0 = float(sigma0)
        self.gamma = float(gamma)
        self.psi = float(psi)

        self.kernels = []
        for s in range(self.scales):
            lam = (self.ksize // 2) / (1.5 * (2 ** s)) + 2.0  # wavelength
            sigma = self.sigma0 * (2 ** s)
            for k in range(self.orientations):
                theta = np.pi * k / self.orientations
                kern = cv2.getGaborKernel((self.ksize, self.ksize), sigma, theta, lam, self.gamma, self.psi, ktype=cv2.CV_32F)
                self.kernels.append(kern)

    def extract(self, img_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        feats = []
        for K in self.kernels:
            resp = cv2.filter2D(gray, cv2.CV_32F, K)
            a = np.abs(resp)
            feats.extend([a.mean(), a.std()])
        v = np.asarray(feats, dtype=np.float32)

        v /= (np.linalg.norm(v) + 1e-12)
        return v


class LawsTexture:
    def __init__(self, subset: str = "9", normalize: str = "l2"):
        L5 = np.array([1, 4, 6, 4, 1], dtype=np.float32)
        E5 = np.array([-1, -2, 0, 2, 1], dtype=np.float32)
        S5 = np.array([-1, 0, 2, 0, -1], dtype=np.float32)
        W5 = np.array([-1, 2, 0, -2, 1], dtype=np.float32)
        R5 = np.array([1, -4, 6, -4, 1], dtype=np.float32)
        bank = {
            "L5": L5, "E5": E5, "S5": S5, "W5": W5, "R5": R5
        }
        pairs = [("E5", "L5"), ("L5", "E5"), ("E5", "E5"),
                 ("S5", "L5"), ("L5", "S5"), ("S5", "S5"),
                 ("W5", "L5"), ("L5", "W5"), ("R5", "R5")]
        self.kernels = []
        for a, b in pairs:
            k2d = np.outer(bank[a], bank[b])  # 5x5
            self.kernels.append(k2d.astype(np.float32))
        self.normalize = normalize

    def extract(self, img_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        # remove local DC (5x5 mean) for stability
        dc = cv2.blur(gray, (5, 5))
        x = gray - dc
        feats = []
        for K in self.kernels:
            resp = cv2.filter2D(x, cv2.CV_32F, K)
            a = np.abs(resp)
            feats.extend([a.mean(), a.std()])
        v = np.asarray(feats, dtype=np.float32)
        if self.normalize == "l2":
            v /= (np.linalg.norm(v) + 1e-12)
        elif self.normalize == "l1":
            v /= (np.sum(np.abs(v)) + 1e-12)
        return v
