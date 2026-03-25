import numpy as np
import cv2
from ..util import l2_normalize

class HSVHist:
    def __init__(self, h_bins=16, s_bins=4, v_bins=4):
        self.h_bins, self.s_bins, self.v_bins = int(h_bins), int(s_bins), int(v_bins)
    def extract(self, bgr_img):
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv],[0,1,2],None,
                            [self.h_bins,self.s_bins,self.v_bins],
                            [0,180,0,256,0,256]).flatten()
        return l2_normalize(hist)

class GridMomentsHSV:
    def __init__(self, grid=3):
        self.grid = int(grid)
    def extract(self, bgr_img):
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        H, W = hsv.shape[:2]
        gy = np.linspace(0, H, self.grid+1, dtype=int)
        gx = np.linspace(0, W, self.grid+1, dtype=int)
        feats = []
        for i in range(self.grid):
            for j in range(self.grid):
                patch = hsv[gy[i]:gy[i+1], gx[j]:gx[j+1], :]
                if patch.size == 0:
                    feats.extend([0,0,0,0,0,0])
                else:
                    flat = patch.reshape(-1,3).astype(np.float32)
                    m, s = flat.mean(0), flat.std(0)
                    feats.extend([m[0],s[0], m[1],s[1], m[2],s[2]])
        return l2_normalize(np.array(feats, dtype=np.float32))

class ChromaticityRGHist:
    def __init__(self, r_bins=32, g_bins=32, eps=1e-6):
        self.r_bins, self.g_bins, self.eps = int(r_bins), int(g_bins), float(eps)
    def extract(self, bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB).astype(np.float32)
        s = rgb.sum(2, keepdims=True) + self.eps
        rg = rgb / s
        r, g = rg[:,:,0], rg[:,:,1]
        hist = cv2.calcHist([r,g],[0,1],None,
                            [self.r_bins,self.g_bins],[0,1,0,1]).flatten()
        return l2_normalize(hist)
