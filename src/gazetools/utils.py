import os
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.neighbors import KernelDensity
from scipy.ndimage import gaussian_filter
from typing import Dict, List, Tuple


def zero_one_similarity(a, b):
    if a == b:
        return 1.0
    else:
        return 0.0


def nw_matching(pred_string, gt_string, gap=0.0):
    # NW string matching with zero_one_similarity
    F = np.zeros((len(pred_string) + 1, len(gt_string) + 1), dtype=np.float32)
    for i in range(1 + len(pred_string)):
        F[i, 0] = gap * i
    for j in range(1 + len(gt_string)):
        F[0, j] = gap * j
    for i in range(1, 1 + len(pred_string)):
        for j in range(1, 1 + len(gt_string)):
            a = pred_string[i - 1]
            b = gt_string[j - 1]
            match = F[i - 1, j - 1] + zero_one_similarity(a, b)
            delete = F[i - 1, j] + gap
            insert = F[i, j - 1] + gap
            F[i, j] = np.max([match, delete, insert])
    score = F[len(pred_string), len(gt_string)]
    return score / max(len(pred_string), len(gt_string))


def scanpath2categories(seg_map, scanpath):
    string = []
    xs = scanpath["X"]
    ys = scanpath["Y"]
    
    if "T" in scanpath:
        ts = scanpath["T"]
        for x, y, t in zip(xs, ys, ts):
            x = int(max(0, min(x, seg_map.shape[1] - 1)))
            y = int(max(0, min(y, seg_map.shape[0] - 1)))
            
            symbol = str(int(seg_map[y, x]))
            string.append((symbol, t))
    else:
        for x, y in zip(xs, ys):
            x = int(max(0, min(x, seg_map.shape[1] - 1)))
            y = int(max(0, min(y, seg_map.shape[0] - 1)))
            
            symbol = str(int(seg_map[y, x]))
            string.append((symbol))
    return string

def scanpath2clusters(meanshift: MeanShift, scanpath):
    string = []
    xs = scanpath["X"]
    ys = scanpath["Y"]
    for i in range(len(xs)):
        symbol = meanshift.predict(np.array([[xs[i], ys[i]]]))[0]
        string.append(symbol)
    return string


def compute_saliency_map(points: np.ndarray, size: Tuple[int, int], method="conv"):

    w = size[1]
    h = size[0]

    if method == "KDEsk":

        sigma = 1 / 0.039

        x1 = np.linspace(0, w, w)
        x2 = np.linspace(0, h, h)
        X1, X2 = np.meshgrid(x1, x2)
        positions = np.vstack([X1.ravel(), X2.ravel()]).T

        kde_skl = KernelDensity(bandwidth=sigma)
        kde_skl.fit(points)

        Z = np.exp(kde_skl.score_samples(positions))
        Z = np.reshape(Z, X1.shape).T

    elif method == "conv":

        sigma = 1 / 0.02
        H, xedges, yedges = np.histogram2d(
            points[:, 1], points[:, 0], bins=(range(w + 1), range(h + 1))
        )
        Z = gaussian_filter(H, sigma=sigma)
        Z = Z / np.sum(Z).astype(np.float32)
    return Z  # type: ignore


def get_all_images_in_folder(directory):
    image_files: List[str] = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp")):
                image_files.append(os.path.join(root, file))
    return image_files
