import argparse
import os
from os.path import join
import json
import numpy as np
import torch

from tqdm import tqdm
import warnings
from sklearn.cluster import MeanShift, estimate_bandwidth

warnings.filterwarnings("ignore")

# https://github.com/cvlab-stonybrook/Scanpath_Prediction/issues/24
def scanpath2clusters(meanshift, scanpath):
    string = []
    xs = scanpath['X']
    ys = scanpath['Y']
    for i in range(len(xs)):
        symbol = meanshift.predict([[xs[i], ys[i]]])[0]
        string.append(symbol)
    return string

def improved_rate(meanshift, scanpaths):
    Nc = len(meanshift.cluster_centers_)
    Nb, Nw = 0, 0
    for scanpath in scanpaths:
        string = scanpath2clusters(meanshift, scanpath)
        for i in range(len(string)-1):
            if string[i]==string[i+1]:
                Nw += 1
            else:
                Nb += 1
    return (Nb-Nw)/Nc

def compute_clusters(gt_scanpaths):
    xs, ys = [], []
    for scanpath in gt_scanpaths:
        xs += list(scanpath['X'])
        ys += list(scanpath['Y'])

    gt_gaze = np.concatenate((np.vstack(xs), np.vstack(ys)), axis=1)
    bandwidth = estimate_bandwidth(gt_gaze)
    rates = []
    factors = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8]  # [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    for factor in factors:
        bd = bandwidth * factor if bandwidth > 0.0 else None
        ms = MeanShift(bandwidth=bd)
        ms.fit(gt_gaze)
        rate = improved_rate(ms, gt_scanpaths)
        rates.append(rate)
    rates = np.vstack(rates)

    best_bd = factors[np.argmax(rates)] * bandwidth if bandwidth > 0.0 else None
    best_ms = MeanShift(bandwidth=best_bd)
    best_ms.fit(gt_gaze)

    gt_strings = []
    subjects = []
    for gt_scanpath in gt_scanpaths:
        gt_string = scanpath2clusters(best_ms, gt_scanpath)
        gt_strings.append(gt_string)
        subjects.append(gt_scanpath['subject'])

    return best_ms, gt_strings, subjects


fixation_root = './data/coco_freeview/'
processed_root = '/home/OSIE/processed'

json_data_train = os.path.join(fixation_root, 'COCOFreeView_fixations_train.json')
json_data_validation = os.path.join(fixation_root, 'COCOFreeView_fixations_validation.json')
json_data_test = os.path.join(fixation_root, 'COCOFreeView_fixations_test.json')

fixations = []

with open(json_data_train, "r") as f:
    fixations += json.load(f)

with open(json_data_validation, "r") as f:
    fixations += json.load(f)
    
with open(json_data_test, "r") as f:
    fixations += json.load(f)

target_height = 384
target_width = 512

raw_height = 1050
raw_width = 1680

data_dict = {}
for scanpath in fixations:
    if scanpath['split'] != 'test':
        continue
    key = '{}-{}'.format(scanpath['split'], scanpath['name'][:-4])
    
    scanpath["X"] = (np.array(scanpath["X"]) / raw_width * target_width).tolist()
    scanpath["Y"] = (np.array(scanpath["Y"]) / raw_height * target_height).tolist()
    
    if len(scanpath["X"]) < 3:
        for idx in range(3 - len(scanpath["X"])):
            scanpath["X"].append(1)
            scanpath["Y"].append(1)
            scanpath["T"].append(1)
    data_dict.setdefault(key, []).append(scanpath)


clusters = {}
for key, value in tqdm(data_dict.items()):
    best_ms, gt_strings, subjects = compute_clusters(value)
    strings = {k: v for k, v in zip(subjects, gt_strings)}
    clusters[key] = {
        "strings": strings,
        "cluster": best_ms
    }
    
    times = []
    for sample in value:
        len_x = len(sample['X'])
        times.append(sample['T'][:len_x])
        
    clusters[key]['times'] = times

np.save(join('clusters_coco_freeview_512_384.npy'), clusters, allow_pickle=True)

