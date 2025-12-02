from typing import Dict, List, Tuple
from tqdm import tqdm
import itertools
from .visual_attention_metrics import string_edit_distance
import numpy as np

def compute_string_edit_distance(
    human_trajs: Dict[Tuple[str, str], List[dict]],
    model_trajs: Dict[Tuple[str, str], List[dict]],
    im_w,
    im_h,
):
    """
    compute scanpath similarity using string edit distance
    """
    all_scores = []
    mean_scores_per_img = []
    for key in tqdm(human_trajs.keys()):
        if isinstance(key, tuple):
            stimulus, task = key
            reals = human_trajs[(stimulus, task)]
            generated = model_trajs[(stimulus, task)]
        else:  # freeview case
            stimulus = key
            task = None
            reals = human_trajs[stimulus]
            generated = model_trajs[stimulus]

        i_w = im_w
        i_h = im_h

        if (
            isinstance(reals, dict) and "size" in reals and "scanpaths" in reals
        ):  # case of freeview interface
            i_w, i_h = reals["size"][1], reals["size"][0]  # type: ignore
            reals = reals["scanpaths"]  # type: ignore
            generated = generated["scanpaths"]  # type: ignore

        _scores = []
        for r, g in itertools.product(reals, generated):
            stimulus = np.zeros((i_h, i_w, 3), dtype=np.float32)
            gt_vector = np.array([r["X"], r["Y"]]).T
            pred_vector = np.array([g["X"], g["Y"]]).T
            sed = string_edit_distance(stimulus, gt_vector, pred_vector, n=10)
            _scores.append(sed)
            
        mean_scores_per_img.append(np.mean(_scores))
        all_scores = all_scores + _scores

    return np.mean(mean_scores_per_img)


def compute_self_string_edit_distance(data: Dict[Tuple[str, str], List[dict]], im_w, im_h):
    mean_scores_per_img = []
    all_scores = []
    for key in tqdm(data.keys()):
        if isinstance(key, tuple):
            stimulus, task = key
            d = data[(stimulus, task)]
        else:  # freeview case
            stimulus = key
            task = None
            d = data[stimulus]

        i_w = im_w
        i_h = im_h

        if (
            isinstance(d, dict) and "size" in d and "scanpaths" in d
        ):  # case of freeview interface
            i_w, i_h = d["size"][1], d["size"][0]  # type: ignore
            d = d["scanpaths"]  # type: ignore

        _scores = []
        for r, g in itertools.combinations(d, 2):
            stimulus = np.zeros((i_h, i_w, 3), dtype=np.float32)
            gt_vector = np.array([r["X"], r["Y"]]).T
            pred_vector = np.array([g["X"], g["Y"]]).T
            sde = string_edit_distance(stimulus, gt_vector, pred_vector, n=10)
            _scores.append(sde)
        
        mean_scores_per_img.append(np.mean(_scores))
        all_scores = all_scores + _scores

    return np.mean(mean_scores_per_img)

