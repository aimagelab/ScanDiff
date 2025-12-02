import itertools
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from multimatch_gaze import docomparison


def multi_match_score(s1: dict, s2: dict, size: Tuple[int, int]):

    s1x = s1["X"]
    s1y = s1["Y"]
    if "T" not in s1:
        s1t = [100] * len(
            s1x
        )  # default duration value for models that do not predict it
    else:
        s1t = s1["T"]
    l1 = len(s1x)
    if l1 < 3:
        scanpath1 = np.ones((3, 3), dtype=np.float32)
        scanpath1[:l1, 0] = s1x
        scanpath1[:l1, 1] = s1y
        scanpath1[:l1, 2] = s1t[:l1]
    else:
        scanpath1 = np.ones((l1, 3), dtype=np.float32)
        scanpath1[:, 0] = s1x
        scanpath1[:, 1] = s1y
        scanpath1[:, 2] = s1t[:l1]
    s2x = s2["X"]
    s2y = s2["Y"]
    if "T" not in s2:
        s2t = [100] * len(
            s2x
        )  # default duration value for models that do not predict it
    else:
        s2t = s2["T"]
    l2 = len(s2x)
    if l2 < 3:
        scanpath2 = np.ones((3, 3), dtype=np.float32)
        scanpath2[:l2, 0] = s2x
        scanpath2[:l2, 1] = s2y
        scanpath2[:l2, 2] = s2t[:l2]
    else:
        scanpath2 = np.ones((l2, 3), dtype=np.float32)
        scanpath2[:, 0] = s2x
        scanpath2[:, 1] = s2y
        scanpath2[:, 2] = s2t[:l2]

    sp1 = pd.DataFrame(
        {
            "start_x": scanpath1[:, 0],
            "start_y": scanpath1[:, 1],
            "duration": scanpath1[:, 2],
        }
    ).to_records()

    sp2 = pd.DataFrame(
        {
            "start_x": scanpath2[:, 0],
            "start_y": scanpath2[:, 1],
            "duration": scanpath2[:, 2],
        }
    ).to_records()

    mm = docomparison(sp1, sp2, screensize=size)
    return mm

def compute_multi_match_score(
    human_trajs: Dict[Tuple[str, str], List[dict]],
    model_trajs: Dict[Tuple[str, str], List[dict]],
    im_w,
    im_h,
):
    """
    compute scanpath similarity using multimatch
    """
    scores = []
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

        _scores = [
            multi_match_score(r, g, size=(i_w, i_h))
            for r, g in itertools.product(reals, generated)
        ]
        scores = scores + _scores

    return scores


def compute_self_multi_match_score(data: Dict[Tuple[str, str], List[dict]], im_w, im_h):
    """
    compute scanpath similarity using multimatch
    """
    scores = []
    for key in tqdm(data.keys()):
        if isinstance(key, tuple):
            stimulus, task = key
            d = data[(stimulus, task)]
        else:
            stimulus = key
            task = None
            d = data[stimulus]

        i_w = im_w
        i_h = im_h

        if isinstance(d, dict) and "size" in d and "scanpaths" in d:
            i_w, i_h = d["size"][1], d["size"][0]  # type: ignore
            d = d["scanpaths"]  # type: ignore

        _scores = [
            multi_match_score(r, g, size=(i_w, i_h))
            for r, g in itertools.combinations(d, 2)
        ]
        scores = scores + _scores

    return scores
