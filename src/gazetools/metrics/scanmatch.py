import itertools
from typing import Optional, Tuple
from GazeParser.ScanMatch import ScanMatch
import numpy as np
from typing import Dict, List
from tqdm import tqdm


def _scan_match_score(
    scan_match: ScanMatch, first: List[dict], second: Optional[List[dict]]
):
    if second is None:
        data = itertools.combinations(first, 2)
    else:
        data = itertools.product(first, second)
    scores = []
    for r, g in data:
        
        if len(r['X']) < 3:
            r['X'] = r['X'] + [1] * (3 - len(r['X']))
            r['Y'] = r['Y'] + [1] * (3 - len(r['Y']))
            if 'T' in r:
                r['T'] = r['T'] + [1] * (3 - len(r['T']))
        
        if len(g['X']) < 3:
            g['X'] = g['X'] + [1] * (3 - len(g['X']))
            g['Y'] = g['Y'] + [1] * (3 - len(g['Y']))
            if 'T' in g:
                g['T'] = g['T'] + [1] * (3 - len(g['T']))
            
        if "T" in g:
            r = np.column_stack((r["X"], r["Y"], r["T"]))
            g = np.column_stack((g["X"], g["Y"], g["T"]))
        else:  # case of models that do not predict duration
            r = np.column_stack((r["X"], r["Y"]))
            g = np.column_stack((g["X"], g["Y"]))
        if g.shape[0] != 0 and r.shape[0] != 0:
            s1 = scan_match.fixationToSequence(r).astype(int)
            s2 = scan_match.fixationToSequence(g).astype(int)
            (score, _, _) = scan_match.match(s1, s2)
            scores.append(score)
    return scores

def scan_match_score(
    real: Dict[Tuple[str, str], List[dict]],
    generated: Dict[Tuple[str, str], List[dict]],
    stimulus_width,
    stimulus_height,
    tempbin=50,
):
    if stimulus_height > stimulus_width:
        Xbin = 14
        Ybin = 8
    else:
        Xbin = 8
        Ybin = 14

    matchObject = ScanMatch(
        Xres=stimulus_width, Yres=stimulus_height, Xbin=16, Ybin=12, TempBin=tempbin
    )

    scores = []

    for key in tqdm(real.keys()):
        if isinstance(key, tuple):
            stimulus, task = key
            r = real[(stimulus, task)]
            g = generated[(stimulus, task)]
        else:
            stimulus = key
            task = None
            r = real[stimulus]
            g = generated[stimulus]

        imw = stimulus_width
        imh = stimulus_height

        if (
            isinstance(r, dict) and "size" in r and "scanpaths" in r
        ):  # case of freeview interface
            imw, imh = r["size"][1], r["size"][0]  # type: ignore
            r = r["scanpaths"]  # type: ignore
            g = g["scanpaths"]  # type: ignore

        if imw > imh:
            Xbin = 14
            Ybin = 8
        else:
            Xbin = 8
            Ybin = 14

        try:
            matchObject = ScanMatch(
                Xres=imw, Yres=imh, Xbin=16, Ybin=12, TempBin=tempbin
            )
        except:
            print(f'Skipping image {key} due to error in ScanMatch initialization')
            continue
        
        _scores = _scan_match_score(matchObject, r, g)
        scores = scores + _scores
    return scores


def self_scan_match_score(
    data: Dict[Tuple[str, str], List[dict]],
    stimulus_width,
    stimulus_height,
    tempbin=50,
):
    scores = []
    for key in tqdm(data.keys()):
        if isinstance(key, tuple):
            stimulus, task = key
            d = data[(stimulus, task)]
        else:
            stimulus = key
            task = None
            d = data[stimulus]

        imw = stimulus_width
        imh = stimulus_height

        if (
            isinstance(d, dict) and "size" in d and "scanpaths" in d
        ):  # case of freeview interface
            imw, imh = d["size"][1], d["size"][0]  # type: ignore
            d = d["scanpaths"]  # type: ignore

        if imw > imh:
            Xbin = 14
            Ybin = 8
        else:
            Xbin = 8
            Ybin = 14

        try:
            matchObject = ScanMatch(
                Xres=imw, Yres=imh, Xbin=16, Ybin=12, TempBin=tempbin
            )
        except:
            print(f'Skipping image {key} due to error in ScanMatch initialization')
            continue
        
        _scores = _scan_match_score(matchObject, d, None)
        scores = scores + _scores
    return scores