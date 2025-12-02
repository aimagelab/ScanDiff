from typing import Optional, Tuple
import numpy as np
from typing import Dict, List
from tqdm import tqdm

from src.gazetools.utils import compute_saliency_map
from src.gazetools.metrics.visual_attention_metrics import NSS, AUC_Judd, KLdiv

def nss(saliency_map: np.ndarray, xs: np.ndarray, ys: np.ndarray):

    mean = saliency_map.mean()
    std = saliency_map.std()

    value = saliency_map[xs, ys].copy()
    value -= mean

    if std:
        value /= std

    return value.mean()


def cc(saliency_map_1: np.ndarray, saliency_map_2: np.ndarray):
    def normalize(saliency_map: np.ndarray):
        saliency_map = np.asarray(saliency_map, dtype=np.float32)
        saliency_map -= saliency_map.mean()
        std = saliency_map.std()

        if std:
            saliency_map /= std

        return saliency_map, std == 0

    smap1, constant1 = normalize(saliency_map_1.copy())
    smap2, constant2 = normalize(saliency_map_2.copy())

    if constant1 and not constant2:
        return 0.0
    else:
        return np.corrcoef(smap1.flatten(), smap2.flatten())[0, 1]


def _compute_saliency_metrics(
    first: List[dict],
    second: Optional[List[dict]] = None,
    image_size=(1680, 1050),
):
    from joblib import Parallel, delayed

    def loo_process(tr, te):
        # Leave one subject out NSS e CC metrics computation
        all_reals = np.vstack([np.column_stack([sub["X"], sub["Y"]]) for sub in tr])
        real_map = compute_saliency_map(all_reals, image_size)
        loo_x = np.clip(te["X"], 0, image_size[0] - 1)
        loo_y = np.clip(te["Y"], 0, image_size[1] - 1)
        loo_map = compute_saliency_map(np.column_stack([loo_x, loo_y]), image_size)
        nss_score = nss(real_map, loo_x.astype(np.int64), loo_y.astype(np.int64))
        cc_score = cc(real_map, loo_map)
        return nss_score, cc_score

    def loo(x):
        train = [x[:i] + x[i + 1 :] for i in range(len(x))]
        test = [x[i] for i in range(len(x))]
        return train, test

    if second is None:
        train, test = loo(first)
        results = []
        results = Parallel(n_jobs=10)(
            delayed(loo_process)(tr, te) for tr, te in zip(train, test)
        )
        results = np.array(results)
        nss_score = np.mean(results[:, 0])
        cc_score = np.mean(results[:, 1])
    else:
        all_reals = np.vstack([np.column_stack([sub["X"], sub["Y"]]) for sub in first])
        real_map = compute_saliency_map(all_reals, image_size)
        all_gen = np.vstack([np.column_stack([sub["X"], sub["Y"]]) for sub in second])
        generated_map = compute_saliency_map(all_gen, image_size)
        
        binarymatrix = np.zeros_like(real_map)
        binarymatrix[all_reals[:, 1].astype(np.int64), all_reals[:, 0].astype(np.int64)] = 1
        
        nss_score = NSS(generated_map, binarymatrix)
        cc_score = cc(real_map, generated_map)
        auc_score = AUC_Judd(generated_map, binarymatrix)
        kl_div_score = KLdiv(generated_map, real_map)
        
        if np.isnan(nss_score):
           print('Encountered a nan value!')

    return nss_score, cc_score, auc_score, kl_div_score


def compute_saliency_metrics(
    human_trajs: Dict[Tuple[str, str], List[dict]],
    model_trajs: Dict[Tuple[str, str], List[dict]],
    image_size=(1680, 1050),
):
    nsss = []
    ccs = []
    aucs = []
    kls = []
    for key in tqdm(human_trajs.keys()):
        if isinstance(key, tuple):
            stimulus, task = key
            reals = human_trajs[(stimulus, task)]
            generated = model_trajs[(stimulus, task)]
        else: # freeview case
            stimulus = key
            task = None
            reals = human_trajs[stimulus]
            generated = model_trajs[stimulus]
        
        i_w = image_size[0]
        i_h = image_size[1]
        
        if isinstance(reals, dict) and "size" in reals and "scanpaths" in reals:
            i_w, i_h = reals["size"][1], reals["size"][0]  # type: ignore
            reals = reals["scanpaths"]  # type: ignore
            generated = generated["scanpaths"]  # type: ignore
        
        image_size = (i_w, i_h)
        r = reals
        g = generated
        try:
            nss_score, cc_score, auc_score, kl_div_score = _compute_saliency_metrics(r, g, image_size)
        except:
            print('Errore in computing binary matrix. some fixations are probably out of the image')
            print('stimulus:', stimulus)
            continue
        
        if np.isnan(nss_score) or np.isnan(cc_score) or np.isnan(auc_score) or np.isnan(kl_div_score):
            print('Encountered a nan value!')
            print('stimulus:', stimulus)
            continue
        
        nsss.append(nss_score)
        ccs.append(cc_score)
        aucs.append(auc_score)
        kls.append(kl_div_score)
        
    return np.mean(nsss), np.mean(ccs), np.mean(aucs), np.mean(kls)


def compute_self_saliency_metrics(
    data: Dict[Tuple[str, str], List[dict]], image_size=(1680, 1050)
):
    nsss = []
    ccs = []
    for key in tqdm(data.keys()):
        if isinstance(key, tuple):
            stimulus, task = key
            d = data[(stimulus, task)]
        else:
            stimulus = key
            task = None
            d = data[stimulus]
        
        i_w = image_size[0]
        i_h = image_size[1]
        
        if (
            isinstance(d, dict) and "size" in d and "scanpaths" in d
        ):  # case of freeview interface
            i_w, i_h = d["size"][1], d["size"][0]  # type: ignore
            d = d["scanpaths"]  # type: ignore
        
        image_size = (i_w, i_h)
        nss_score, cc_score = _compute_saliency_metrics(d, None, image_size)
        nsss.append(nss_score)
        ccs.append(cc_score)
    return np.mean(nsss), np.mean(ccs)
