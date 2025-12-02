import gzip
import os

import numpy as np
from tqdm import tqdm
from pathlib import Path
from src.gazetools.utils import nw_matching, scanpath2categories, scanpath2clusters
import itertools
import copy


def compute_SS(preds, clusters, truncate=16, reduce="mean", print_clusters=False, is_validation=False, threshold_no_dur=None):
    results = []
    global_scores = []
    covered_human_scanpaths = {}
    
    split = 'validation' if is_validation else 'test'
    for scanpath in tqdm(preds):
        dataset = scanpath['dataset']
        if dataset in ['cocosearch18_tp', 'cocosearch18_ta']: # manage the case of potted plant and stop sign
            scanpath['task'] = scanpath['task'].replace('_', ' ')            
            key = "{}-{}-{}-{}".format(split, scanpath['condition'], scanpath['task'], scanpath['name'])
        elif dataset == 'aird':
            key = "{}-{}-{}".format(split, scanpath['question_id'], scanpath['name'])
        elif dataset in ['osie','coco_freeview','mit1003']:
            key = "{}-{}".format(split, scanpath['name'])
        else:
            raise NotImplementedError('Dataset not implemented')
            
        ms = clusters[key]
        strings = ms["strings"]
        cluster = ms["cluster"]

        pred = scanpath2clusters(cluster, scanpath)
        scores = []
        for subj, gt in strings.items():
            
            if key + "-" + str(subj) not in covered_human_scanpaths:
                covered_human_scanpaths[key + "-" + str(subj)] = 0
            
            if len(gt) > 0:
                pred = pred[:truncate] if len(pred) > truncate else pred
                gt = gt[:truncate] if len(gt) > truncate else gt
                if print_clusters:
                    print(pred, gt)

                if len(pred) == 0:
                    print('WARNING: empty prediction')
                if len(gt) == 0:
                    print('WARNING: empty ground truth')
                score = nw_matching(pred, gt)
                scores.append(score)
                
                if threshold_no_dur and score > threshold_no_dur:
                    covered_human_scanpaths[key + "-" + str(subj)] += 1
                
                
                global_scores.append(score)

        result = {}
        result["name"] = scanpath["name"]
        if reduce == "mean":
            result["score"] = np.array(scores).mean()
        elif reduce == "max":
            result["score"] = max(scores)
        else:
            raise NotImplementedError
        results.append(result)
        
    if threshold_no_dur:
        num_covered = 0
        for key in covered_human_scanpaths:
            if covered_human_scanpaths[key] > 0:
                num_covered += 1
        
        scanpath_recall = num_covered / len(covered_human_scanpaths)
    else:
        scanpath_recall = -1 # default value    
    
    return results, global_scores, scanpath_recall, covered_human_scanpaths


def compute_SSS(preds, gts, segmentation_map_dir, truncate=16, reduce="mean", print_clusters=False, is_validation=False):
    results = []
    global_scores = []
    
    preds_copy = copy.deepcopy(preds)
    gts_copy = copy.deepcopy(gts)
    
    for scanpath in tqdm(preds_copy):
        # NB: at this point predictions are in 512x384. We need to convert it back to 512x320.
        pred = copy.deepcopy(scanpath)
        pred['X'] = (np.array(pred['X']) / 512 * 512).tolist()
        pred['Y'] = (np.array(pred['Y']) / 384 * 320).tolist()
        
        with gzip.GzipFile(Path(segmentation_map_dir, scanpath['name'] + '.npy.gz'), "r") as r:
            segmentation_map = np.load(r, allow_pickle=True)
            r.close()
        
        pred = scanpath2categories(segmentation_map, pred)
        gt_scanpaths = gts_copy[(scanpath['name']+'.jpg', scanpath['task'].replace(' ', '_'))]['scanpaths']
        
        pred_strings = pred[:truncate] if len(pred) > truncate else pred
        pred_noT = [i[0] for i in pred_strings]
        
        scores = []
        for gt_scanpath in gt_scanpaths:
            gt = copy.deepcopy(gt_scanpath)
            gt['X'] = (np.array(gt['X']) / 512 * 512).tolist()
            gt['Y'] = (np.array(gt['Y']) / 384 * 320).tolist()
            
            gt = scanpath2categories(segmentation_map, gt)
            gt = gt[:truncate] if len(gt) > truncate else gt
            gt_noT = [i[0] for i in gt]

            if len(pred) == 0:
                print('WARNING: empty prediction')
            if len(gt) == 0:
                print('WARNING: empty ground truth')                

            if len(gt_noT) > 0 and len(pred_noT) > 0:
                score = nw_matching(pred_noT, gt_noT)
            else:
                score = 0
                
            scores.append(score)
            global_scores.append(score)

        result = {}
        result["name"] = scanpath["name"]
        if reduce == "mean":
            result["score"] = np.array(scores).mean()
        elif reduce == "max":
            result["score"] = max(scores)
        else:
            raise NotImplementedError
        results.append(result)
    return results, global_scores

def compute_SSS_Time(preds, gts, segmentation_map_dir, truncate=16, reduce="mean", print_clusters=False, is_validation=False, tempbin=50):
    results = []
    global_scores = []
    split = 'validation' if is_validation else 'test'
    
    preds_copy = copy.deepcopy(preds)
    gts_copy = copy.deepcopy(gts)
    
    for scanpath in tqdm(preds_copy):
        # NB: at this point predictions are in 512x384. We need to convert it back to 512x320.
        pred = copy.deepcopy(scanpath)
        pred['X'] = (np.array(pred['X']) / 512 * 512).tolist()
        pred['Y'] = (np.array(pred['Y']) / 384 * 320).tolist()
        
        with gzip.GzipFile(Path(segmentation_map_dir, scanpath['name'] + '.npy.gz'), "r") as r:
            segmentation_map = np.load(r, allow_pickle=True)
            r.close()
        
        pred = scanpath2categories(segmentation_map, pred)
        gt_scanpaths = gts_copy[(scanpath['name']+'.jpg', scanpath['task'].replace(' ', '_'))]['scanpaths']
        
        pred_strings = pred[:truncate] if len(pred) > truncate else pred
        pred_T = []
        for p in pred_strings:
            pred_T.extend([p[0] for _ in range(int(p[1]/tempbin))])
        
        scores = []
        for gt_scanpath in gt_scanpaths:
            gt = copy.deepcopy(gt_scanpath)
            gt['X'] = (np.array(gt['X']) / 512 * 512).tolist()
            gt['Y'] = (np.array(gt['Y']) / 384 * 320).tolist()
            
            gt = scanpath2categories(segmentation_map, gt)
            gt_T = []

            gt = gt[:truncate] if len(gt) > truncate else gt
            for g in gt:
                gt_T.extend([g[0] for _ in range(int(g[1]/tempbin))])

            if len(pred) == 0:
                print('WARNING: empty prediction')
            if len(gt) == 0:
                print('WARNING: empty ground truth')                

            if len(gt_T) > 0 and len(pred_T) > 0:
                score = nw_matching(pred_T, gt_T)
            else:
                score = 0
                
            scores.append(score)
            global_scores.append(score)

        result = {}
        result["name"] = scanpath["name"]
        if reduce == "mean":
            result["score"] = np.array(scores).mean()
        elif reduce == "max":
            result["score"] = max(scores)
        else:
            raise NotImplementedError
        results.append(result)
    return results, global_scores

def compute_self_SSS(preds, gts, segmentation_map_dir, truncate=100, reduce="mean", print_clusters=False, is_validation=False):
    results = []
    global_scores = []
    
    for key in tqdm(gts):
        img_name = key[0]
        scanpaths = gts[key]
        scanpaths_combination = list(itertools.combinations(scanpaths['scanpaths'], 2))
        
        with gzip.GzipFile(Path(segmentation_map_dir, Path(img_name).stem + '.npy.gz'), "r") as r:
            segmentation_map = np.load(r, allow_pickle=True)
            r.close()
        
        scores = []
        for comb in scanpaths_combination:
            scan1, scan2 = comb[0], comb[1]
        
            gt1 = copy.deepcopy(scan1)
            gt2 = copy.deepcopy(scan2)
            
            gt1['X'] = (np.array(gt1['X']) / 512 * 512).tolist()
            gt1['Y'] = (np.array(gt1['Y']) / 384 * 320).tolist()
            
            gt2['X'] = (np.array(gt2['X']) / 512 * 512).tolist()
            gt2['Y'] = (np.array(gt2['Y']) / 384 * 320).tolist()
            
            gt_string_1 = scanpath2categories(segmentation_map, gt1)
            gt_string_2 = scanpath2categories(segmentation_map, gt2)
            
            gt_string_1 = gt_string_1[:truncate] if len(gt_string_1) > truncate else gt_string_1
            gt_string_2 = gt_string_2[:truncate] if len(gt_string_2) > truncate else gt_string_2

            pred_noT = [i[0] for i in gt_string_1]
            gt_noT = [i[0] for i in gt_string_2]

        
            if len(gt_string_1) == 0:
                print('WARNING: empty prediction')
            if len(gt_string_2) == 0:
                print('WARNING: empty ground truth')
            
            if len(pred_noT) > 0 and len(gt_noT) > 0:
                score = nw_matching(pred_noT, gt_noT)
            else:
                score = 0
            
            scores.append(score)
            global_scores.append(score)

        result = {}
        result["name"] = Path(img_name).stem
        if reduce == "mean":
            result["score"] = np.array(scores).mean()
        elif reduce == "max":
            result["score"] = max(scores)
        else:
            raise NotImplementedError
        results.append(result)
    
    return results, global_scores

def compute_self_SSS_Time(preds, gts, segmentation_map_dir, truncate=16, reduce="mean", print_clusters=False, is_validation=False, tempbin=50):
    results = []
    global_scores = []
    split = 'validation' if is_validation else 'test'
    
    for key in tqdm(gts):
        img_name = key[0]
        scanpaths = gts[key]
        scanpaths_combination = list(itertools.combinations(scanpaths['scanpaths'], 2))
        
        with gzip.GzipFile(Path(segmentation_map_dir, Path(img_name).stem + '.npy.gz'), "r") as r:
            segmentation_map = np.load(r, allow_pickle=True)
            r.close()
        
        scores = []
        for comb in scanpaths_combination:
            scan1, scan2 = comb[0], comb[1]
        
            gt1 = copy.deepcopy(scan1)
            gt2 = copy.deepcopy(scan2)
            
            gt1['X'] = (np.array(gt1['X']) / 512 * 512).tolist()
            gt1['Y'] = (np.array(gt1['Y']) / 384 * 320).tolist()
            
            gt2['X'] = (np.array(gt2['X']) / 512 * 512).tolist()
            gt2['Y'] = (np.array(gt2['Y']) / 384 * 320).tolist()
            
            gt_string_1 = scanpath2categories(segmentation_map, gt1)
            gt_string_2 = scanpath2categories(segmentation_map, gt2)
            
            gt_string_1 = gt_string_1[:truncate] if len(gt_string_1) > truncate else gt_string_1
            gt_string_2 = gt_string_2[:truncate] if len(gt_string_2) > truncate else gt_string_2

            pred_T = [] # this name to differentiate between the other gt name
            for p in gt_string_1:
                pred_T.extend([p[0] for _ in range(int(p[1]/tempbin))])
            
            gt_T = []
            for g in gt_string_2:
                gt_T.extend([g[0] for _ in range(int(g[1]/tempbin))])
        
            if len(gt_string_1) == 0:
                print('WARNING: empty prediction')
            if len(gt_string_2) == 0:
                print('WARNING: empty ground truth')
            
            if len(pred_T) > 0 and len(gt_T) > 0:
                score = nw_matching(pred_T, gt_T)
            else:
                score = 0
            
            scores.append(score)
            global_scores.append(score)

        result = {}
        result["name"] = Path(img_name).stem
        if reduce == "mean":
            result["score"] = np.array(scores).mean()
        elif reduce == "max":
            result["score"] = max(scores)
        else:
            raise NotImplementedError
        results.append(result)
    
    return results, global_scores

def compute_self_SS(originals, clusters, truncate=16, reduce="mean", print_clusters=False, is_validation=False):
    results = []
    global_scores = []
    
    split = 'validation' if is_validation else 'test'
    
    self_preds = []
    for key in tqdm(clusters):
        if split not in key:
            continue
        for subj in clusters[key]["strings"]:
            self_preds.append({'key': key, 'subj': subj, 'pred': clusters[key]["strings"][subj]})
                    
    for scanpath in tqdm(self_preds):
        key = scanpath['key']
        name = key.split('-')[1]
        ms = clusters[key]
        strings = ms["strings"]
        pred = scanpath['pred']
        pred_subj = scanpath['subj']

        #pred = scanpath2clusters(cluster, scanpath)
        scores = []
        for subj, gt in strings.items():
            
            if subj == pred_subj:
                continue # do not compare a scanpath with itself when computing human consistency
            
            if len(gt) > 0:
                pred = pred[:truncate] if len(pred) > truncate else pred
                gt = gt[:truncate] if len(gt) > truncate else gt
                if print_clusters:
                    print(pred, gt)
                score = nw_matching(pred, gt)
                scores.append(score)
                global_scores.append(score)
            else:
                score = 0
    
        result = {}
        result["name"] = name
        if reduce == "mean":
            result["score"] = np.array(scores).mean()
        elif reduce == "max":
            result["score"] = max(scores)
        else:
            raise NotImplementedError
        results.append(result)
        
    return results, global_scores

def compute_DSS_Time(preds, clusters, truncate, time_dict, threshold=None, reduce="mean", print_clusters=False, tempbin=50, is_validation=False):
        modified_preds = {}
        split = 'validation' if is_validation else 'test'
        DSS_per_img = []
        
        for key in clusters:
            if split not in key:
                continue
            modified_preds[key] = []
        
        for scanpath in preds:
            dataset = scanpath['dataset']
            
            if dataset in ['cocosearch18_tp', 'cocosearch18_ta']: # manage the case of potted plant and stop sign
                scanpath['task'] = scanpath['task'].replace('_', ' ')            
                key = "{}-{}-{}-{}".format(split, scanpath['condition'], scanpath['task'], scanpath['name'])
            elif dataset == 'aird':
                key = "{}-{}-{}".format(split, scanpath['question_id'], scanpath['name'])
            elif dataset in ['osie', 'coco_freeview', 'mit1003']:
                key = "{}-{}".format(split, scanpath['name'])
            else:
                raise NotImplementedError('Dataset not implemented')
            
            modified_preds[key].append(scanpath)
                
        for key in clusters:
            if split not in key:
                continue
            
            _, all_human_vs_human = compute_self_SS_Time(None, {key: clusters[key]}, truncate=truncate, time_dict=time_dict, 
                                reduce=reduce, print_clusters=print_clusters, tempbin=tempbin, is_validation=is_validation)
            human_vs_human = np.mean(all_human_vs_human)
            
            if np.isnan(human_vs_human):
                print('Found nan score in DSS Time human vs human')
                print(key)

            _, all_human_vs_model, _, _ = compute_SS_Time(modified_preds[key], {key: clusters[key]}, truncate=truncate, time_dict=time_dict, threshold=None, 
                                reduce=reduce, print_clusters=print_clusters, tempbin=tempbin, is_validation=is_validation)
            
            human_vs_model = np.mean(all_human_vs_model)
            
            if np.isnan(human_vs_model):
                print('Found nan score in DSS Time human vs model')
                print(key)
            
            all_model_vs_model = compute_SS_Time_model_vs_model(modified_preds[key], clusters=clusters, truncate=truncate, time_dict=time_dict, threshold=threshold, 
                                reduce=reduce, print_clusters=print_clusters, tempbin=tempbin, is_validation=is_validation)

            model_vs_model = np.mean(all_model_vs_model)
            
            if np.isnan(model_vs_model):
                print('Found nan score in DSS Time model vs model')
                print(key)
            
            DSS = human_vs_model / (1 + abs(model_vs_model - human_vs_human))
            DSS_per_img.append(DSS)
            
        return np.mean(DSS_per_img), DSS_per_img, all_human_vs_human, all_human_vs_model, all_model_vs_model


def compute_DSS_no_dur(preds, clusters, truncate, threshold_no_dur=None, reduce="mean", print_clusters=False, is_validation=False):
        modified_preds = {}
        split = 'validation' if is_validation else 'test'
        DSS_per_img = []
        
        
        for key in clusters:
            if split not in key:
                continue
            modified_preds[key] = []
        
        for scanpath in preds:
            dataset = scanpath['dataset']
            
            if dataset in ['cocosearch18_tp', 'cocosearch18_ta']: # manage the case of potted plant and stop sign
                scanpath['task'] = scanpath['task'].replace('_', ' ')            
                key = "{}-{}-{}-{}".format(split, scanpath['condition'], scanpath['task'], scanpath['name'])
            elif dataset == 'aird':
                key = "{}-{}-{}".format(split, scanpath['question_id'], scanpath['name'])
            elif dataset in ['osie', 'coco_freeview', 'mit1003']:
                key = "{}-{}".format(split, scanpath['name'])
            else:
                raise NotImplementedError('Dataset not implemented')
            
            modified_preds[key].append(scanpath)
                
        for key in clusters:
            if split not in key:
                continue
            
            _, human_vs_human = compute_self_SS(None, {key: clusters[key]}, truncate=truncate,
                                reduce=reduce, print_clusters=print_clusters, is_validation=is_validation)
            human_vs_human = np.mean(human_vs_human)
            
            if np.isnan(human_vs_human):
                print('Found nan score in DSS Time human vs human')
                print(key)

            _, human_vs_model, _, _ = compute_SS(modified_preds[key], {key: clusters[key]}, truncate=truncate, threshold_no_dur=None, 
                                reduce=reduce, print_clusters=print_clusters, is_validation=is_validation)
            
            human_vs_model = np.mean(human_vs_model)
            
            if np.isnan(human_vs_model):
                print('Found nan score in DSS Time human vs model')
                print(key)
            
            model_vs_model = compute_SS_model_vs_model(modified_preds[key], clusters=clusters, truncate=truncate, threshold_no_dur=threshold_no_dur, 
                                reduce=reduce, print_clusters=print_clusters, is_validation=is_validation)

            model_vs_model = np.mean(model_vs_model)
            
            if np.isnan(model_vs_model):
                print('Found nan score in DSS Time model vs model')
                print(key)
            
            DSS = human_vs_model / (1 + abs(model_vs_model - human_vs_human))
            DSS_per_img.append(DSS)

        return np.mean(DSS_per_img), DSS_per_img


def compute_SS_model_vs_model(
    predictions, clusters, truncate, threshold_no_dur=None, reduce="mean", 
    print_clusters=False, is_validation=False):
    results = []
    global_scores = []
    split = 'validation' if is_validation else 'test'
    
    scanpath_combinations = list(itertools.combinations(predictions, 2))
    
    for scanpath1, scanpath2 in tqdm(scanpath_combinations):
        dataset = scanpath1['dataset']
        
        if dataset in ['cocosearch18_tp', 'cocosearch18_ta']: # manage the case of potted plant and stop sign
            scanpath1['task'] = scanpath1['task'].replace('_', ' ')           
            scanpath2['task'] = scanpath2['task'].replace('_', ' ') 
            key1 = "{}-{}-{}-{}".format(split, scanpath1['condition'], scanpath1['task'], scanpath1['name'])
            key2 = "{}-{}-{}-{}".format(split, scanpath2['condition'], scanpath2['task'], scanpath2['name'])
        elif dataset == 'aird':
            key1 = "{}-{}-{}".format(split, scanpath1['question_id'], scanpath1['name'])
            key2 = "{}-{}-{}".format(split, scanpath2['question_id'], scanpath2['name'])
        elif dataset in ['osie', 'coco_freeview', 'mit1003']:
            key1 = "{}-{}".format(split, scanpath1['name'])
            key2 = "{}-{}".format(split, scanpath2['name'])
        else:
            raise NotImplementedError('Dataset not implemented')
        
        ms1 = clusters[key1]
        strings1 = ms1["strings"] # ce un problema nei clusters nel senso che non per forza sono incrementali. 
        cluster1 = ms1["cluster"]
        
        ms2 = clusters[key2]
        strings2 = ms2["strings"] # ce un problema nei clusters nel senso che non per forza sono incrementali.
        cluster2 = ms2["cluster"]
        
        pred1 = scanpath2clusters(cluster1, scanpath1)
        pred2 = scanpath2clusters(cluster2, scanpath2)

        if len(pred1) > 0 and len(pred2) > 0:
            pred1 = pred1[:truncate] if len(pred1) > truncate else pred1
            pred2 = pred2[:truncate] if len(pred2) > truncate else pred2
            
            p_string1 = (
                scanpath1["T"][:truncate] if len(scanpath1["T"]) > truncate else scanpath1["T"]
            )
            
            p_string2 = (
                scanpath2["T"][:truncate] if len(scanpath2["T"]) > truncate else scanpath2["T"]
            )
            
            # pred_time1 = []
            # pred_time2 = []
            
            # for p, t_p in zip(pred1, ptime_string1):
            #     pred_time1.extend([p for _ in range(int(t_p / tempbin))])

            # for p, t_p in zip(pred2, ptime_string2):    
            #     pred_time2.extend([p for _ in range(int(t_p / tempbin))])          

            score = nw_matching(p_string1, p_string2)
        else:
            score = 0
        
        global_scores.append(score)
    
    return global_scores


def compute_SS_Time_model_vs_model(
    predictions, clusters, truncate, time_dict, threshold=None, reduce="mean", 
    print_clusters=False, tempbin=50, is_validation=False):
    results = []
    global_scores = []
    split = 'validation' if is_validation else 'test'
    
    scanpath_combinations = list(itertools.combinations(predictions, 2))
    
    for scanpath1, scanpath2 in tqdm(scanpath_combinations):
        dataset = scanpath1['dataset']
        
        if dataset in ['cocosearch18_tp', 'cocosearch18_ta']: # manage the case of potted plant and stop sign
            scanpath1['task'] = scanpath1['task'].replace('_', ' ')           
            scanpath2['task'] = scanpath2['task'].replace('_', ' ') 
            key1 = "{}-{}-{}-{}".format(split, scanpath1['condition'], scanpath1['task'], scanpath1['name'])
            key2 = "{}-{}-{}-{}".format(split, scanpath2['condition'], scanpath2['task'], scanpath2['name'])
        elif dataset == 'aird':
            key1 = "{}-{}-{}".format(split, scanpath1['question_id'], scanpath1['name'])
            key2 = "{}-{}-{}".format(split, scanpath2['question_id'], scanpath2['name'])
        elif dataset in ['osie', 'coco_freeview', 'mit1003']:
            key1 = "{}-{}".format(split, scanpath1['name'])
            key2 = "{}-{}".format(split, scanpath2['name'])
        else:
            raise NotImplementedError('Dataset not implemented')
        
        ms1 = clusters[key1]
        strings1 = ms1["strings"] # ce un problema nei clusters nel senso che non per forza sono incrementali. 
        cluster1 = ms1["cluster"]
        
        ms2 = clusters[key2]
        strings2 = ms2["strings"] # ce un problema nei clusters nel senso che non per forza sono incrementali.
        cluster2 = ms2["cluster"]
        
        pred1 = scanpath2clusters(cluster1, scanpath1)
        pred2 = scanpath2clusters(cluster2, scanpath2)

        if len(pred1) > 0 and len(pred2) > 0:
            pred1 = pred1[:truncate] if len(pred1) > truncate else pred1
            pred2 = pred2[:truncate] if len(pred2) > truncate else pred2
            
            ptime_string1 = (
                scanpath1["T"][:truncate] if len(scanpath1["T"]) > truncate else scanpath1["T"]
            )
            
            ptime_string2 = (
                scanpath2["T"][:truncate] if len(scanpath2["T"]) > truncate else scanpath2["T"]
            )
            
            pred_time1 = []
            pred_time2 = []
            
            for p, t_p in zip(pred1, ptime_string1):
                pred_time1.extend([p for _ in range(int(t_p / tempbin))])

            for p, t_p in zip(pred2, ptime_string2):    
                pred_time2.extend([p for _ in range(int(t_p / tempbin))])          

            score = nw_matching(pred_time1, pred_time2)
        else:
            score = 0
        
        global_scores.append(score)
    
    return global_scores

def compute_SS_Time(
    preds, clusters, truncate, time_dict, threshold=None, reduce="mean", print_clusters=False, tempbin=50, is_validation=False
):
    results = []
    global_scores = []
    split = 'validation' if is_validation else 'test'
    
    covered_human_scanpaths = {}
    
    # SS(M,H) / (1 + | SS(M,M) + SS (H, H)|) new metric
    # SS(M,H) / (1 + SS(M,M)) old metric version
    
    for scanpath in tqdm(preds):
        dataset = scanpath['dataset']
        if dataset in ['cocosearch18_tp', 'cocosearch18_ta']: # manage the case of potted plant and stop sign
            scanpath['task'] = scanpath['task'].replace('_', ' ')            
            key = "{}-{}-{}-{}".format(split, scanpath['condition'], scanpath['task'], scanpath['name'])
        elif dataset == 'aird':
            key = "{}-{}-{}".format(split, scanpath['question_id'], scanpath['name'])
        elif dataset in ['osie', 'coco_freeview', 'mit1003']:
            key = "{}-{}".format(split, scanpath['name'])
        else:
            raise NotImplementedError('Dataset not implemented')
        
        ms = clusters[key]
        strings = ms["strings"] # ce un problema nei clusters nel senso che non per forza sono incrementali. 
        cluster = ms["cluster"]
        
        pred = scanpath2clusters(cluster, scanpath)
        scores = []
        
        for subj, gt in strings.items():
            
            if key + "-" + str(subj) not in covered_human_scanpaths:
                covered_human_scanpaths[key + "-" + str(subj)] = 0
            
            if len(gt) > 0:
                #key = key.replace(' ', '_') #manage the case of stop sign and potted plant
                time_string = time_dict[key + "-" + str(subj)]
                
                assert len(gt) == len(time_string), 'ERROR: different length between scanpath and time'
                
                pred = pred[:truncate] if len(pred) > truncate else pred
                gtime_string = (
                    time_string[:truncate] if len(time_string) > truncate else time_string
                )
                ptime_string = (
                    scanpath["T"][:truncate] if len(scanpath["T"]) > truncate else scanpath["T"]
                )
                gt = gt[:truncate] if len(gt) > truncate else gt
                if print_clusters:
                    print(pred, gt)
                pred_time = []
                gt_time = []
                for p, t_p in zip(pred, ptime_string):
                    pred_time.extend([p for _ in range(int(t_p / tempbin))])
                for g, t_g in zip(gt, gtime_string):
                    gt_time.extend([g for _ in range(int(t_g / tempbin))])
                score = nw_matching(pred_time, gt_time)
            else:
                score = 0
            
            if threshold and score > threshold:
                covered_human_scanpaths[key + "-" + str(subj)] += 1
                
            global_scores.append(score)
            scores.append(score)
        
        result = {}
        result["name"] = scanpath["name"]
        if reduce == "mean":
            result["score"] = np.array(scores).mean()
        elif reduce == "max":
            result["score"] = max(scores)
        else:
            raise NotImplementedError
        results.append(result)
    
    if threshold:
        num_covered = 0
        for key in covered_human_scanpaths:
            if covered_human_scanpaths[key] > 0:
                num_covered += 1
        
        scanpath_recall = num_covered / len(covered_human_scanpaths)
    else:
        scanpath_recall = -1 # default value
        
    return results, global_scores, scanpath_recall, covered_human_scanpaths

def compute_self_SS_Time(originals, clusters, truncate, time_dict, reduce="mean", print_clusters=False, tempbin=50, is_validation=False):
    results = []
    global_scores = []
    
    split = 'validation' if is_validation else 'test'
    
    for key in tqdm(clusters):
        if split not in key:
            continue
        strings = clusters[key]["strings"]

        strings_combination = list(itertools.combinations(strings.keys(), 2))
        
        scores = []
        for comb in strings_combination:
            pred = strings[comb[0]]
            if len(pred) < 3:
                print('WARNING: very short predictionnnn')
            gt = strings[comb[1]]
            
            pred_subj = comb[0]
            time_scanpath = time_dict[key + "-" + str(pred_subj)]
            assert len(pred) == len(time_scanpath), 'ERROR: different length between scanpath and time'
            
            if len(gt) > 0:
                subj = comb[1]
                time_string = time_dict[key + "-" + str(subj)]
                pred = pred[:truncate] if len(pred) > truncate else pred
                gtime_string = (
                    time_string[:truncate] if len(time_string) > truncate else time_string
                )
                ptime_string = (
                    time_scanpath[:truncate] if len(time_scanpath) > truncate else time_scanpath
                )
                gt = gt[:truncate] if len(gt) > truncate else gt
                if print_clusters:
                    print(pred, gt)
                pred_time = []
                gt_time = []
                for p, t_p in zip(pred, ptime_string):
                    pred_time.extend([p for _ in range(int(t_p / tempbin))])
                for g, t_g in zip(gt, gtime_string):
                    gt_time.extend([g for _ in range(int(t_g / tempbin))])

                score = nw_matching(pred_time, gt_time)
            else:
                score = 0
            
            global_scores.append(score)
            scores.append(score)
        
        result = {}
        result["name"] = key
        if reduce == "mean":
            result["score"] = np.array(scores).mean()
        elif reduce == "max":
            result["score"] = max(scores)
        else:
            raise NotImplementedError
        results.append(result)
    
    return results, global_scores