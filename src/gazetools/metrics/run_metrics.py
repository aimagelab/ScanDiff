import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Dict, List
from tqdm import tqdm

from src.gazetools.metrics.sequence_score import (
    compute_SS, compute_SSS, 
    compute_self_SS, compute_self_SSS,
    compute_SS_Time, compute_DSS_Time, compute_DSS_no_dur, compute_self_SS_Time,
    compute_SSS_Time, compute_self_SSS_Time
)

from src.gazetools.density import (
    compute_lengths_density_chart,
    compute_durations_density_chart,
    compute_marks_density_chart,
    compute_saccades_amplitude_chart,
    compute_saccades_direction_chart,
)
from src.gazetools.metrics.kld import perform_kld
from src.gazetools.metrics.multimatch import (
    compute_multi_match_score,
    compute_self_multi_match_score,
)
from src.gazetools.metrics.saliency import (
    compute_saliency_metrics,
    compute_self_saliency_metrics,
)

from src.gazetools.metrics.scanmatch import scan_match_score, self_scan_match_score
from src.gazetools.metrics.string_edit_distance import compute_string_edit_distance, compute_self_string_edit_distance

def run_density(
    originals: Dict[str, List[dict]],
    generated: Dict[str, List[dict]],
    save_directory: str,
):

    originals_list = list(
        itertools.chain.from_iterable(map(lambda x: x[1], originals.items()))
    )
    generated_list = list(
        itertools.chain.from_iterable(map(lambda x: x[1], generated.items()))
    )

    scanpath_lengths_original = [len(originals_list[i]['X']) for i in range(len(originals_list))]
    scanpath_lengths_generated = [len(generated_list[i]['X']) for i in range(len(generated_list))]
    
    original_xs: List[float] = list(
        itertools.chain.from_iterable(
            map(
                lambda x: itertools.chain.from_iterable((map(lambda x: x["X"], x[1]))),
                originals.items(),
            )
        )
    )
    original_ys: List[float] = list(
        itertools.chain.from_iterable(
            map(
                lambda x: itertools.chain.from_iterable((map(lambda x: x["Y"], x[1]))),
                originals.items(),
            )
        )
    )
    original_ts: List[float] = list(
        itertools.chain.from_iterable(
            map(
                lambda x: itertools.chain.from_iterable((map(lambda x: x["T"], x[1]))),
                originals.items(),
            )
        )
    )

    generated_xs: List[float] = list(
        itertools.chain.from_iterable(
            map(
                lambda x: itertools.chain.from_iterable((map(lambda x: x["X"], x[1]))),
                generated.items(),
            )
        )
    )
    generated_ys: List[float] = list(
        itertools.chain.from_iterable(
            map(
                lambda x: itertools.chain.from_iterable((map(lambda x: x["Y"], x[1]))),
                generated.items(),
            )
        )
    )
    generated_ts: List[float] = list(
        itertools.chain.from_iterable(
            map(
                lambda x: itertools.chain.from_iterable((map(lambda x: x["T"], x[1]))),
                generated.items(),
            )
        )
    )

    scanpath_lengths_original = [len(originals_list[i]['X']) for i in range(len(originals_list))]
    scanpath_lengths_generated = [len(generated_list[i]['X']) for i in range(len(generated_list))]
        
    print("Generating density charts...")
    compute_lengths_density_chart(scanpath_lengths_original, scanpath_lengths_generated, os.path.join(save_directory, "density_scanpath_lengths.png"))
    
    compute_durations_density_chart(
        original_ts,
        generated_ts,
        os.path.join(save_directory, "density_durations.png"),
    )

    compute_marks_density_chart(
        (original_xs, original_ys),
        (generated_xs, generated_ys),
        os.path.join(save_directory, "density_marks.png"),
    )

    o = list(map(lambda x: (x["X"], x["Y"]), originals_list))
    g = list(map(lambda x: (x["X"], x["Y"]), generated_list))

    compute_saccades_amplitude_chart(
        o,
        g,
        os.path.join(save_directory, "density_amplitude.png"),
    )

    compute_saccades_direction_chart(
        o, g, os.path.join(save_directory, "density_direction.png")
    )

    print("Done.")


def run_metrics(
    originals: dict,
    generated: dict,
    datamodule = None,
    metrics_list: List[str] = [],
    ss_clusters = None,
    images_width=1680,
    images_height=1050,
    is_validation=False
):
    #NB: this mean values is not the correct way to compute...
    #it should compute the mean across subject of an image. Then another mean across all images. 
    #Instead of doing a single global mean. However results should not be very different. 
    #Instead the KLD i still correct. 
    metrics = pd.DataFrame()

    if "multi_match" in metrics_list or len(metrics_list) == 0:
        print("Generating Model Multi Match metrics...")
        generated_mm = compute_multi_match_score(
           originals, generated, images_width, images_height
        )
        print("Done.")

        print("Generating Human Multi Match metrics...")
        human_mm = compute_self_multi_match_score(
            originals, images_width, images_height
        )
        print("Done.")

        g_vectors = [x[0] for x in generated_mm]
        g_directions = [x[1] for x in generated_mm]
        g_lengths = [x[2] for x in generated_mm]
        g_positions = [x[3] for x in generated_mm]
        g_durations = [x[4] for x in generated_mm]
        g_mms = [np.mean(x) for x in generated_mm]

        g_vector = np.mean(g_vectors)
        g_direction = np.mean(g_directions)
        g_length = np.mean(g_lengths)
        g_position = np.mean(g_positions)
        g_duration = np.mean(g_durations)
        g_mm = np.mean(g_mms)

        metrics.loc["Model", "MM_Vector"] = g_vector  # type: ignore
        metrics.loc["Model", "MM_Direction"] = g_direction  # type: ignore
        metrics.loc["Model", "MM_Length"] = g_length  # type: ignore
        metrics.loc["Model", "MM_Position"] = g_position  # type: ignore
        metrics.loc["Model", "MM_Duration"] = g_duration  # type: ignore
        metrics.loc["Model", "MultiMatch"] = g_mm  # type: ignore

        h_vectors = [x[0] for x in human_mm]
        h_directions = [x[1] for x in human_mm]
        h_lengths = [x[2] for x in human_mm]
        h_positions = [x[3] for x in human_mm]
        h_durations = [x[4] for x in human_mm]
        h_mms = [np.mean(x) for x in human_mm]

        h_vector = np.mean(h_vectors)
        h_direction = np.mean(h_directions)
        h_length = np.mean(h_lengths)
        h_position = np.mean(h_positions)
        h_duration = np.mean(h_durations)
        h_mm = np.mean(h_mms)

        metrics.loc[f"Human", "MM_Vector"] = h_vector  # type: ignore
        metrics.loc[f"Human", "MM_Direction"] = h_direction  # type: ignore
        metrics.loc[f"Human", "MM_Length"] = h_length  # type: ignore
        metrics.loc[f"Human", "MM_Position"] = h_position  # type: ignore
        metrics.loc[f"Human", "MM_Duration"] = h_duration  # type: ignore
        metrics.loc[f"Human", "MultiMatch"] = h_mm  # type: ignore

        if "kld" in metrics_list or len(metrics_list) == 0:
            print("Calculating KDL in Multi Match metrics...")
            
            k_vector = perform_kld(h_vectors, g_vectors)
            k_direction = perform_kld(h_directions, g_directions)
            k_length = perform_kld(h_lengths, g_lengths)
            k_position = perform_kld(h_positions, g_positions)
            k_duration = perform_kld(h_durations, g_durations)
            k_mm = np.mean([k_vector, k_direction, k_length, k_position, k_duration])

            metrics.loc["KLD", "MM_Vector"] = k_vector
            metrics.loc["KLD", "MM_Direction"] = k_direction
            metrics.loc["KLD", "MM_Length"] = k_length
            metrics.loc["KLD", "MM_Position"] = k_position
            metrics.loc["KLD", "MM_Duration"] = k_duration
            metrics.loc["KLD", "MultiMatch"] = k_mm  # type: ignore

    if "scan_match_no_dur" in metrics_list or len(metrics_list) == 0:
        print("Generating Model Scan Match No Duration metrics...")
        g_scores = scan_match_score(
            real=originals,
            generated=generated,
            stimulus_width=images_width,
            stimulus_height=images_height,
            tempbin=0,
        )
        metrics.loc["Model", "ScanMatchNoDur"] = np.mean(g_scores)  # type: ignore
        print("Done.")

        print("Generating Human Scan Match No Duration metrics...")
        h_scores = self_scan_match_score(
            data=originals,
            stimulus_width=images_width,
            stimulus_height=images_height,
            tempbin=0,
        )
        metrics.loc[f"Human", "ScanMatchNoDur"] = np.mean(h_scores)  # type: ignore
        print("Done.")

        if "kld" in metrics_list or len(metrics_list) == 0:
            print("Calculating KDL in ScanMatch No Duration...")
            
            metrics.loc["KLD", "ScanMatchNoDur"] = perform_kld(h_scores, g_scores)

    if "scan_match" in metrics_list or len(metrics_list) == 0:
        print("Generating Model Scan Match metric...")
        g_scores = scan_match_score(
           real=originals,
           generated=generated,
           stimulus_width=images_width,
           stimulus_height=images_height,
           tempbin=50,
        )
        metrics.loc["Model", "ScanMatch"] = np.mean(g_scores)  # type: ignore
        print("Done.")

        print("Generating Human Scan Match metric...")
        h_scores = self_scan_match_score(
            data=originals,
            stimulus_width=images_width,
            stimulus_height=images_height,
            tempbin=50,
        )
        metrics.loc[f"Human", "ScanMatch"] = np.mean(h_scores)  # type: ignore
        print("Done.")

        if "kld" in metrics_list or len(metrics_list) == 0:
            print("Calculating KDL in ScanMatch...")
            metrics.loc["KLD", "ScanMatch"] = perform_kld(h_scores, g_scores)

    if "sed" in metrics_list or len(metrics_list) == 0:
        print('Generating Model String Edit Distance metrics...')
        sed = compute_string_edit_distance(originals, generated, images_width, images_height)
        metrics.loc["Model", "SED"] = sed  # type: ignore
        
        #print('Generating Human String Edit Distance metrics...')
        #h_scores = compute_self_string_edit_distance(originals, images_width, images_height)
        #metrics.loc["Human", "SED"] = np.mean(h_scores)  # type: ignore
        
        #if "kld" in metrics_list or len(metrics_list) == 0:
            #print("Calculating KDL in SDE...")
            #metrics.loc["KLD", "SED"] = perform_kld(h_scores, g_scores)
        print("Done.")
    
    DSS_per_img_no_dur = None
    covered_human_scanpaths_no_dur = None
    if "sequence_score" in metrics_list or len(metrics_list) == 0:
        #compute sequence score
        print('Generating Model Sequence Score metrics...')
        # adjust predictions for SS computation
        modified_generations = []
        for key, value in tqdm(generated.items()):
            for sample in value['scanpaths']:
                scanpath = {}
                
                if isinstance(key, tuple):
                    scanpath['name'] = str(Path(key[0]).stem)
                    scanpath['task'] = key[1]
                else:
                    scanpath['name'] = str(Path(key).stem)
                
                scanpath['dataset'] = value['dataset']
                
                if scanpath['dataset'] == 'cocosearch18_tp':
                    scanpath['condition'] = 'present'
                elif scanpath['dataset'] == 'cocosearch18_ta':
                    scanpath['condition'] = 'absent'
                else:
                    scanpath['condition'] = ''
                
                scanpath['question_id'] = value['question_id']

                scanpath['X'] = sample['X']
                scanpath['Y'] = sample['Y']
                if 'T' in sample:
                    scanpath['T'] = sample['T']
                modified_generations.append(scanpath)
        
        if "sequence_score" in metrics_list or len(metrics_list) == 0:
            threshold_no_dur = datamodule.test_datasets[0].threshold_no_dur
            g_sequence_score_results, g_scores, scanpath_recall, covered_human_scanpaths_no_dur = compute_SS(modified_generations, ss_clusters, truncate=16, 
                                                            is_validation=is_validation, threshold_no_dur=threshold_no_dur)
            ss_final_score = np.mean([r['score'] for r in g_sequence_score_results])
            metrics.loc["Model", "SequenceScore"] = ss_final_score
            metrics.loc["Scanpath_Recall", "SequenceScore"] = scanpath_recall
            
            h_sequence_score_results, h_scores = compute_self_SS(originals, ss_clusters, truncate=16, is_validation=is_validation)
            ss_final_score = np.mean([r['score'] for r in h_sequence_score_results])
            metrics.loc["Human", "SequenceScore"] = ss_final_score
        
        if "kld" in metrics_list or len(metrics_list) == 0:
            print("Calculating KDL in Sequence Score...")
            metrics.loc["KLD", "SequenceScore"] = perform_kld(h_scores, g_scores)
        
        if "diversity_sequence_score" in metrics_list or len(metrics_list) == 0:
            print('Generating Model Diversity Sequence Score metrics...')
            threshold = datamodule.test_datasets[0].threshold
            DSS_no_dur, DSS_per_img_no_dur = compute_DSS_no_dur(modified_generations, clusters=ss_clusters, threshold_no_dur=threshold_no_dur,
                                                                truncate=16, is_validation=is_validation)
            metrics.loc["Model", "DSS_no_dur"] = DSS_no_dur
        
        print('Done!')
        
        
    all_human_vs_human, all_human_vs_model, all_model_vs_model = None, None, None
    DSS_per_img = None
    covered_human_scanpaths = None
    if "sequence_score_time" in metrics_list or "diversity_sequence_score" in metrics_list or len(metrics_list) == 0:
        #compute sequence score
        print('Generating Model Sequence Score Time metrics...')
        #adjust predictions for SS computation
        modified_generations = []
        for key, value in tqdm(generated.items()):
            for sample in value['scanpaths']:
                scanpath = {}
                
                if isinstance(key, tuple):
                    scanpath['name'] = str(Path(key[0]).stem)
                    scanpath['task'] = key[1]
                else:
                    scanpath['name'] = str(Path(key).stem)    
            
                scanpath['dataset'] = value['dataset']
                
                if scanpath['dataset'] == 'cocosearch18_tp':
                    scanpath['condition'] = 'present'
                elif scanpath['dataset'] == 'cocosearch18_ta':
                    scanpath['condition'] = 'absent'
                else:
                    scanpath['condition'] = ''
                
                scanpath['question_id'] = value['question_id']

                scanpath['X'] = sample['X']
                scanpath['Y'] = sample['Y']
                scanpath['T'] = sample['T']
                modified_generations.append(scanpath)
                
        t_dict = {}
        split = 'validation' if is_validation else 'test'
        
        for traj in ss_clusters:
            if split not in traj:
                continue
            if len(traj.split('-')) == 4: # visual search case
                task = traj.split('-')[2]
            for idx, (subj, sample) in enumerate(ss_clusters[traj]['strings'].items()):
                name = traj.split('-')[-1]
                if len(traj.split('-')) == 4 and task != '': # visual search case
                    condition = traj.split('-')[1] # absent or present
                    key = '{}-{}-{}-{}-{}'.format(split, condition, task, name, subj) # subject count starts from 1 instead of 0 in the clusters
                else: #freeviewing case, both with and without empty string as embedding
                    if isinstance(traj, tuple): #freeview with task embedding
                        key = '{}-{}-{}'.format(split, str(Path(traj[0]).stem), subj)
                    else:
                        if len(traj.split('-')) == 3: # aird case
                            question_id = traj.split('-')[1]
                            img_id = traj.split('-')[2]
                            key = '{}-{}-{}-{}'.format(split, question_id, img_id, subj)
                        else:
                            key = '{}-{}-{}'.format(split, name, subj) # subject count starts from 1 instead of 0 in the clusters
                
                times = ss_clusters[traj]['times'][idx]
                t_dict[key] = np.array(times)        
        if "sequence_score_time" in metrics_list or len(metrics_list) == 0:
            threshold = datamodule.test_datasets[0].threshold
            g_sequence_score_results, g_scores, scanpath_recall, covered_human_scanpaths = compute_SS_Time(modified_generations, clusters=ss_clusters, threshold=threshold,
                                                                truncate=16, time_dict=t_dict, is_validation=is_validation) #review the truncate value
            ss_final_score = np.mean([r['score'] for r in g_sequence_score_results])
            metrics.loc["Model", "SequenceScore_Time"] = ss_final_score
            metrics.loc["Scanpath_Recall", "SequenceScore_Time"] = scanpath_recall
            
            h_sequence_score_results, h_scores = compute_self_SS_Time(originals=originals, clusters=ss_clusters, 
                                                                    truncate=16, time_dict=t_dict, is_validation=is_validation) #review the truncate value
            ss_final_score = np.mean([r['score'] for r in h_sequence_score_results])
            metrics.loc["Human", "SequenceScore_Time"] = ss_final_score  # type: ignore
            
            if "kld" in metrics_list or len(metrics_list) == 0:
                print("Calculating KDL in Sequence Score Time...")
                metrics.loc["KLD", "SequenceScore_Time"] = perform_kld(h_scores, g_scores)
                
        if "diversity_sequence_score_time" in metrics_list or len(metrics_list) == 0:
            print('Generating Model Diversity Sequence Score metrics...')
            threshold = datamodule.test_datasets[0].threshold
            DSS, DSS_per_img, all_human_vs_human, all_human_vs_model, all_model_vs_model = compute_DSS_Time(modified_generations, clusters=ss_clusters, threshold=threshold,
                                                                truncate=16, time_dict=t_dict, is_validation=is_validation)
            metrics.loc["Model", "DSS_Time"] = DSS
            
        print('Done!')
        
    if 'semantic_sequence_score' in metrics_list or len(metrics_list) == 0:
        #compute semantic sequence score
        print('Generating Model Semantic Sequence Score metrics...')
        # adjust predictions for SSS computation
        modified_generations = []
        for key, value in tqdm(generated.items()):
            for sample in value['scanpaths']:
                scanpath = {}
                
                if isinstance(key, tuple):
                    scanpath['name'] = str(Path(key[0]).stem)
                    scanpath['task'] = key[1]
                else:
                    scanpath['name'] = str(Path(key).stem)
                
                scanpath['dataset'] = value['dataset']
                
                if scanpath['dataset'] == 'cocosearch18_tp':
                    scanpath['condition'] = 'present'
                elif scanpath['dataset'] == 'cocosearch18_ta':
                    scanpath['condition'] = 'absent'
                else:
                    scanpath['condition'] = ''
                
                scanpath['question_id'] = value['question_id']

                scanpath['X'] = sample['X']
                scanpath['Y'] = sample['Y']
                if 'T' in sample:
                    scanpath['T'] = sample['T']
                modified_generations.append(scanpath)
                
        modified_originals = []
        for key, value in tqdm(originals.items()):
            for sample in value['scanpaths']:
                scanpath = {}
                
                if isinstance(key, tuple):
                    scanpath['name'] = str(Path(key[0]).stem)
                    scanpath['task'] = key[1]
                else:
                    scanpath['name'] = str(Path(key).stem)
                
                scanpath['dataset'] = value['dataset']
                
                if scanpath['dataset'] == 'cocosearch18_tp':
                    scanpath['condition'] = 'present'
                elif scanpath['dataset'] == 'cocosearch18_ta':
                    scanpath['condition'] = 'absent'
                else:
                    scanpath['condition'] = ''
                
                scanpath['question_id'] = value['question_id']

                scanpath['X'] = sample['X']
                scanpath['Y'] = sample['Y']
                if 'T' in sample:
                    scanpath['T'] = sample['T']
                modified_originals.append(scanpath)
        
        segmentation_maps_dir = './data/cocosearch18/SemSS/segmentation_maps'
        g_sequence_score_results, g_scores = compute_SSS(modified_generations, originals, segmentation_maps_dir, truncate=16, is_validation=is_validation)
        sss_final_score = np.mean([r['score'] for r in g_sequence_score_results])
        metrics.loc["Model", "SemanticSequenceScore"] = sss_final_score
        
        h_sequence_score_results, h_scores = compute_self_SSS(modified_originals, originals, segmentation_maps_dir, truncate=16, is_validation=is_validation)
        ss_final_score = np.mean([r['score'] for r in h_sequence_score_results])
        metrics.loc["Human", "SemanticSequenceScore"] = ss_final_score
        
        print(metrics)
        
        if "kld" in metrics_list or len(metrics_list) == 0:
            print("Calculating KDL in Semantic Sequence Score...")
            metrics.loc["KLD", "SemanticSequenceScore"] = perform_kld(h_scores, g_scores)
        
        print('Done!')
    
    if 'semantic_sequence_score_time' in metrics_list or len(metrics_list) == 0:
        #compute semantic sequence score time
        print('Generating Model Semantic Sequence Score Time metrics...')
        # adjust predictions for SSSTime computation
        modified_generations = []
        for key, value in tqdm(generated.items()):
            for sample in value['scanpaths']:
                scanpath = {}
                
                if isinstance(key, tuple):
                    scanpath['name'] = str(Path(key[0]).stem)
                    scanpath['task'] = key[1]
                else:
                    scanpath['name'] = str(Path(key).stem)
                
                scanpath['dataset'] = value['dataset']
                
                if scanpath['dataset'] == 'cocosearch18_tp':
                    scanpath['condition'] = 'present'
                elif scanpath['dataset'] == 'cocosearch18_ta':
                    scanpath['condition'] = 'absent'
                else:
                    scanpath['condition'] = ''
                
                scanpath['question_id'] = value['question_id']

                scanpath['X'] = sample['X']
                scanpath['Y'] = sample['Y']
                if 'T' in sample:
                    scanpath['T'] = sample['T']
                modified_generations.append(scanpath)
                
        modified_originals = []
        for key, value in tqdm(originals.items()):
            for sample in value['scanpaths']:
                scanpath = {}
                
                if isinstance(key, tuple):
                    scanpath['name'] = str(Path(key[0]).stem)
                    scanpath['task'] = key[1]
                else:
                    scanpath['name'] = str(Path(key).stem)
                
                scanpath['dataset'] = value['dataset']
                
                if scanpath['dataset'] == 'cocosearch18_tp':
                    scanpath['condition'] = 'present'
                elif scanpath['dataset'] == 'cocosearch18_ta':
                    scanpath['condition'] = 'absent'
                else:
                    scanpath['condition'] = ''
                
                scanpath['question_id'] = value['question_id']

                scanpath['X'] = sample['X']
                scanpath['Y'] = sample['Y']
                if 'T' in sample:
                    scanpath['T'] = sample['T']
                modified_originals.append(scanpath)
        
        segmentation_maps_dir = './data/cocosearch18/SemSS/segmentation_maps'
        g_sequence_score_results, g_scores = compute_SSS_Time(modified_generations, originals, segmentation_maps_dir, truncate=16, is_validation=is_validation)
        sss_final_score = np.mean([r['score'] for r in g_sequence_score_results])
        metrics.loc["Model", "SemanticSequenceScoreTime"] = sss_final_score

        h_sequence_score_results, h_scores = compute_self_SSS_Time(modified_originals, originals, segmentation_maps_dir, truncate=16, is_validation=is_validation)
        sss_final_score = np.mean([r['score'] for r in h_sequence_score_results])
        metrics.loc["Human", "SemanticSequenceScoreTime"] = sss_final_score
        
        print(metrics)
        
        if "kld" in metrics_list or len(metrics_list) == 0:
            print("Calculating KDL in Semantic Sequence Score TIme...")
            metrics.loc["KLD", "SemanticSequenceScoreTime"] = perform_kld(h_scores, g_scores)
        
        print('Done!')


    if 'conditional_information_gain' in metrics_list or len(metrics_list) == 0:
        pass  
    
    if "cc" in metrics_list or "nss" in metrics_list or "auc" in metrics_list or "kldiv" in metrics_list or len(metrics_list) == 0:
        print("Generating Model Saliency Metrics...")
        print('Metrics list is: ', metrics_list)
        nss, cc, auc, kldiv = compute_saliency_metrics(
            originals, generated, (images_width, images_height)
        )
        if "nss" in metrics_list or len(metrics_list) == 0:
            metrics.loc["Model", "NSS"] = nss  # type: ignore
        if "cc" in metrics_list or len(metrics_list) == 0:
            metrics.loc["Model", "CC"] = cc  # type: ignore
        if "auc" in metrics_list or len(metrics_list) == 0:
            metrics.loc["Model", "AUC"] = auc
        if "kldiv" in metrics_list or len(metrics_list) == 0:
            metrics.loc["Model", "KLDiv"] = kldiv
        print("Done.")

    print('The final metrics are:')
    print(DSS_per_img)
    print(DSS_per_img_no_dur)
    print(covered_human_scanpaths_no_dur)
    print(covered_human_scanpaths)
    print(all_human_vs_human)
    print(all_human_vs_model)
    print(all_model_vs_model)
    
    return metrics, DSS_per_img, DSS_per_img_no_dur, covered_human_scanpaths_no_dur, covered_human_scanpaths, all_human_vs_human, all_human_vs_model, all_model_vs_model