import torch
import numpy as np
import random
from tqdm import tqdm

def process_data_for_metrics_computation(datamodule, fixations, is_original=False, remove_first_fix=True):
    
    #we should convert coords back to the original size of the image
    
    scanpaths = {}
    for key in tqdm(fixations.keys()):
        if isinstance(fixations[key], dict):
            #size = fixations[key]['size'][1], fixations[key]['size'][0] # H, W. this is the original size of the image
            size = (384, 512)
            question_id = fixations[key]['question_id']
            dataset = fixations[key]['dataset']
            scanpaths[key] = {'size': size, 'scanpaths': [], 'question_id': question_id, 'dataset': dataset}
            all_scanpaths = fixations[key]['scanpaths']
        else:
            scanpaths[key] = {'scanpaths': []}
            all_scanpaths = fixations[key]
        
        original_H, original_W = (384, 512)
        
        for scanpath in all_scanpaths:
            if isinstance(scanpath, torch.Tensor):
                scanpath = scanpath.cpu().numpy() #TODO: REMOVE THIS FROM HERE. IT SHOULD BE ALREADY SAVED IN CPU.
            s_length = torch.ones(scanpath.shape[0])
            s_length[scanpath[:,0]==datamodule.test_collators[0].PAD[0]] = 0
            s_length = int(s_length.sum().item())
            
            #convert coords from (512, 320) to the original dimension
            if datamodule.use_abs_coords:
                scanpath[:, 0] = scanpath[:, 0] * original_W / 512 #TODO: fix this hardcoding
                scanpath[:, 1] = scanpath[:, 1] * original_H / 320
            else: #coords are in [0, 1]
                scanpath[:, 0] = scanpath[:, 0] * original_W
                scanpath[:, 1] = scanpath[:, 1] * original_H
                
            x = list(scanpath[remove_first_fix : s_length, 0])
            y = list(scanpath[remove_first_fix : s_length, 1])
            
            use_ms = 1 if datamodule.time_in_ms else 1000 # if time is in seconds, convert to ms
            if scanpath.shape[1] == 3:
                if is_original:
                    t = list(scanpath[remove_first_fix : s_length, 2]*use_ms)  
                else:
                    t = list(scanpath[remove_first_fix : s_length, 2]*use_ms) 
                
                dict_to_append = {'X': x, 'Y': y, 'T': t}
            else:
                dict_to_append = {'X': x, 'Y': y}
                
            scanpaths[key]['scanpaths'].append(dict_to_append)
    
    return scanpaths