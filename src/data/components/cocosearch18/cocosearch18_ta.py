import torch
from pathlib import Path
from PIL import Image
import numpy as np
from pathlib import Path
import json

class COCOSearch18TargetAbsentDataset:
    def __init__(self, name: str, root_path: str, task: str, split: str, num_subjects: int, threshold: float = -1., threshold_no_dur: float = -1.,
                time_in_ms: bool = False, use_abs_coords: bool = True, truncate_seconds: int = 3, 
                task_embeddings_file: str = 'task_embeddings.npy', img_features_dir: str = 'dinov2_base_timm_image_features_ta') -> None:
        self.name = name
        self.root_path = root_path
        self.task = task
        self.threshold = threshold
        self.threshold_no_dur = threshold_no_dur
        
        if split == 'valid':
            split = 'validation'
        
        if split in ['train', 'validation']:
            with open(Path(self.root_path, f'coco_search18_fixations_TA_trainval.json'), 'rb') as f: # here each scanpath starts from the second fixation. The first is discarded
                self.samples = json.load(f)
                
            # only consider the right split
            if split == 'train':
                self.samples = [s for s in self.samples if s['split'] == 'train']
            else:
                self.samples = [s for s in self.samples if s['split'] == 'validation']
                
        else: # test split
            with open(Path(self.root_path, f'coco_search18_fixations_TA_test.json'), 'rb') as f: # here each scanpath starts from the second fixation. The first is discarded
                self.samples = json.load(f)
            
        self.task_embeddings = np.load(
            open(
                Path('./data', task_embeddings_file),
                mode="rb",
            ),
            allow_pickle=True,
        ).item()
            
        self.num_subjects = num_subjects
        self.time_in_ms = time_in_ms
        self.use_abs_coords = use_abs_coords
        self.truncate_seconds = truncate_seconds
        self.img_features_dir = img_features_dir
        
        self.subject_per_img_per_task = {} # this variable also contains the number of unique images of that split
        
        for s in self.samples:
            task = s['task']
            if task == 'potted plant':
                task = 'potted_plant'
            
            if task == 'stop sign':
                task = 'stop_sign'
            
            if f'{s["name"]}_{task}' not in self.subject_per_img_per_task: # the number is not the same for all imgs because too short scanpaths are discarded.
                self.subject_per_img_per_task[f'{s["name"]}_{task}'] = 1
            else:
                self.subject_per_img_per_task[f'{s["name"]}_{task}'] += 1
    
    def __getitem__(self, index: int):
        sample = self.samples[index]
        img_filename = sample['name']
        task_string = sample['task']
        
        task_embedding = torch.from_numpy(self.task_embeddings[task_string]) # embedding extracted with roberta
        
        #treat specific edge cases
        if task_string == 'potted plant':
            task_string = 'potted_plant'
            
        if task_string == 'stop sign':
            task_string = 'stop_sign'
        
        num_viewers = self.subject_per_img_per_task[f'{img_filename}_{task_string}']
        original_img_size = Image.open(Path(self.root_path, 'images_ta', task_string, Path(img_filename))).size
        img_features_path = Path(self.root_path, self.img_features_dir, task_string, Path(img_filename).stem + '.pth')
        img_feats = torch.load(img_features_path).unsqueeze(0)
        
        #if self.split == 'train' or self.split == 'val':
        num_fixations = len(sample['X'])
        x = sample['X']
        y = sample['Y']
        t = sample['T'][:num_fixations]
        
        scanpath = np.stack((x,y,t)).T # coords are in range [512x320]
        #coords = (coords + 1) / 2 #put in range [0,1]
        
        if not self.use_abs_coords:
            # coords in the original json file for TA are in range [1680, 1050]
            #put coords in [0,1]
            scanpath[:,0] /= 1680
            scanpath[:,1] /= 1050
        
        #durations = np.hstack((sample['arrival_times'], sample['t_end']))[1:] - sample['arrival_times']
        #durations = np.reshape(durations, (-1, 1))
        #scanpath = np.hstack((coords, durations))
        scanpath = torch.from_numpy(scanpath).float() # gt duration is in seconds
        

        if not self.time_in_ms: # by default time is in milliseconds in the annotations
            scanpath[:,2] /= 1000.0
        
        return {'img_filename': img_filename, 'original_img_size': original_img_size, 'img': img_feats.squeeze(), 'scanpath': scanpath, 'task': task_string, 
                'task_embedding': task_embedding, 'num_viewers': num_viewers, 'dataset': self.name, 'question_id': ""}

    def __len__(self) -> int:
        return len(self.samples)