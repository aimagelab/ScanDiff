import torch
from pathlib import Path
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
import json



class MIT1003Dataset:
    def __init__(self, name: str, root_path: str, task: str, split: str, num_subjects: int, threshold: float = -1., threshold_no_dur: float = -1.,
                time_in_ms: bool = False, use_abs_coords: bool = True, truncate_seconds: int = 3,
                task_embeddings_file: str = 'task_embeddings.npy', img_features_dir: str = 'dinov2_base_timm_image_features') -> None:
        self.name = name
        self.root_path = root_path
        self.task = task
        self.threshold = threshold
        self.threshold_no_dur = threshold_no_dur

        if split == 'valid':
            split = 'validation'
        
        with open(Path(self.root_path, f'mit1003_fixations_{split}.json'), 'rb') as f:
            data = json.load(f)
            
        
        self.samples = {'sequences': []}
        self.samples['sequences'] = [s for s in data if s['split'] == split]
        
        self.num_subjects = num_subjects
        self.time_in_ms = time_in_ms
        self.use_abs_coords = use_abs_coords
        self.truncate_seconds = truncate_seconds
        self.img_features_dir = img_features_dir
        
        self.subject_per_img = {} # this variable also contains the number of unique images of that split
        
        for s in self.samples['sequences']:
            if s['name'] not in self.subject_per_img: # the number is not the same for all imgs because too short scanpaths are discarded.
                self.subject_per_img[s['name']] = 1
            else:
                self.subject_per_img[s['name']] += 1
                
        self.embeddings = np.load(
            open(
                Path('./data', task_embeddings_file),
                mode="rb",
            ),
            allow_pickle=True,
        ).item()
        
        self.task_embedding = torch.from_numpy(self.embeddings[""])

    def __getitem__(self, index: int):
        sample = self.samples['sequences'][index]
        img_filename = sample['name']
        
        num_viewers = self.subject_per_img[img_filename]
        original_width = sample['width']
        original_height = sample['height']
        img_features_path = Path(self.root_path, self.img_features_dir, Path(sample['name']).stem + '.pth')
        img_feats = torch.load(img_features_path).unsqueeze(0)
                
        x_coords = sample['X'] # x, y coords in osie are already in absolute terms
        y_coords = sample['Y']
        durations = sample['T'] #durations are already in ms
        
        scanpath = np.column_stack((x_coords, y_coords, durations))
        
        # convert coords in [0, 1]
        scanpath[:,0] /= original_width
        scanpath[:,1] /= original_height
    
        # put time in seconds
        scanpath[:,2] /= 1000.0
        
        scanpath = torch.from_numpy(scanpath).float()
        
        if len(scanpath) == 0:
            print('Found scanpath of zero len, discarding it from the training size...')
            scanpath = None 
        
        return {'img_filename': img_filename, 'original_img_size': (original_width, original_height), 'img': img_feats.squeeze(), 'scanpath': scanpath,
                'task': "", 'task_embedding': self.task_embedding, 'num_viewers': num_viewers, 'dataset': self.name, 'question_id': ""}

    def __len__(self) -> int:
        return len(self.samples['sequences'])