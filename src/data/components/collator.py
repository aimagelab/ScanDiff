from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np

class Collator:
    def __init__(self, max_len=40, use_abs_coords: bool = True, time_in_ms: bool = False):
        self.PAD = [-3, -3, -3]
        self.max_len = max_len
        self.use_abs_coords = use_abs_coords
        self.time_in_ms = time_in_ms

    def __call__(self, batch):
        batch_imgs = []
        tasks = []
        task_embeddings = []
        batch_tgt_x = []
        batch_tgt_y = []
        batch_tgt_t = []
        filenames = []
        dataset_names = []
        original_sizes = []
        num_viewers = []
        question_ids = []
        
        for s in batch:
            if s['scanpath'] is None:
                continue
            
            s['scanpath'] = s['scanpath'][:self.max_len] # truncate ground truth scanpaths to max_len
            
            batch_imgs.append(s['img'].unsqueeze(0))
            batch_tgt_x.append(s['scanpath'][:,0])
            batch_tgt_y.append(s['scanpath'][:,1])
            batch_tgt_t.append(s['scanpath'][:,2])
            tasks.append(s['task'])
            
            if 'task_embedding' in s: #for the visual search task embed the task with textual embedding
                task_embeddings.append(s['task_embedding'].unsqueeze(0))
            
            filenames.append(s['img_filename'])
            dataset_names.append(s['dataset'])
            original_sizes.append(s['original_img_size'])
            num_viewers.append(s['num_viewers'])
            question_ids.append(s['question_id'])
            
        batch_imgs = torch.cat(batch_imgs, dim=0)
        
        if len(task_embeddings) == 0: #if there are no task embeddings for visual search task
            batch_task_embeddings  = None
        else:
            batch_task_embeddings = torch.cat(task_embeddings, dim=0)
        
        batch_tgt_x.append(torch.zeros(self.max_len))
        batch_tgt_y.append(torch.zeros(self.max_len))
        batch_tgt_t.append(torch.zeros(self.max_len))
        
        batch_tgt_x = pad_sequence(batch_tgt_x, batch_first=True, padding_value=self.PAD[0])
        batch_tgt_y = pad_sequence(batch_tgt_y, batch_first=True, padding_value=self.PAD[1])
        batch_tgt_t = pad_sequence(batch_tgt_t, batch_first=True, padding_value=self.PAD[2])
        
        scanpath = torch.stack([batch_tgt_x, batch_tgt_y, batch_tgt_t], dim=2)[:-1]
            
        padding_mask = torch.where(scanpath[:,:,0] == self.PAD[0], 0, 1) # 0 if it is padding, 1 otherwise
        
        return {'img_filename': filenames, 'original_img_size': original_sizes, 'img': batch_imgs, 'scanpath': scanpath,
                'padding_mask': padding_mask, 'task': tasks, 'task_embedding': batch_task_embeddings,
                'num_viewers': num_viewers, 'dataset': dataset_names, 'question_id': question_ids}