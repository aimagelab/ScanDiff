import torch
from pathlib import Path

from src.utils.train_utils import get_device

class ModelCheckpoint:
    def __init__(self, dirpath, filename, monitor, verbose, save_last, save_top_k, mode, auto_insert_metric_name,
                save_weights_only, every_n_train_steps, train_time_interval, every_n_epochs, save_on_train_epoch_end) -> None:
        
        self.dirpath = dirpath
        self.filename = filename
        self.monitor = monitor
        self.verbose = verbose
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.mode = mode
        self.auto_insert_metric_name = auto_insert_metric_name
        self.save_weights_only = save_weights_only
        self.every_n_train_steps = every_n_train_steps
        self.train_time_interval = train_time_interval
        self.every_n_epochs = every_n_epochs
        self.save_on_train_epoch_end = save_on_train_epoch_end
        
        self.global_step = 0
        self.cur_step = 0
        self.global_epoch = 0
        
        self._initialize_checkpoint_dir()
        
    def _initialize_checkpoint_dir(self):
        Path(self.dirpath).mkdir(parents=True, exist_ok=True)
    
    def update_global_step(self):
        self.global_step += 1
    
    def set_cur_step(self, step):
        self.cur_step = step
    
    def update_global_epoch(self):
        self.global_epoch += 1

    def save_checkpoint(self, model, optimizer, save_last=False):
        ckpt_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'global_step': self.global_step, 'global_epoch': self.global_epoch}
        if not save_last:
            torch.save(ckpt_dict, Path(self.dirpath, f"epoch_{str(self.global_epoch)}" + '.pt'))
        else:
            torch.save(ckpt_dict, Path(self.dirpath, 'last' + '.pt'))
    
    def resume_checkpoint(self, model, optimizer, ckpt_path):
        
        device = get_device()
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        model.to(device)
        
        if optimizer is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            
            # Move optimizer state to the same device
            for param in optimizer.state.keys():
                # 'param' is a model parameter, its state is a dictionary
                state = optimizer.state[param]
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            
        self.global_step = ckpt['global_step']
        self.global_epoch = ckpt['global_epoch']
    
    def on_epoch_end(self, model, optimizer):
        if self.global_epoch % self.every_n_epochs == 0:
            self.save_checkpoint(model, optimizer)
        
        if self.save_last:
            self.save_checkpoint(model, optimizer, save_last=True)
        
        self.update_global_epoch()