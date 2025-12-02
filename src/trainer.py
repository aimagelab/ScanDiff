import torch.utils
import torch.utils.data
from tqdm import tqdm
import torch
import numpy as np
from src.model.components.step_sample import LossAwareSampler, UniformSampler
from src.utils.pylogger import RankedLogger
from src.utils.train_utils import get_device

log = RankedLogger(__name__, rank_zero_only=True)

class Trainer:
    def __init__(
        self, default_root_dir, min_epochs, max_epochs, accelerator,
        limit_train_batches=1.0, limit_val_batches=1.0, limit_test_batches=1.0, learning_rate=1e-4,
        weight_decay=1e-2, validation_every_n_epochs=5, test_every_n_epochs=5, schedule_sampler=None, evaluator=None,
        model_checkpoint=None, deterministic=False, loggers=None
    ) -> None:
        self.default_root_dir = default_root_dir
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.deterministic = deterministic
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.limit_test_batches = limit_test_batches
        self.validation_every_n_epochs = validation_every_n_epochs
        self.test_every_n_epochs = test_every_n_epochs
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.model_checkpoint = model_checkpoint
        self.schedule_sampler = schedule_sampler
        self.evaluator = evaluator
        self.model=None
        self.loggers = loggers


    def _setup_train(self, datamodule):
        datamodule.setup()
        self.configure_optimizer(lr=self.learning_rate)
    
    def train(self, model, diffusion, datamodule, ckpt_path=None, evaluator=None):
        self.model = model
        self.evaluator = evaluator
        if evaluator is not None:
            self.evaluator.limit_test_batches = self.limit_test_batches
        
        self._setup_train(datamodule)
        
        print("Starting Training...")
        start_epoch = 0
        
        if self.model_checkpoint and ckpt_path is not None:
            self.model_checkpoint.resume_checkpoint(self.model, self.opt, ckpt_path)
            start_epoch = self.model_checkpoint.global_epoch + 1 #+1 because the global_epoch is updated after the checkpoint is saved
                                                                #see on_epoch_end() method in ModelCheckpoint class
        model.to(self.accelerator)
        model.train()

        for cur_epoch in range(start_epoch, self.max_epochs):
            self._train_one_epoch(cur_epoch, model, diffusion, datamodule)
            
            if self.model_checkpoint is not None:
                self.log({'Epoch Number': self.model_checkpoint.global_epoch})
                self.model_checkpoint.on_epoch_end(model, self.opt)
                
            if cur_epoch % self.validation_every_n_epochs == 0: #validation
                validation_metrics = self.evaluator.test(self.model, diffusion, self.model_checkpoint.global_epoch, is_validation=True)
                if validation_metrics is not None:
                    self.log(validation_metrics)
                    
            if cur_epoch % self.test_every_n_epochs == 0 and self.evaluator is not None:
                metrics = self.evaluator.test(self.model, diffusion, self.model_checkpoint.global_epoch)
                
                if metrics is not None:
                    self.log(metrics)
                
                log.info('************Computed metrics for epoch {cur_epoch}**********')
                
    def _train_one_epoch(self, i, model, diffusion, datamodule):
        print(f"Epoch number {i}")

        train_loader = datamodule.train_dataloader()
        
        for curr_step, batch in enumerate(tqdm(train_loader)):
            
            if curr_step >= len(train_loader) * self.limit_train_batches: #limit the number of loops during debugging
                break
            
            x = batch['scanpath'].to(self.accelerator)
            padding_mask = batch['padding_mask'].to(self.accelerator)
            if 'task_embedding' in batch:
                task_embedding = batch['task_embedding'].to(self.accelerator)
            else:
                task_embedding = None
            
            t, weights = self.schedule_sampler.sample(x.shape[0], get_device())
            
            model_kwargs = dict(y=batch['img'].to(self.accelerator),
                                padding_mask=padding_mask,
                                task_embedding=task_embedding,
                                ) #specify possible conditions for the diffusion model
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, loss_dict["loss"].detach()
                )

            loss = (loss_dict["loss"] * weights).mean()
            
            #adjust loss for logging
            for key in loss_dict:
                self.log({f'train/{key}': loss_dict[key].mean().item()})
                    
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            if self.model_checkpoint is not None:
                self.model_checkpoint.set_cur_step(curr_step)
                self.model_checkpoint.update_global_step()
                
    def validation(self, i, model, diffusion, datamodule):
        print(f"Running validation for epoch {i}")
        self.opt.zero_grad()
        
        with torch.no_grad():
            val_loader = datamodule.val_dataloader()
            
            all_val_losses = []
            for curr_step, batch in enumerate(tqdm(val_loader)):
                
                if curr_step >= len(val_loader) * self.limit_val_batches: #limit the number of loops during debugging
                    break
                
                x = batch['scanpath'].to(self.accelerator)
                padding_mask = batch['padding_mask'].to(self.accelerator)
                
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=self.accelerator)
                
                model_kwargs = dict(y=batch['img'].to(self.accelerator),
                                    padding_mask=padding_mask) #specify possible conditions for the diffusion model
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                all_val_losses.append(loss.item())
                self.log({'validation/loss': loss.item()})
            
            self.log({'validation/mean_loss': np.mean(all_val_losses)})
            
    def configure_optimizer(self, lr):
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        
    def log(self, log_dict):
        if self.loggers is not None:
            for logger in self.loggers:
                logger.log(log_dict)
                
    def test(self, diffusion, ckpt_path=None):
        assert self.evaluator is not None
        self.evaluator.limit_test_batches = self.limit_test_batches
        
        if ckpt_path is not None:
            self.model_checkpoint.resume_checkpoint(self.model, None, ckpt_path)
        
        self.evaluator.test(self.model, diffusion, self.model_checkpoint.global_epoch)