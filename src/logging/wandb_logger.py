import wandb
from typing import Dict

class WandbLogger:
    def __init__(self, id, save_dir, offline, project, entity, name) -> None:
        self.id = id
        self.save_dir = save_dir
        self.offline = offline
        self.project = project
        self.entity = entity
        self.name = name # the name of a run
    
    def setup(self, config: Dict):
        
        
        if self.offline:
            mode = "offline"
        else:
            mode = "online"
            wandb.login()
        
        wandb.init(
            project=self.project,
            config=config,
            name=self.name,
            resume="allow",
            id=self.id,
            mode=mode,
        )
        
    def log(self, data: Dict):
        assert isinstance(data, dict), "Data should be of type Dict"
        wandb.log(data)
        
    def log_hyperparams(self, config):
        wandb.config.update(config)