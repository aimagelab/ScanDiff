import os
from omegaconf import DictConfig, OmegaConf
import rootutils
import hydra
from typing import Dict, List

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.components.collator import Collator
from lightning.pytorch.loggers import Logger

from src.model.components.model import DiT
from src.utils.create_diffusion import create_diffusion
from src.utils.instantiators import (
    instantiate_datasets,
    instantiate_collators,
    instantiate_loggers,
    instantiate_callbacks,
)

from src.utils.utils import extras
from src.utils.logging_utils import log_hyperparameters

from src.utils.train_utils import seed_everything
from src.utils.pylogger import RankedLogger
from src.evaluator import Evaluator
from src.data.unified_datamodule import UnifiedDataModule
from src.model.components.step_sample import create_named_schedule_sampler

log = RankedLogger(__name__, rank_zero_only=True)

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    extras(cfg)

    log.info("Instantiating callbacks...")
    callbacks: List = instantiate_callbacks(cfg.get("callbacks"))
    callbacks = callbacks[0]  # return only the model checkpoint callback

    log.info("Instantiating datasets...")
    datasets: Dict = instantiate_datasets(cfg.get("data"))
    collators: Dict = instantiate_collators(cfg.get("data"))

    log.info("Instantiating datamodule...")
    to_dict_config: Dict = OmegaConf.to_container(cfg.data)
    to_dict_config.pop("train_datasets", None)
    to_dict_config.pop("val_datasets", None)
    to_dict_config.pop("test_datasets", None)
    to_dict_config.pop("train_collators", None)
    to_dict_config.pop("val_collators", None)
    to_dict_config.pop("test_collators", None)
    datamodule = UnifiedDataModule(
        train_datasets=datasets["train_datasets"],
        val_datasets=datasets["val_datasets"],
        test_datasets=datasets["test_datasets"],
        train_collators=collators["train_collators"],
        val_collators=collators["val_collators"],
        test_collators=collators["test_collators"],
        **to_dict_config,
    )

    model = hydra.utils.instantiate(cfg.model)
    diffusion = create_diffusion(cfg,
        timestep_respacing="", diffusion_steps=cfg.get("diffusion").num_timesteps, 
        noise_schedule=cfg.get("diffusion").noise_schedule, predict_xstart=cfg.get("diffusion").predict_xstart
    )  # default: 1000 steps, linear noise schedule

    sampler = create_named_schedule_sampler(cfg.get('diffusion').sampler, diffusion)
    
    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    trainer = hydra.utils.instantiate(
        cfg.trainer, model_checkpoint=callbacks, loggers=logger, schedule_sampler=sampler
    )
    

    if cfg.get("test"):
        log.info("Instantiating evaluator...")
        evaluator = hydra.utils.instantiate(cfg.evaluation, datamodule=datamodule, _target_=Evaluator)
        trainer.evaluator = evaluator
        trainer.model = model
    else:
        evaluator = None

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters...")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        trainer.train(
            model,
            diffusion,
            datamodule,
            ckpt_path=cfg.get("ckpt_path"),
            evaluator=evaluator,
        )

    if cfg.get("test"):
        trainer.test(diffusion, ckpt_path=cfg.get("ckpt_path"))


if __name__ == "__main__":
    main()
