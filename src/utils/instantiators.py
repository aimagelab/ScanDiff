from typing import Dict, List

import hydra
#from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from src.callbacks.model_checkpoint import ModelCheckpoint

from src.utils import pylogger
log = pylogger.RankedLogger(__name__, rank_zero_only=True)

def instantiate_callbacks(callbacks_cfg: DictConfig) -> ModelCheckpoint:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[ModelCheckpoint] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def instantiate_datasets(dataset_cfg: DictConfig) -> Dict[str, List]:
    """Instantiates datasets from config.

    :param dataset_cfg: A DictConfig object containing dataset configurations.
    :return: A list of instantiated datasets.
    """
    datasets: Dict = {}

    if not dataset_cfg:
        log.warning("No dataset configs found! Skipping...")
        return datasets

    if not isinstance(dataset_cfg, DictConfig):
        raise TypeError("Dataset config must be a DictConfig!")
    
    time_in_ms = dataset_cfg.get("time_in_ms")
    use_abs_coords = dataset_cfg.get("use_abs_coords")
    task_embeddings_file = dataset_cfg.get("task_embeddings_file")
    
    if 'mit1003' in dataset_cfg.train_datasets or 'coco_freeview' in dataset_cfg.train_datasets:
        img_features_dir = dataset_cfg.get("img_features_dir")
    
    for key in ["train_datasets", "val_datasets", "test_datasets"]:
        datasets[key] = []
        if key not in dataset_cfg:
            continue
        if not isinstance(dataset_cfg[key], DictConfig):
            raise TypeError(
                f"Dataset config must be a DictConfig! Found {type(dataset_cfg[key])} instead."
            )

        for d in dataset_cfg[key]:
            if isinstance(dataset_cfg[key][d], DictConfig) and "_target_" in dataset_cfg[key][d]:
                log.info(f"Instantiating dataset <{dataset_cfg[key][d]._target_}>")
                
                dataset_cfg[key][d].time_in_ms = time_in_ms
                dataset_cfg[key][d].use_abs_coords = use_abs_coords
                dataset_cfg[key][d].task_embeddings_file = task_embeddings_file
                
                datasets[key].append(
                    hydra.utils.instantiate(dataset_cfg[key][d], _recursive_=True)
                )

    return datasets

def instantiate_collators(collator_cfg: DictConfig) -> Dict[str, List]:
    """Instantiates collators from config.

    :param collator_cfg: A DictConfig object containing collator configurations.
    :return: A list of instantiated collators.
    """
    collators: Dict = {}

    if not collator_cfg:
        log.warning("No collator configs found! Skipping...")
        return collators

    if not isinstance(collator_cfg, DictConfig):
        raise TypeError("collator config must be a DictConfig!")

    for key in ["train_collators", "val_collators", "test_collators"]:
        collators[key] = []
        if key not in collator_cfg:
            continue
        if not isinstance(collator_cfg[key], DictConfig):
            raise TypeError(
                f"collator config must be a DictConfig! Found {type(collator_cfg[key])} instead."
            )

        for d in collator_cfg[key]:
            if isinstance(collator_cfg[key][d], DictConfig) and "_target_" in collator_cfg[key][d]:
                log.info(f"Instantiating collator <{collator_cfg[key][d]._target_}>")
                collators[key].append(
                    hydra.utils.instantiate(collator_cfg[key][d], _recursive_=True)
                )

    return collators