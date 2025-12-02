#!/bin/bash

python src/train.py -m data/train_datasets=[coco_freeview,mit1003] \
    data/val_datasets=[mit1003] \
    data/test_datasets=[coco_freeview] \
    trainer=gpu diffusion.num_timesteps=1000 \
    data.num_workers=8 callbacks=default \
    logger=wandb slurm=null \
    tags=[scandiff_training_freeview] \
    trainer.validation_every_n_epochs=10 \
    trainer.test_every_n_epochs=10 \
    trainer.max_epochs=200 \
    train=true test=true \
    evaluation.data_to_extract=['preds','metrics','qualitatives'] \
    evaluation.metrics_to_compute=['multi_match','scan_match','scan_match_no_dur','sequence_score','sequence_score_time','kld','diversity_sequence_score','diversity_sequence_score_time'] \
    diffusion_class=spaced_diffusion \
    callbacks.model_checkpoint.every_n_epochs=10 \
    ckpt_path=null \