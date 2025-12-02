#!/bin/bash

python src/eval.py -m trainer=gpu seed=0 data/train_datasets=[osie] \
    data/test_datasets=[osie] \
    evaluation.data_to_extract=['preds','qualitatives','metrics'] \
    data.num_workers=8 callbacks=default \
    tags=[scandiff_evaluation_freeview] \
    ckpt_path=./checkpoints/scandiff_freeview.pth \
    evaluation.eval_root_path=scandiff_freeviewing_evaluation/ \
    evaluation.metrics_to_compute=['multi_match','scan_match','scan_match_no_dur','sequence_score','sequence_score_time','kld','diversity_sequence_score','diversity_sequence_score_time'] \
    diffusion_class=spaced_diffusion \
    slurm=null
