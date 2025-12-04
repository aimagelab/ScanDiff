<p align="center">
  <img src="assets/iccv2025_logo.svg" alt="ICCV 2025" height="55">
</p>

<div align="center">
<img align="left" height="90" style="margin-left: 20px" src="assets/logo.png" alt="">

# Modeling Human Gaze Behavior with Diffusion Models for Unified Scanpath Prediction
# Modeling Human Gaze Behavior with Diffusion Models for Unified Scanpath Prediction

[**Giuseppe Cartella**](https://scholar.google.com/citations?hl=en&user=0sJ4VCcAAAAJ),
[**Vittorio Cuculo**](https://scholar.google.com/citations?hl=en&user=usEfqxoAAAAJ&hl=it&oi=ao),
[**Alessandro D'Amelio**](https://scholar.google.com/citations?user=chkawtoAAAAJ&hl=en&oi=ao),<br>
[**Marcella Cornia**](https://scholar.google.com/citations?hl=en&user=DzgmSJEAAAAJ),
[**Giuseppe Boccignone**](https://scholar.google.com/citations?user=LqM0uJwAAAAJ&hl),
[**Rita Cucchiara**](https://scholar.google.com/citations?hl=en&user=OM3sZEoAAAAJ)

Official implementation of "Modeling Human Gaze Behavior with Diffusion Models for Unified Scanpath Prediction", ICCV 2025 🌺

</div>

## Overview

<p align="center">
    <img src="assets/figure.jpg">
</p>

>**Abstract**: <br>
> Predicting human gaze scanpaths is crucial for understanding visual attention, with applications in human-computer interaction, autonomous systems, and cognitive robotics. While deep learning models have advanced scanpath prediction, most existing approaches generate averaged behaviors, failing to capture the variability of human visual exploration. In this work, we present ScanDiff, a novel architecture that combines diffusion models with Vision Transformers to generate diverse and realistic scanpaths. Our method explicitly models scanpath variability by leveraging the stochastic nature of diffusion models, producing a wide range of plausible gaze trajectories. Additionally, we introduce textual conditioning to enable task-driven scanpath generation, allowing the model to adapt to different visual search objectives. Experiments on benchmark datasets show that ScanDiff surpasses state-of-the-art methods in both free-viewing and task-driven scenarios, producing more diverse and accurate scanpaths. These results highlight its ability to better capture the complexity of human visual behavior, pushing forward gaze prediction research.

## Installation

### 1️⃣ Install Requirements

```bash
# Clone the repository
git clone https://github.com/aimagelab/ScanDiff.git
cd ScanDiff

# Install dependencies
conda create --name scandiff python=3.10
conda activate scandiff
pip install -r requirements.txt

# Install PyTorch and Torchvision (CUDA 12.1)
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```
---
### ⚙️ Configuration Management with Hydra
This project uses [Hydra](https://github.com/facebookresearch/hydra) to manage configurations in a flexible and composable way.
All experiment settings (e.g., data, model, diffusion parameters) are defined in YAML files inside the ```configs/``` directory.

Hydra allows to easily override or combine configuration options directly from the command line without modifying the source files.

For more details, visit the [Hydra documentation](https://hydra.cc/docs/intro/)!

---
### 2️⃣ Download Checkpoints
Download freeviewing and visual search [checkpoints](https://ailb-web.ing.unimore.it/publicfiles/ScanDiff_ICCV2025/checkpoints.zip) with the following command:
```bash
wget -O checkpoints.zip https://ailb-web.ing.unimore.it/publicfiles/ScanDiff_ICCV2025/checkpoints.zip && unzip checkpoints.zip
```
### 3️⃣ Download Data
Download [data](https://ailb-web.ing.unimore.it/publicfiles/ScanDiff_ICCV2025/data.zip) (~53 GB) with the following command. It also containes the pre-computed visual features from DINOv2.
```bash
wget -O data.zip https://ailb-web.ing.unimore.it/publicfiles/ScanDiff_ICCV2025/data.zip && unzip data.zip -d data
```
If you have disk limitations, it is also possible to download the datasets separately:

- COCOFreeView (~22 GB):
```bash
wget -O coco_freeview.zip https://ailb-web.ing.unimore.it/publicfiles/ScanDiff_ICCV2025/coco_freeview.zip && unzip coco_freeview.zip -d coco_freeview
```
- COCOSearch18 (~25 GB):
```bash
wget -O cocosearch18.zip https://ailb-web.ing.unimore.it/publicfiles/ScanDiff_ICCV2025/cocosearch18.zip && unzip cocosearch18.zip -d cocosearch18
```
- MIT1003 (~3.8 GB):
```bash
wget -O mit1003.zip https://ailb-web.ing.unimore.it/publicfiles/ScanDiff_ICCV2025/mit1003.zip && unzip mit1003.zip -d mit1003
```
- OSIE (~2.6 GB):
```bash
wget -O osie.zip https://ailb-web.ing.unimore.it/publicfiles/ScanDiff_ICCV2025/osie.zip && unzip osie.zip -d osie
```

At this point the project root should look like:
```shell
ScanDiff/
├── data/
│     └── task_embeddings.npy
|     └── osie/
|     |    └── images/
|     |    └── dinov2_base_timm_image_features/
|     |    └── clusters_osie_512_384.npy
|     |    └── processed_fixations/
|     └── mit1003/
|     |    └── images/
|     |    └── dinov2_base_timm_image_features/
|     |    └── clusters_mit1003_512_384.npy
|     |    └── mit1003_fixations_train.json
|     |    └── mit1003_fixations_validation.json
|     |    └── mit1003_fixations_test.json
|     └── coco_freeview/
|     |    └── images/
|     |    └── dinov2_base_timm_image_features/
|     |    └── clusters_coco_freeview_512_384.npy
|     |    └── COCOFreeView_fixations_train.json
|     |    └── COCOFreeView_fixations_validation.json
|     |    └── COCOFreeView_fixations_test.json
|     └── cocosearch18/
|     |    └── images_tp/
|     |    └── images_ta/
|     |    └── dinov2_base_timm_image_features_tp/
|     |    └── dinov2_base_timm_image_features_ta/
|     |    └── clusters_cocosearch18_tp_512_384.npy
|     |    └── clusters_cocosearch18_ta_512_384.npy
|     |    └── SemSS/
|     |    └── coco_search18_fixations_TP_train.json
|     |    └── coco_search18_fixations_TP_validation.json
|     |    └── coco_search18_fixations_TP_test.json
|     |    └── coco_search18_fixations_TA_train.json
|     |    └── coco_search18_fixations_TA_validation.json
|     |    └── coco_search18_fixations_TA_test.json
└── checkpoints/
      ├── scandiff_freeview.pth
      └── scandiff_visualsearch.pth
```
### 4️⃣ Quick Start!
We provide a simple ```demo.py``` script to generate scanpaths for a certain image

1. Generate scanpaths in the freeviewing setting:
```bash
python demo.py image_path=./sample_images/dog.jpg viewing_task="" checkpoint_path=./checkpoints/scandiff_freeview.pth num_output_scanpaths=10
```

2. Generate scanpaths in the visual search setting:
```bash
python demo.py image_path=./sample_images/car.jpg viewing_task="car" checkpoint_path=./checkpoints/scandiff_visualsearch.pth num_output_scanpaths=10
```

# Train the model
To train ScanDiff run the following command:
```bash
./jobs/train.sh
```
The parameters of the configuration can be modified in   the ```/configs/train.yaml``` file. 

#### Explanation of some training parameters:

- ```data/train_datasets``` -> the combination of datasets on which the model is trained.
- ```data/val_datasets``` -> the combination of datasets on which the model is validated. NB: only one dataset is supported. 
- ```data/test_datasets``` -> the combination of datasets on which the model is tested. NB: only one dataset is supported. If you want to evaluate on more datasets after training just execute different parallel runs.
- ```tags``` -> name of the experiment.
- ```evaluation.data_to_extract``` -> data to save. It includes prediction, metrics and qualitative results.
- ```evaluation.metrics_to_compute``` -> the metrics to be comnputed. NB: **'semantic_sequence_score'** and **'semantic_sequence_score_time'** can be computed only in the visual search setting. 
- ```ckpt_path``` -> If set to null it means a training from scratch. By specifying a path, the training is resumed from that checkpoint.

The default configuration trains the model in the freeviewing setting. To train it for the visual search task, you need to specify the correct training, validation and test datasets.

This codebase also supports SLURM. If you want to run your experiments within a SLURM environment just set ```slurm=default``` in the ```train.yaml``` file and specify the **partition** and **account** parameters in the ```configs/slurm/default.yaml``` file.

---
# Evaluate the model
To test the ScanDiff model run the following command:
```bash
./jobs/eval.sh
```
The parameters of the configuration can be modified in the ```/configs/eval.yaml``` file.

### Explanation of some evaluation parameters:
- ```ckpt_path``` -> The checkpoint to test.
- ```evaluation.eval_root_path``` -> Directory path where to save the predictions, metrics or qualitatives.

---
### 📝 TODO

- [X] Release data and train-val-test splits.
- [X] Release **training code** for ScanDiff.
- [X] Release **evaluation scripts** for benchmark comparisons.

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{cartella2025modeling,
  title={Modeling Human Gaze Behavior with Diffusion Models for Unified Scanpath Prediction},
  author={Cartella, Giuseppe and Cuculo, Vittorio and D'Amelio, Alessandro and Cornia, Marcella and Boccignone, Giuseppe and Cucchiara, Rita},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```

## Acknowledgments
Thanks to [Lukas Ashleve](https://github.com/ashleve) for his [template](https://github.com/ashleve/lightning-hydra-template) that inspired this repository.
