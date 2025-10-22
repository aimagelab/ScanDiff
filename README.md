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
wget -O checkpoints.zip https://ailb-web.ing.unimore.it/publicfiles/ScanDiff_ICCV2025/checkpoints.zip && unzip checkpoints.zip -d checkpoints && rm checkpoints.zip
```


### 3️⃣ Quick Start!
We provide a simple ```demo.py``` script to generate scanpaths for a certain image

1. Generate scanpaths in the freeviewing setting:
```bash
python demo.py image_path=<image_path> viewing_task="" checkpoint_path=<freeviewing_ckpt_path> num_output_scanpaths=10
```

2. Generate scanpaths in the visual search setting:
```bash
python demo.py image_path=<image_path> viewing_task="bottle" checkpoint_path=<freeviewing_ckpt_path> num_output_scanpaths=10
```

### 📝 TODO

- [ ] Release data and train-val-test splits.
- [ ] Release **training code** for ScanDiff.
- [ ] Release **evaluation scripts** for benchmark comparisons.

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