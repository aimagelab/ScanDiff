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