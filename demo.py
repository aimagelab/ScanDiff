import random
import os
import timm
import numpy as np
import PIL
import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.create_diffusion import create_diffusion
import torch
from torch import nn
from src.gazetools.display import save_image_scanpaths

def extract_img_features(img_path):
    model = timm.create_model(
        'vit_base_patch14_reg4_dinov2.lvd142m',
        pretrained=True,
        num_classes=0,  # remove classifier nn.Linear
    )
    model = model.eval()
    
    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    
    PIL_image = PIL.Image.open(img_path)
    original_size = PIL_image.size  # (width, height)
    PIL_image = PIL_image.resize((518,518))
    
    features = model.forward_features(transforms(PIL_image).unsqueeze(0))
    features = features[:, 5:, :] # remove register and CLS tokens
    features = features.squeeze().detach().cpu()

    return features, original_size

def get_task_embedding(viewing_task, cfg):
    task_embeddings = np.load(
            open(
                cfg.task_embeddings_path,
                mode="rb",
            ),
            allow_pickle=True,
    ).item()
        
    return torch.from_numpy(task_embeddings[viewing_task])

def save_scanpaths(PIL_image, pred_scanpath, scanpath_lengths, original_size):
    # save scanpaths
    os.makedirs('./demo_outputs', exist_ok=True)
    num_viewers = pred_scanpath.shape[0]

    for i in range(num_viewers):
        length = scanpath_lengths[i].item()
        scanpath = pred_scanpath[i][:length]
        x = scanpath[:,0] * original_size[0]
        y = scanpath[:,1] * original_size[1]
        t = scanpath[:,2] * 1000
        save_image_scanpaths(PIL_image, x, y, t, save_path=f'./demo_outputs/subject_{i+1}.jpg')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@hydra.main(version_base="1.3", config_path="configs", config_name="demo.yaml")
def main(cfg: DictConfig):
    seed_everything(0)
    
    img_feats, original_size = extract_img_features(cfg.image_path)
    img = PIL.Image.open(cfg.image_path).convert("RGB")
    
    viewing_task = cfg.viewing_task
    ckpt_path = cfg.checkpoint_path
    
    model = hydra.utils.instantiate(cfg.model)
    
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    model.cuda().eval()
    
    diffusion = create_diffusion(cfg,
        timestep_respacing="", diffusion_steps=cfg.get("diffusion").num_timesteps, 
        noise_schedule=cfg.get("diffusion").noise_schedule, predict_xstart=cfg.get("diffusion").predict_xstart
    )  # default: 1000 steps, linear noise schedule
    
    num_viewers = cfg.num_output_scanpaths
    img_condition = img_feats.repeat(num_viewers, 1, 1).cuda()
    task_embedding = get_task_embedding(viewing_task, cfg).cuda()
    task_embedding = task_embedding.repeat(num_viewers, 1)
    max_len = cfg.data.max_len
    
    with torch.no_grad():
        initial_noise = torch.randn(num_viewers, max_len, model.scanpath_emb_size).cuda()
        
        y = img_condition
        model_kwargs = dict(y=y, task_embedding=task_embedding)  # img conditioning

        samples = diffusion.p_sample_loop(
            model,
            initial_noise.shape,
            initial_noise,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device='cuda',
        )
        
        pred_scanpath = model.get_coords_and_time(samples)
        token_validity_preds = model.token_validity_predictor(samples)
        token_validity_preds = nn.Softmax(dim=-1)(token_validity_preds)
        token_validity_preds = token_validity_preds.argmax(
            dim=-1
        )  # NB: 1 means that the fixation is valid, 0 otherwise
        scanpath_lengths = torch.cumprod(token_validity_preds, dim=-1).sum(-1)

    save_scanpaths(img, pred_scanpath.detach().cpu().numpy(), scanpath_lengths, original_size)

if __name__ == "__main__":
    main()