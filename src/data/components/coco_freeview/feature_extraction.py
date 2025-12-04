from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import PIL
import os
from os.path import join, isdir, isfile
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoProcessor, CLIPVisionModel
import timm


class ResNetCOCO(nn.Module):
    def __init__(self, device="cuda:0"):
        super(ResNetCOCO, self).__init__()
        self.resnet = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1).backbone.body.to(device)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        bs, ch, _, _ = x.size()
        x = x.view(bs, ch, -1).permute(0, 2, 1)

        return x


def image_data(dataset_path, device='cuda:0', overwrite=False):
    resize_dim = (384 * 2, 512 * 2)
    src_path = join(dataset_path, 'images/')
    target_path = join(dataset_path, 'image_features_512_384/')
    resize = T.Resize(resize_dim)
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    bbone = ResNetCOCO(device=device).to(device).eval()
    files = [i for i in os.listdir(src_path) if isfile(join(src_path, i)) and i.endswith('.jpg')]
    for f in tqdm(files):
        if overwrite == False and os.path.exists(join(target_path, f.replace('jpg', 'pth'))):
            continue
        PIL_image = PIL.Image.open(join(src_path, f))
        tensor_image = normalize(resize(T.functional.to_tensor(PIL_image))).unsqueeze(0)

        features = bbone(tensor_image).squeeze().detach().cpu()
        torch.save(features, join(target_path, f.replace('jpg', 'pth')))

def clip_image_data(dataset_path, device='cuda:0', overwrite=False):
    src_path = join(dataset_path, 'images/')
    target_path = join(dataset_path, 'clip_hf_vitb32_image_features/')
    os.makedirs(target_path, exist_ok=True)

    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    files = [i for i in os.listdir(src_path) if isfile(join(src_path, i)) and i.endswith('.jpg')]
    for f in tqdm(files):
        if overwrite == False and os.path.exists(join(target_path, f.replace('jpg', 'pth'))):
            continue
        PIL_image = PIL.Image.open(join(src_path, f))
        PIL_image = PIL_image.resize((224,224))
        inputs = processor(images=PIL_image, return_tensors='pt')
        outputs = model(**inputs)
        features = outputs["last_hidden_state"]
        features = features[:, 1:, :] #remove CLS token
        features = features.squeeze().detach().cpu()

        torch.save(features, join(target_path, f.replace('jpg', 'pth')))

def dinoV2_image_data(dataset_path, device='cuda:0', overwrite=False):
    src_path = join(dataset_path, 'images/')
    target_path = join(dataset_path, 'dinov2_base_timm_image_features/')
    os.makedirs(target_path, exist_ok=True)

    model = timm.create_model(
    'vit_base_patch14_reg4_dinov2.lvd142m',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
    )
    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    
    files = [i for i in os.listdir(src_path) if isfile(join(src_path, i)) and i.endswith('.jpg')]
    for f in tqdm(files):
        if overwrite == False and os.path.exists(join(target_path, f.replace('jpg', 'pth'))):
            continue
        PIL_image = PIL.Image.open(join(src_path, f))
        PIL_image = PIL_image.resize((518,518))
        
        features = model.forward_features(transforms(PIL_image).unsqueeze(0))
        features = features[:, 5:, :]
        features = features.squeeze().detach().cpu()

        torch.save(features, join(target_path, f.replace('jpg', 'pth')))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Feature Extractor Utils', add_help=False)
    parser.add_argument('--dataset_path', default= './data/coco_freeview', type=str)
    parser.add_argument('--cuda', default=0, type=int)
    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args.cuda))
    dinoV2_image_data(dataset_path = args.dataset_path, device=device, overwrite = True)