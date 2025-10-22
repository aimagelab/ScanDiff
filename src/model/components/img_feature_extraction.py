import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader


#ResNet-50 backbone
class ResNetCOCO(nn.Module):
    def __init__(self):
        super(ResNetCOCO, self).__init__()
        self.resnet = maskrcnn_resnet50_fpn(pretrained=True).backbone.body.to('cuda')
        
        
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        x = x
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

from pathlib import Path
from PIL import Image
from tqdm import tqdm

class SimpleMIT1003(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.transform = T.Compose([T.ToTensor(),
                                    T.Resize((320 * 2,512 * 2)),
                                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) #put image in range [-1,1]
        
        self.imgs = list(Path('/homes/gcartella/Projects/ScanDiff/data', 'mit1003', 'images').glob('*'))
        
    def __getitem__(self, index):
        img_file = self.imgs[index]
        
        img = Image.open(img_file)
        img = self.transform(img)
        
        return {'img': img, 'filename': str(img_file)}
        
    def __len__(self):
        return len(self.imgs)
    
    
class SimpleOSIE(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.transform = T.Compose([T.ToTensor(),
                                    T.Resize((320 * 2,512 * 2)),
                                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) #put image in range [-1,1]
        
        self.imgs = list(Path('/homes/gcartella/Projects/ScanDiff/data', 'osie', 'images').glob('*'))
        
    def __getitem__(self, index):
        img_file = self.imgs[index]
        
        img = Image.open(img_file)
        img = self.transform(img)
        
        return {'img': img, 'filename': str(img_file)}
        
    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    bsz = 16
    dataset = SimpleOSIE()
    dataloader = DataLoader(dataset, batch_size=bsz, shuffle=True, drop_last=False)
    extractor = ResNetCOCO()

    for batch in tqdm(dataloader):
        img_feats = extractor(batch['img'].to('cuda'))
        img_feats = img_feats.detach().cpu()
        print(img_feats.shape)
        
        for img_feat, filename in zip(img_feats, batch['filename']):
            torch.save(img_feat, Path('/homes/gcartella/Projects/ScanDiff/data/osie/image_features', Path(filename).stem + '.pth'))