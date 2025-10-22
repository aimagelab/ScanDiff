import torch.nn as  nn
import torch

class ImgTextProjector(nn.Module):
    def __init__(self, img_dim, text_dim, common_dim):
        super(ImgTextProjector, self).__init__()
        self.img_dim = img_dim
        self.text_dim = text_dim
        self.common_dim = common_dim

        self.text_transform = nn.Linear(text_dim, common_dim)
        self.img_transform = nn.Linear(common_dim, common_dim)

        self.multimodal_transform = nn.Linear(common_dim * 2, common_dim)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def forward(self, img_feats, text_feats, return_only_text=False):
        img = self.img_transform(img_feats)
        text = self.text_transform(text_feats)
        
        if return_only_text:
            return text
        
        img_task_fusion = self.multimodal_transform(torch.cat([img, text.unsqueeze(1).repeat(1,img.size(1),1)], dim = -1))
        img_task_fusion = self.activation(img_task_fusion)
        img_task_fusion = self.dropout(img_task_fusion)
        
        return img_task_fusion