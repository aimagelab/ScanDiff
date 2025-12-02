from src.model.components.img_text_projector import ImgTextProjector
from src.model.components.model import DiT
import torch.nn as nn
import torch

from src.utils.train_utils import get_device

class DiTModel(DiT):
    def __init__(
        self,
        encoder,
        decoder,
        max_len=40,
        spatial_dim=(37,37),
        patch_size=2,
        in_channels=4,
        scanpath_emb_size=256,
        hidden_size=512,
        img_feature_dim=768,
        text_feature_dim=512,
        depth=6,
        num_heads=8,
        mlp_ratio=4,
        learn_sigma=False,
        task='unified_model',
    ):
        super().__init__(
            max_len,
            spatial_dim,
            patch_size,
            in_channels,
            scanpath_emb_size,
            hidden_size,
            img_feature_dim,
            depth,
            num_heads,
            mlp_ratio,
            learn_sigma,
        )
        
        self.encoder = encoder
        self.decoder = decoder
        self.scanpath_emb_size = scanpath_emb_size
        
        self.output_down_proj = nn.Linear(hidden_size, scanpath_emb_size)
        self.token_validity_predictor = nn.Linear(scanpath_emb_size, 2)
        self.task = task
        
        self.input_up_proj = nn.Linear(scanpath_emb_size, hidden_size)
        
        self.dropout = nn.Dropout(0.1)
        self.pos_embed = nn.Embedding(max_len, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.position_ids = torch.arange(self.max_len).to(get_device())
        
        if self.task in ['visual_search', 'unified_model']:
            self.img_text_projector = ImgTextProjector(img_feature_dim, text_feature_dim, hidden_size)
        else:
            self.img_text_projector = None

    def forward(self, x, t, y, task_embedding=None):
        """y
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, num_pathces, dim) tensor representing the img feats to condition the generation process
        """
        x = self.input_up_proj(x)
        
        pos_emb = self.pos_embed(self.position_ids)  # (N, max_len, dim), where 3 is x, y, and time
                                                    # learnable positional embedding
        t = self.t_embedder(t).unsqueeze(1)                  # (N, 1, dim)
        
        x = x + t + pos_emb
        y = self.img_proj(y)
        y = self.img_patch_pos(y)

        if self.task in ['visual_search', 'unified_model']:
            y = self.img_text_projector(y, task_embedding)

        c = y
        
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x)                
        
        x = self.output_down_proj(x)
        return x
    
    def get_embeds(self, gt_scanpath):
        return self.encoder(gt_scanpath)
    
    def get_coords_and_time(self, model_output):
        return self.decoder(model_output)