import math,os,sys
# env_path = os.path.join('/data1/Code/luandong/WWY_code_data/Codes/sigma_fusion')
# if env_path not in sys.path:
#     sys.path.append(env_path)
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_,Mlp


class MambaFusionModule(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, mlp_ratio=2, drop=0., drop_path=0., 
                act_layer=nn.GELU, bimamba_type="v3", num_layers=12):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type=bimamba_type,
                fusion_layers=num_layers
        )

    def forward(self, x):
        x_mamba = self.mamba(self.norm1(x))
        x_mamba = x_mamba + x
        return self.norm2(x_mamba)

class MambaFusion(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, mlp_ratio=2, drop=0., drop_path=0., 
                act_layer=nn.GELU, bimamba_type="v_shift", num_mamba=1, interact_layer=None):
        super().__init__()
        self.num_layers = len(interact_layer)
        self.fusion = nn.Sequential(*[
            MambaFusionModule(dim, d_state, d_conv, expand, mlp_ratio, drop, drop_path, act_layer ,
                              bimamba_type, num_layers=self.num_layers)
                for _ in range(num_mamba)
            ])
        
        self.norm = nn.LayerNorm(dim)
        self.len = 64
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, feats):
        feats = [feat[:,self.len:] for feat in feats]
        
        B,N,C = feats[0].shape
        feats = torch.cat(feats, 1)     # B, N*12, C
        feats_cross = self.fusion(feats)
        
        
        feats_fused = 0
        for i in range(self.num_layers):
            feats_fused += feats_cross[:, i*N:(i+1)*N]
            
        return self.norm(feats_fused)
        # return feats
