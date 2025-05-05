import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from timm.models.layers import DropPath, trunc_normal_,Mlp
        
class SigmaFusion(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=1, mlp_ratio=2, drop=0., drop_path=0., 
                act_layer=nn.GELU, bimamba_type="v1", fusion_layers=12):
        super().__init__()
        self.fusion_layers = fusion_layers
        
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type=bimamba_type,
        )
        # self.blk = Block(d_model=dim)

        self.fuse = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.tanh = nn.Tanh()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
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
    
    def forward(self, oral_vi, oral_ir):        
        vi, ir = oral_vi, oral_ir
        
        feat_sub = self.mamba(self.norm(vi - ir))
        # feat_sub2 = self.mamba2(self.norm2(ir - vi))
        vi = oral_vi + self.tanh(feat_sub)*oral_ir
        ir = oral_ir + self.tanh(feat_sub)*oral_vi
        
        fused = self.fuse(torch.cat([vi, ir],-1))
 
        return fused, vi, ir
