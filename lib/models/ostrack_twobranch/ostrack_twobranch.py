"""
Basic OSTrack model.
"""
import math
import sys, os
from typing import List
# env_path = os.path.join('/data1/Code/luandong/WWY_code_data/Codes/sigma_fusion')
# if env_path not in sys.path:
#     sys.path.append(env_path)
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_clones
from lib.models.layers.head import build_box_head
# from lib.models.ostrack_twobranch.vit_mine import vit_base_patch16_224
from lib.models.ostrack_twobranch.vit_mine import vit_base_patch16_224
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.ostrack_twobranch.component.mamba_fusion import MambaFusion
import numpy as np



class OSTrack_twobranch(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer, box_head, pretrained, aux_loss=False, head_type="CORNER",interact_layer=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.embed_dim = 768
        self.interact_layer = interact_layer
        self.backbone = transformer
        self.mamba_fusion = MambaFusion(dim=self.embed_dim, interact_layer=interact_layer, bimamba_type="v_shift", num_mamba=1)
        
        self.box_head = box_head

        self.aux_loss = aux_loss  
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)


    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        # —— Joint Feature Extraction & Relation Modeling ——
        _, fused_feats, aux_dict, enhanced = self.backbone(z=template, x=search,
                                                ce_template_mask=ce_template_mask,
                                                ce_keep_rate=ce_keep_rate)

        x = self.mamba_fusion(fused_feats)

        # —— Forward head ——
        feat_last = x
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        out['enhanced'] = enhanced
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        # enc_opt = cat_feature
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_ostrack_twobranch(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and ('DropTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = cfg.MODEL.PRETRAIN_FILE
    pretrained=False

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':   # this is selected
        backbone = vit_base_patch16_224(pretrained,
                                        drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                        interact_layer=cfg.MODEL.BACKBONE.INTERACT_LAYER
                                        )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        # backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
        #                                    ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
        #                                    ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
        #                                    )

        # hidden_dim = backbone.embed_dim + backbone.embed_dim
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = OSTrack_twobranch(
        backbone,
        box_head,
        pretrained,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        interact_layer = cfg.MODEL.BACKBONE.INTERACT_LAYER
    )


    if training and ('OSTrack' in cfg.MODEL.PRETRAIN_FILE or 'DropTrack' in cfg.MODEL.PRETRAIN_FILE or 'ODTrack' in cfg.MODEL.PRETRAIN_FILE):
    # if False:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        param_dict_rgbt = dict()
        if 'DropTrack' in cfg.MODEL.PRETRAIN_FILE:
            for k,v in checkpoint["net"].items():
                if k in ['box_head.conv1_ctr.0.weight','box_head.conv1_offset.0.weight','box_head.conv1_size.0.weight']:
                    # v = torch.cat([v,v],1)
                    v = v
                elif 'pos_embed_x' in k:
                    v = resize_pos_embed(v, 16, 16) + checkpoint["net"]['backbone.temporal_pos_embed_x']
                elif 'pos_embed_z' in k:
                    v = resize_pos_embed(v, 8, 8) + checkpoint["net"]['backbone.temporal_pos_embed_z']
                else:
                    v = v
                param_dict_rgbt[k] = v
                
        # elif 'OSTrack' in cfg.MODEL.PRETRAIN_FILE:
        #     for k,v in checkpoint["net"].items():
        #         if 'pos_embed_x' in k:
        #             v = resize_pos_embed(v, 24, 24)
        #         elif 'pos_embed_z' in k:
        #             v = resize_pos_embed(v, 12, 12)
           
        #         param_dict_rgbt[k] = v
        
        # if 'ODTrack' in cfg.MODEL.PRETRAIN_FILE:
        #     for k,v in checkpoint["net"].items():
        #         if 'cls_pos_embed' in k:
        #             continue
        #         if k in ['box_head.conv1_ctr.0.weight','box_head.conv1_offset.0.weight','box_head.conv1_size.0.weight']:
        #                 # v = torch.cat([v,v],1)
        #                 v = v
        #         # elif 'pos_embed_x' in k:
        #         #     v = resize_pos_embed(v, 16, 16)
        #         # elif 'pos_embed_z' in k:
        #         #     v = resize_pos_embed(v, 8, 8)
        #         else:
        #             v = v
        #         param_dict_rgbt[k] = v

        missing_keys, unexpected_keys = model.load_state_dict(param_dict_rgbt, strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        print('missing_keys, unexpected_keys',missing_keys, unexpected_keys)
    return model

def resize_pos_embed(posemb, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb_grid = posemb[0, :]
    
    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to new token with height:{} width: {}'.format(posemb_grid.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    # posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb_grid
