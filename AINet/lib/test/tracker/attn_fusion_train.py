import math

from lib.models.ostrack_attnFusion import build_ostrack_attn_fusion_train
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target

import cv2
import os
import torch.nn.functional as F

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class OSTrack_twobranch(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrack_twobranch, self).__init__(params)
        network = build_ostrack_attn_fusion_train(params.cfg, training=False)
        try:
            m,n = network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
            if len(m)+len(n)!=0:
                print(m,'\n',n)
        except:
            network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu'), strict=True)
        self.cfg = params.cfg
        self.update = params.update
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image_v, image_i, info: dict):
        self.contrs = None
        self.contrs_ts = None
        # forward the template once
        z_patch_arr_rgb, resize_factor_rgb, z_amask_arr_rgb = sample_target(image_v, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)

        z_patch_arr_tir, resize_factor_tir, z_amask_arr_tir = sample_target(image_i, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        # z_patch_arr_tir=z_patch_arr_rgb
        # z_patch_arr_rgb=z_patch_arr_tir

        self.z_patch_arr_rgb = z_patch_arr_rgb
        self.z_patch_arr_tir = z_patch_arr_tir

        template_rgb = self.preprocessor.process(z_patch_arr_rgb, z_amask_arr_rgb)
        template_tir = self.preprocessor.process(z_patch_arr_tir, z_amask_arr_tir)
        with torch.no_grad():
            self.z_dict1_rgb = template_rgb
            self.z_dict1_tir = template_tir
            self.z_dict = [self.z_dict1_rgb,self.z_dict1_tir]
        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox_rgb = self.transform_bbox_to_crop(info['init_bbox'], resize_factor_rgb,
                                                        template_rgb.tensors.device).squeeze(1)
            self.box_mask_z_rgb = generate_mask_cond(self.cfg, 1, template_rgb.tensors.device, template_bbox_rgb)

            template_bbox_tir = self.transform_bbox_to_crop(info['init_bbox'], resize_factor_tir,
                                                        template_tir.tensors.device).squeeze(1)
            self.box_mask_z_tir = generate_mask_cond(self.cfg, 1, template_tir.tensors.device, template_bbox_tir)

            self.box_mask_z = self.box_mask_z_rgb
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image_v,image_i, info: dict = None):
        H, W, _ = image_v.shape
        self.frame_id += 1
        x_patch_arr_rgb, resize_factor_rgb, x_amask_arr_rgb = sample_target(image_v, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        x_patch_arr_tir, resize_factor_tir, x_amask_arr_tir = sample_target(image_i, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)

        # x_patch_arr_tir*=0
        # x_patch_arr_tir=x_patch_arr_rgb

        search_rgb = self.preprocessor.process(x_patch_arr_rgb, x_amask_arr_rgb)
        search_tir = self.preprocessor.process(x_patch_arr_tir, x_amask_arr_tir)

        with torch.no_grad():
            x_dict_rgb = search_rgb
            x_dict_tir = search_tir

            x_dict = [x_dict_rgb,x_dict_tir]
            # merge the template and the search
            # run the transformer
            # out_dict = self.network.forward(
            #     template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)
            out_dict = self.network.forward(
                template=[self.z_dict[0].tensors, self.z_dict[1].tensors], 
                search=[x_dict[0].tensors, x_dict[1].tensors], 
                ce_template_mask=self.box_mask_z)    

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map

        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor_rgb).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor_rgb), H, W, margin=10)


        # 计算模态贡献值
        # contr = torch.zeros([1,2], device=pred_score_map.device)
        # blk_num = 0
        # for blk in self.network.backbone.blocks:
        #     if hasattr(blk, 'inner_attn') and hasattr(blk.inner_attn, 'attn'):
        #         blk_num += 1
        #         attn_sum = sum(blk.inner_attn.attn)
        #         length = attn_sum.shape[-1]//2
        #         contr[0,0] += attn_sum[..., :length].sum()/length   # RGB贡献
        #         contr[0,1] += attn_sum[..., length:].sum()/length   # TIR贡献
        # contr = (contr/blk_num).softmax(dim=1)
        
        contr_ts = torch.zeros([1,2], device=pred_score_map.device)
        blk_num_ts = 0
        # for blk in self.network.backbone.blocks:
        #     if hasattr(blk, 'inner_attn_ts') and hasattr(blk.inner_attn_ts, 'attn'):
        #         blk_num_ts += 1
        #         attn_sum = sum(blk.inner_attn_ts.attn)
        #         length = attn_sum.shape[-1]//2
        #         contr_ts[0,1] += attn_sum[..., :length].sum()/length   # TIR贡献
        #         contr_ts[0,0] += attn_sum[..., length:].sum()/length   # RGB贡献
        # contr_ts = (contr_ts/blk_num_ts).softmax(dim=1)
        
        if self.debug:
            if self.contrs == None:
                self.contrs = contr
                self.contrs_ts = contr_ts
            else:
                self.contrs = torch.cat([self.contrs, contr], dim=0)
                self.contrs_ts = torch.cat([self.contrs_ts, contr_ts], dim=0)
            self.visdom.register(self.contrs, 'lineplot', 1, 'contribution')
            
            if blk_num_ts:
                if self.contrs_ts == None:
                    self.contrs_ts = contr_ts
                else:
                    self.contrs_ts = torch.cat([self.contrs_ts, contr_ts], dim=0)
                self.visdom.register(self.contrs_ts, 'lineplot', 1, 'contribution—ts')

        # 简单写个模板更新
        if response.max()>self.update[0]:
            
            if contr[0,0]<self.update[1]:
                z_patch_arr_rgb, resize_factor_rgb, z_amask_arr_rgb = sample_target(image_v, self.state, self.params.template_factor,
                                                            output_sz=self.params.template_size)
                self.z_patch_arr_rgb = z_patch_arr_rgb
                template_rgb = self.preprocessor.process(z_patch_arr_rgb, z_amask_arr_rgb)
                self.z_dict1_rgb = template_rgb

                if self.cfg.MODEL.BACKBONE.CE_LOC:
                    template_bbox_rgb = self.transform_bbox_to_crop(self.state, resize_factor_rgb,
                                                                template_rgb.tensors.device).squeeze(1)
                    self.box_mask_z_rgb = generate_mask_cond(self.cfg, 1, template_rgb.tensors.device, template_bbox_rgb)


            if contr[0,1]<self.update[2]:
                z_patch_arr_tir, resize_factor_tir, z_amask_arr_tir = sample_target(image_i, self.state, self.params.template_factor,
                                                            output_sz=self.params.template_size)
                self.z_patch_arr_tir = z_patch_arr_tir
                template_tir = self.preprocessor.process(z_patch_arr_tir, z_amask_arr_tir)
                self.z_dict1_tir = template_tir

                if self.cfg.MODEL.BACKBONE.CE_LOC:
                    template_bbox_tir = self.transform_bbox_to_crop(self.state, resize_factor_tir,
                                                                template_tir.tensors.device).squeeze(1)
                    self.box_mask_z_tir = generate_mask_cond(self.cfg, 1, template_tir.tensors.device, template_bbox_tir)

            with torch.no_grad():
                self.z_dict = [self.z_dict1_rgb,self.z_dict1_tir]

                self.box_mask_z = [self.box_mask_z_rgb,self.box_mask_z_tir]
                self.box_mask_z = self.box_mask_z[0]



        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image_v, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image_v, info['gt_bbox'].tolist(), self.state, info['baseline_rect'].tolist()), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr_rgb).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(x_patch_arr_tir).permute(2, 0, 1), 'image', 1, 'search_region_t')
                self.visdom.register(torch.from_numpy(self.z_patch_arr_rgb).permute(2, 0, 1), 'image', 1, 'template_v')
                self.visdom.register(torch.from_numpy(self.z_patch_arr_tir).permute(2, 0, 1), 'image', 1, 'template_t')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')
                
                # enc_opt = out_dict['backbone_feat'][0, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
                # opt = enc_opt.reshape()

                # 注意力权重可视化
                if 0:
                    # 数据提取
                    rgb_attn_map = []
                    tir_attn_map = []
                    for blk in self.network.backbone.blocks:
                        if hasattr(blk, 'attn_map'):
                            blk_num += 1
                            rgb_attn_map.append(blk.attn_map['RGB'])
                            tir_attn_map.append(blk.attn_map['TIR'])
                    # 格式化
                    # 层 -> 选择attn_map -> batch_size,hn,H,W
                    layer = -1
                    a=2
                    # st_1_tir=torch.ones([256]).cuda()*tir_attn_map[layer][0][0].sum(0)[64:,:64].mean(-1).min()
                    # st_2_tir=torch.ones([256]).cuda()*tir_attn_map[layer][1][0].sum(0).mean(-1).min()
                    # st_3_tir=torch.ones([256]).cuda()*tir_attn_map[layer][2][0].sum(0).mean(-1).min()
                    st_1_tir=torch.zeros([256]).cuda()
                    st_2_tir=torch.zeros([256]).cuda()
                    st_3_tir=torch.zeros([256]).cuda()
                    st_1_tir[out_dict['aux_dict_tir']['global_index_s'][a].squeeze().long()] = tir_attn_map[layer][0][0].sum(0)[64:,:64].max(-1).values
                    st_2_tir[out_dict['aux_dict_tir']['global_index_s'][a].squeeze().long()] = tir_attn_map[layer][1][0].sum(0).max(-1).values
                    st_3_tir[out_dict['aux_dict_tir']['global_index_s'][a].squeeze().long()] = tir_attn_map[layer][2][0].sum(0)[64:,:64].max(-1).values
                    # st_1_tir[out_dict['aux_dict_tir']['global_index_s'][a].squeeze().long()] = tir_attn_map[layer][0][0].sum(0)[64:,:64].mean(-1)
                    # st_2_tir[out_dict['aux_dict_tir']['global_index_s'][a].squeeze().long()] = tir_attn_map[layer][1][0].sum(0).mean(-1)
                    # st_3_tir[out_dict['aux_dict_tir']['global_index_s'][a].squeeze().long()] = tir_attn_map[layer][2][0].sum(0)[64:,:64].mean(-1)
                    st_1_tir = F.interpolate(st_1_tir.reshape(16,16).unsqueeze(0).unsqueeze(0), scale_factor=8, mode='bilinear')
                    self.visdom.register(st_1_tir, 'heatmap', 1, 'attn-tir-st-ori')
                    # st_2_tir = F.interpolate(st_2_tir.reshape(16,16).unsqueeze(0).unsqueeze(0), scale_factor=8, mode='bilinear')
                    # self.visdom.register(st_2_tir, 'heatmap', 1, 'attn-tir-st-add')
                    st_3_tir = F.interpolate(st_3_tir.reshape(16,16).unsqueeze(0).unsqueeze(0), scale_factor=8, mode='bilinear')
                    self.visdom.register(st_3_tir, 'heatmap', 1, 'attn-tir-st-new')

                    # st_1_rgb=torch.ones([256]).cuda()*rgb_attn_map[layer][0][0].sum(0)[64:,:64].mean(-1).min()
                    # st_2_rgb=torch.ones([256]).cuda()*rgb_attn_map[layer][1][0].sum(0).mean(-1).min()
                    # st_3_rgb=torch.ones([256]).cuda()*rgb_attn_map[layer][2][0].sum(0).mean(-1).min()
                    st_1_rgb=torch.zeros([256]).cuda()
                    st_2_rgb=torch.zeros([256]).cuda()
                    st_3_rgb=torch.zeros([256]).cuda()
                    st_1_rgb[out_dict['aux_dict_rgb']['global_index_s'][a].squeeze().long()] = rgb_attn_map[layer][0][0].sum(0)[64:,:64].max(-1).values
                    st_2_rgb[out_dict['aux_dict_rgb']['global_index_s'][a].squeeze().long()] = rgb_attn_map[layer][1][0].sum(0).max(-1).values
                    st_3_rgb[out_dict['aux_dict_rgb']['global_index_s'][a].squeeze().long()] = rgb_attn_map[layer][2][0].sum(0)[64:,:64].max(-1).values
                    # st_1_rgb[out_dict['aux_dict_rgb']['global_index_s'][a].squeeze().long()] = rgb_attn_map[layer][0][0].sum(0)[64:,:64].mean(-1)
                    # st_2_rgb[out_dict['aux_dict_rgb']['global_index_s'][a].squeeze().long()] = rgb_attn_map[layer][1][0].sum(0).mean(-1)
                    # st_3_rgb[out_dict['aux_dict_rgb']['global_index_s'][a].squeeze().long()] = rgb_attn_map[layer][2][0].sum(0)[64:,:64].mean(-1)
                    st_1_rgb = F.interpolate(st_1_rgb.reshape(16,16).unsqueeze(0).unsqueeze(0), scale_factor=8, mode='bilinear')
                    self.visdom.register(st_1_rgb, 'heatmap', 1, 'attn-rgb-st-ori')
                    # st_2_rgb = F.interpolate(st_2_rgb.reshape(16,16).unsqueeze(0).unsqueeze(0), scale_factor=8, mode='bilinear')
                    # self.visdom.register(st_2_rgb, 'heatmap', 1, 'attn-rgb-st-add')
                    st_3_rgb = F.interpolate(st_3_rgb.reshape(16,16).unsqueeze(0).unsqueeze(0), scale_factor=8, mode='bilinear')
                    self.visdom.register(st_3_rgb, 'heatmap', 1, 'attn-rgb-st-new')


                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr_rgb, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor_rgb, resize_factor_rgb)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return OSTrack_twobranch
