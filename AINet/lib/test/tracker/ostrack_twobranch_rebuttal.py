import math

from lib.models.ostrack_twobranch import build_ostrack_twobranch
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
import torch.nn.functional as F
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class OSTrack_twobranch(BaseTracker):
    def __init__(self, params, dataset_name=None):
        super(OSTrack_twobranch, self).__init__(params)
        network = build_ostrack_twobranch(params.cfg, training=False)
        # try:
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        # except:
        #     network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu'), strict=True)
        self.cfg = params.cfg
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
                self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image_v, image_i, info: dict):
        self.temps = []
        self.temps_score = []
        # forward the template once
        z_patch_arr_rgb, resize_factor_rgb, z_amask_arr_rgb = sample_target(image_v, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)

        z_patch_arr_tir, resize_factor_tir, z_amask_arr_tir = sample_target(image_i, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)

        # z_patch_arr_tir[:16,:16,:]*=0
        # z_patch_arr_tir[-16:,:16,:]=255.
        # z_patch_arr_rgb[:,-32:,:]=255
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

            self.box_mask_z = [self.box_mask_z_rgb,self.box_mask_z_tir]
            self.box_mask_z = self.box_mask_z[0]
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image_v,image_i, info: dict = None,vis_frame=-1, gt=None):
        H, W, _ = image_v.shape
        self.frame_id += 1
        x_patch_arr_rgb, resize_factor_rgb, x_amask_arr_rgb = sample_target(image_v, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        x_patch_arr_tir, resize_factor_tir, x_amask_arr_tir = sample_target(image_i, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)

        # x_patch_arr_rgb*=0
        # x_patch_arr_tir*=0
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
        # ious = []
        # for i in range(12):
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map

        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor_rgb).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor_rgb), H, W, margin=10)
            # iou = self.cal_iou(self.state, gt[vis_frame])
            # ious.append(round(iou, 3))
        
        # iou = self.cal_iou(self.state, gt[vis_frame])
        # print('iou:', iou)
        
        if self.frame_id == vis_frame:
            return x_dict,out_dict
        else:
            return None,None

        # 简单写个模板更新
        
        # self.temps.append([sample_target(image_v, self.state, self.params.template_factor, output_sz=self.params.template_size), 
        #                    sample_target(image_i, self.state, self.params.template_factor, output_sz=self.params.template_size)])
        # self.temps_score.append(response.max())
        # while len(self.temps)>5:
        #     del self.temps[0]
        #     del self.temps_score[0]

        # if not self.frame_id%10: # 定时更新
        #     if max(self.temps_score)>0.65:
        #         max_idx = self.temps_score.index(max(self.temps_score))
            
        #         z_patch_arr_rgb, resize_factor_rgb, z_amask_arr_rgb = self.temps[max_idx][0]

        #         z_patch_arr_tir, resize_factor_tir, z_amask_arr_tir = self.temps[max_idx][1]

        #         self.z_patch_arr_rgb = z_patch_arr_rgb
        #         self.z_patch_arr_tir = z_patch_arr_tir

        #         template_rgb = self.preprocessor.process(z_patch_arr_rgb, z_amask_arr_rgb)
        #         template_tir = self.preprocessor.process(z_patch_arr_tir, z_amask_arr_tir)
        #         with torch.no_grad():
        #             self.z_dict1_rgb = template_rgb
        #             self.z_dict1_tir = template_tir
        #             self.z_dict = [self.z_dict1_rgb,self.z_dict1_tir]
        #         self.box_mask_z = None
        #         if self.cfg.MODEL.BACKBONE.CE_LOC:
        #             template_bbox_rgb = self.transform_bbox_to_crop(self.state, resize_factor_rgb,
        #                                                         template_rgb.tensors.device).squeeze(1)
        #             self.box_mask_z_rgb = generate_mask_cond(self.cfg, 1, template_rgb.tensors.device, template_bbox_rgb)

        #             template_bbox_tir = self.transform_bbox_to_crop(self.state, resize_factor_tir,
        #                                                         template_tir.tensors.device).squeeze(1)
        #             self.box_mask_z_tir = generate_mask_cond(self.cfg, 1, template_tir.tensors.device, template_bbox_tir)

        #             self.box_mask_z = [self.box_mask_z_rgb,self.box_mask_z_tir]
        #             self.box_mask_z = self.box_mask_z[0]

        

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor_rgb, resize_factor_rgb)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def cal_iou(self, box1, box2):
        # 计算两个边界框的左上角和右下角坐标
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # 计算交集区域的坐标
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1 + w1, x2 + w2)
        inter_y2 = min(y1 + h1, y2 + h2)

        # 计算交集区域的宽度和高度
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)

        # 计算交集区域的面积
        intersection_area = inter_w * inter_h

        # 计算box1和box2的面积
        box1_area = w1 * h1
        box2_area = w2 * h2

        # 计算并集区域的面积
        union_area = box1_area + box2_area - intersection_area

        # 计算IoU
        iou = intersection_area / union_area if union_area != 0 else 0

        return iou
    
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
