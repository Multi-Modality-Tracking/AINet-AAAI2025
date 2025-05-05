import math

from lib.models.ostrack_twobranch import build_ostrack_twobranch
from lib.test.tracker.basetracker import BaseTracker
import torch
import numpy as np
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
import torch.nn.functional as F
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond

def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''

    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou

def gen_samples(generator, bbox, n, overlap_range=None, scale_range=None):

    if overlap_range is None and scale_range is None:
        return generator(bbox, n)

    else:
        samples = None
        remain = n
        factor = 2
        while remain > 0 and factor < 64:
            samples_ = generator(bbox, remain*factor)

            idx = np.ones(len(samples_), dtype=bool)
            if overlap_range is not None:
                r = overlap_ratio(samples_, bbox)
                idx *= (r >= overlap_range[0]) * (r <= overlap_range[1])
            if scale_range is not None:
                s = np.prod(samples_[:,2:], axis=1) / np.prod(bbox[2:])
                idx *= (s >= scale_range[0]) * (s <= scale_range[1])

            samples_ = samples_[idx,:]
            samples_ = samples_[:min(remain, len(samples_))]
            if samples is None:
                samples = samples_
            else:
                samples = np.concatenate([samples, samples_])
            remain = n - len(samples)
            factor = factor*2
        if remain!=0:
            return None
        return samples
    
class SampleGenerator():
    def __init__(self, type, img_size, trans_f=1, scale_f=1, aspect_f=None, valid=False):
        self.type = type
        self.img_size = np.array(img_size) # (w, h)
        self.trans_f = trans_f
        self.scale_f = scale_f
        self.aspect_f = aspect_f
        self.valid = valid

    def __call__(self, bb, n):
        #
        # bb: target bbox (min_x,min_y,w,h)
        bb = np.array(bb, dtype='float32')

        # (center_x, center_y, w, h)
        sample = np.array([bb[0]+bb[2]/2, bb[1]+bb[3]/2, bb[2], bb[3]], dtype='float32')
        samples = np.tile(sample[None,:],(n,1))

        # vary aspect ratio
        if self.aspect_f is not None:
            ratio = np.random.rand(n,1)*2-1
            samples[:,2:] *= self.aspect_f ** np.concatenate([ratio, -ratio],axis=1)

        # sample generation
        if self.type=='gaussian':
            samples[:,:2] += self.trans_f * np.mean(bb[2:]) * np.clip(0.5*np.random.randn(n,2),-1,1)
            samples[:,2:] *= self.scale_f ** np.clip(0.5*np.random.randn(n,1),-1,1)

        elif self.type=='uniform':
            samples[:,:2] += self.trans_f * np.mean(bb[2:]) * (np.random.rand(n,2)*2-1)
            samples[:,2:] *= self.scale_f ** (np.random.rand(n,1)*2-1)

        elif self.type=='whole':
            m = int(2*np.sqrt(n))
            xy = np.dstack(np.meshgrid(np.linspace(0,1,m),np.linspace(0,1,m))).reshape(-1,2)
            xy = np.random.permutation(xy)[:n]
            samples[:,:2] = bb[2:]/2 + xy * (self.img_size-bb[2:]/2-1)
            samples[:,2:] *= self.scale_f ** (np.random.rand(n,1)*2-1)

        # adjust bbox range
        samples[:,2:] = np.clip(samples[:,2:], 5, self.img_size-5.)
        if self.valid:
            samples[:,:2] = np.clip(samples[:,:2], samples[:,2:]/2, self.img_size-samples[:,2:]/2-1)
        else:
            samples[:,:2] = np.clip(samples[:,:2], 0, self.img_size)

        # (min_x, min_y, w, h)
        samples[:,:2] -= samples[:,2:]/2

        return samples

    def set_trans_f(self, trans_f):
        self.trans_f = trans_f

    def get_trans_f(self):
        return self.trans_f

class OSTrack_twobranch(BaseTracker):
    def __init__(self, params, dataset_name=None):
        super(OSTrack_twobranch, self).__init__(params)
        network = build_ostrack_twobranch(params.cfg, training=False)
        # try:     
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
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

    def track(self, image_v,image_i, info: dict = None, seq_name=None):
        H, W, _ = image_v.shape
        self.frame_id += 1
        gt = np.loadtxt('/data1/Datasets/Tracking/RGBT234/{}/init.txt'.format(seq_name),delimiter=',')
        
        if self.frame_id == 100:
            pos_examples = gen_samples(SampleGenerator('gaussian', (H, W), 0.1, 1.2), gt[self.frame_id], 128, [0.4,1])
            if pos_examples is None:
                print('{}_{} failed'.format(seq_name,self.frame_id))
                return None
            
            pos_idx = 0
            for pos_index in pos_examples:
                pos_idx = pos_idx+1
                print('pos', pos_idx, pos_index)
                self.state = pos_index.tolist()
                x_patch_arr_rgb, resize_factor_rgb, x_amask_arr_rgb = sample_target(image_v, self.state, self.params.search_factor,
                                                                        output_sz=self.params.search_size)  # (x1, y1, w, h)
                x_patch_arr_tir, resize_factor_tir, x_amask_arr_tir = sample_target(image_i, self.state, self.params.search_factor,
                                                                        output_sz=self.params.search_size)  # (x1, y1, w, h)

                search_rgb = self.preprocessor.process(x_patch_arr_rgb, x_amask_arr_rgb)
                search_tir = self.preprocessor.process(x_patch_arr_tir, x_amask_arr_tir)

                with torch.no_grad():
                    x_dict_rgb = search_rgb
                    x_dict_tir = search_tir

                    x_dict = [x_dict_rgb,x_dict_tir]
                    out_dict = self.network.forward(
                        template=[self.z_dict[0].tensors, self.z_dict[1].tensors], 
                        search=[x_dict[0].tensors, x_dict[1].tensors], 
                        ce_template_mask=self.box_mask_z)
                        
                backbone_feat = out_dict['enhanced']
                if pos_idx==1:
                    pos_feature0 = backbone_feat[0].cpu().numpy().reshape(1,-1)
                    pos_feature1 = backbone_feat[1].cpu().numpy().reshape(1,-1)
                    pos_feature2 = backbone_feat[2].cpu().numpy().reshape(1,-1)
                    pos_feature3 = backbone_feat[3].cpu().numpy().reshape(1,-1)
                    pos_feature4 = backbone_feat[4].cpu().numpy().reshape(1,-1)
                    pos_feature5 = backbone_feat[5].cpu().numpy().reshape(1,-1)
                    pos_feature6 = backbone_feat[6].cpu().numpy().reshape(1,-1)
                    pos_feature7 = backbone_feat[7].cpu().numpy().reshape(1,-1)
                    pos_feature8 = backbone_feat[8].cpu().numpy().reshape(1,-1)
                    pos_feature9 = backbone_feat[9].cpu().numpy().reshape(1,-1)
                    pos_feature10 = backbone_feat[10].cpu().numpy().reshape(1,-1)
                    pos_feature11 = backbone_feat[11].cpu().numpy().reshape(1,-1)
                else:
                    pos_feature0 = np.vstack((pos_feature0, backbone_feat[0].cpu().numpy().reshape(1,-1)))
                    pos_feature1 = np.vstack((pos_feature1, backbone_feat[1].cpu().numpy().reshape(1,-1)))
                    pos_feature2 = np.vstack((pos_feature2, backbone_feat[2].cpu().numpy().reshape(1,-1)))
                    pos_feature3 = np.vstack((pos_feature3, backbone_feat[3].cpu().numpy().reshape(1,-1)))
                    pos_feature4 = np.vstack((pos_feature4, backbone_feat[4].cpu().numpy().reshape(1,-1)))
                    pos_feature5 = np.vstack((pos_feature5, backbone_feat[5].cpu().numpy().reshape(1,-1)))
                    pos_feature6 = np.vstack((pos_feature6, backbone_feat[6].cpu().numpy().reshape(1,-1)))
                    pos_feature7 = np.vstack((pos_feature7, backbone_feat[7].cpu().numpy().reshape(1,-1)))
                    pos_feature8 = np.vstack((pos_feature8, backbone_feat[8].cpu().numpy().reshape(1,-1)))
                    pos_feature9 = np.vstack((pos_feature9, backbone_feat[9].cpu().numpy().reshape(1,-1)))
                    pos_feature10 = np.vstack((pos_feature10, backbone_feat[10].cpu().numpy().reshape(1,-1)))
                    pos_feature11 = np.vstack((pos_feature11, backbone_feat[11].cpu().numpy().reshape(1,-1)))
            
            pos_features = [pos_feature0,pos_feature1,pos_feature2,pos_feature3,pos_feature4,pos_feature5,\
                pos_feature6,pos_feature7,pos_feature8,pos_feature9,pos_feature10,pos_feature11]
            
            # 负样本
            neg_examples = gen_samples(SampleGenerator('gaussian', (H, W), 0.5, 2, 2), gt[self.frame_id], 128, [0.0,0.6])
            neg_idx = 0
            for neg_index in neg_examples:
                neg_idx = neg_idx+1
                print('neg', neg_idx, neg_index)
                self.state = neg_index.tolist()
                x_patch_arr_rgb, resize_factor_rgb, x_amask_arr_rgb = sample_target(image_v, self.state, self.params.search_factor,
                                                                        output_sz=self.params.search_size)  # (x1, y1, w, h)
                x_patch_arr_tir, resize_factor_tir, x_amask_arr_tir = sample_target(image_i, self.state, self.params.search_factor,
                                                                        output_sz=self.params.search_size)  # (x1, y1, w, h)

                search_rgb = self.preprocessor.process(x_patch_arr_rgb, x_amask_arr_rgb)
                search_tir = self.preprocessor.process(x_patch_arr_tir, x_amask_arr_tir)

                with torch.no_grad():
                    x_dict_rgb = search_rgb
                    x_dict_tir = search_tir

                    x_dict = [x_dict_rgb,x_dict_tir]
                    out_dict = self.network.forward(
                        template=[self.z_dict[0].tensors, self.z_dict[1].tensors], 
                        search=[x_dict[0].tensors, x_dict[1].tensors], 
                        ce_template_mask=self.box_mask_z)
                        
                backbone_feat = out_dict['enhanced']
                if neg_idx==1:
                    neg_feature0 = backbone_feat[0].cpu().numpy().reshape(1,-1)
                    neg_feature1 = backbone_feat[1].cpu().numpy().reshape(1,-1)
                    neg_feature2 = backbone_feat[2].cpu().numpy().reshape(1,-1)
                    neg_feature3 = backbone_feat[3].cpu().numpy().reshape(1,-1)
                    neg_feature4 = backbone_feat[4].cpu().numpy().reshape(1,-1)
                    neg_feature5 = backbone_feat[5].cpu().numpy().reshape(1,-1)
                    neg_feature6 = backbone_feat[6].cpu().numpy().reshape(1,-1)
                    neg_feature7 = backbone_feat[7].cpu().numpy().reshape(1,-1)
                    neg_feature8 = backbone_feat[8].cpu().numpy().reshape(1,-1)
                    neg_feature9 = backbone_feat[9].cpu().numpy().reshape(1,-1)
                    neg_feature10 = backbone_feat[10].cpu().numpy().reshape(1,-1)
                    neg_feature11 = backbone_feat[11].cpu().numpy().reshape(1,-1)
                else:
                    neg_feature0 = np.vstack((neg_feature0, backbone_feat[0].cpu().numpy().reshape(1,-1)))
                    neg_feature1 = np.vstack((neg_feature1, backbone_feat[1].cpu().numpy().reshape(1,-1)))
                    neg_feature2 = np.vstack((neg_feature2, backbone_feat[2].cpu().numpy().reshape(1,-1)))
                    neg_feature3 = np.vstack((neg_feature3, backbone_feat[3].cpu().numpy().reshape(1,-1)))
                    neg_feature4 = np.vstack((neg_feature4, backbone_feat[4].cpu().numpy().reshape(1,-1)))
                    neg_feature5 = np.vstack((neg_feature5, backbone_feat[5].cpu().numpy().reshape(1,-1)))
                    neg_feature6 = np.vstack((neg_feature6, backbone_feat[6].cpu().numpy().reshape(1,-1)))
                    neg_feature7 = np.vstack((neg_feature7, backbone_feat[7].cpu().numpy().reshape(1,-1)))
                    neg_feature8 = np.vstack((neg_feature8, backbone_feat[8].cpu().numpy().reshape(1,-1)))
                    neg_feature9 = np.vstack((neg_feature9, backbone_feat[9].cpu().numpy().reshape(1,-1)))
                    neg_feature10 = np.vstack((neg_feature10, backbone_feat[10].cpu().numpy().reshape(1,-1)))
                    neg_feature11 = np.vstack((neg_feature11, backbone_feat[11].cpu().numpy().reshape(1,-1)))
            
            neg_features = [neg_feature0,neg_feature1,neg_feature2,neg_feature3,neg_feature4,neg_feature5,\
                neg_feature6,neg_feature7,neg_feature8,neg_feature9,neg_feature10,neg_feature11]
            for i in range(12):
                path = '/data1/Code/luandong/WWY_code_data/Codes/sigma_fusion/vis/vis_feat/tsne/tsne_file/{}'.format(seq_name)
                os.makedirs(path,exist_ok=True)
                np.savetxt(os.path.join(path, 'pos_{}_{}.txt'.format(self.frame_id, i)), pos_features[i], delimiter=',')
                np.savetxt(os.path.join(path, 'neg_{}_{}.txt'.format(self.frame_id, i)), neg_features[i], delimiter=',')
                print('{}_{}_{} saved'.format(seq_name,self.frame_id,i))

        return {"target_bbox": [1,1,1,1],
                    'w':1}


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
