import os
import os.path
import torch,cv2
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader,opencv_loader
from lib.train.admin import env_settings

class DMET_trainingSet(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, data_fraction=None, attr=None):
        self.root = env_settings().dmet_dir if root is None else root
        super().__init__('DMET_trainingSet', root, image_loader)

        # video_name for each sequence
        with open('evaluate_DMET/train_set.txt','r') as f:
            seq_list =  [line.strip() for line in f]
        # video_name for each sequence
        self.sequence_list = seq_list
        
    def get_name(self):
        return 'DMET_trainingSet'

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, 'init.txt')
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)

    # def _get_sequence_path(self, seq_id):
    #     return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        #print('seq_id', seq_id)
        seq_name = self.sequence_list[seq_id]
        #print('seq_name', seq_name)
        seq_path = os.path.join(self.root, seq_name)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        dict_ = {'bbox': bbox, 'valid': valid, 'visible': visible}
        # return {'bbox': bbox, 'valid': valid, 'visible': visible},seq_name
        return dict_

    def _get_frame_v(self, seq_path, frame_id):
        frame_path_v = os.path.join(seq_path, 'up', sorted([p for p in os.listdir(os.path.join(seq_path, 'up')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        return self.image_loader(frame_path_v)
        
    def _get_frame_i(self, seq_path, frame_id):
        frame_path_i = os.path.join(seq_path, 'down', sorted([p for p in os.listdir(os.path.join(seq_path, 'down')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        return self.image_loader(frame_path_i)


    def _get_frame_path(self, seq_path, frame_id):
        vis_frame_names = sorted(os.listdir(os.path.join(seq_path, 'up')))
        inf_frame_names = sorted(os.listdir(os.path.join(seq_path, 'down')))
        return os.path.join(seq_path, 'up', vis_frame_names[frame_id]), os.path.join(seq_path, 'down', inf_frame_names[frame_id])   
    #修改为concatenate 和tbsi一样
    def _get_frame(self,seq_path,frame_id):
        frame_path_v = os.path.join(seq_path, 'up', sorted([p for p in os.listdir(os.path.join(seq_path, 'up')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        frame_path_i = os.path.join(seq_path, 'down', sorted([p for p in os.listdir(os.path.join(seq_path, 'down')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        return np.concatenate((self.image_loader(frame_path_v),self.image_loader(frame_path_i)), 2)


    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_name = self.sequence_list[seq_id]
        seq_path = os.path.join(self.root, seq_name)
        # frame_list = [self._get_frame(seq_path, f) for f in frame_ids]
        frame_list_v = [self._get_frame_v(seq_path, f) for f in frame_ids]
        frame_list_i = [self._get_frame_i(seq_path, f) for f in frame_ids]

        #print(frame_ids)
        #print(len(frame_list_v),len(frame_list_i),len(frame_list))
        #@print(len(frame_list_i))
        frame_list  = frame_list_v + frame_list_i # 6
        
        #print(len(frame_list))
        #exit()
        if seq_name not in self.sequence_list:
            print('warning!!!'*100)
        if anno is None:
            anno = self.get_sequence_info(seq_path)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        #return frame_list_v, frame_list_i, anno_frames, object_meta
        return frame_list, anno_frames, object_meta

def get_frame(seq_pathrgb,seq_pathtir):
    # frame_path_v = os.path.join(seq_pathrgb, '', sorted([p for p in os.listdir(seq_pathrgb) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
    # frame_path_i = os.path.join(seq_pathtir, '', sorted([p for p in os.listdir(seq_pathtir) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
    # imgrgb = jpeg4py_loader(seq_pathrgb)
    # imgtir = jpeg4py_loader(seq_pathtir)
    #for png image
    imgrgb = opencv_loader(seq_pathrgb)
    imgtir = opencv_loader(seq_pathtir)


    actual_height, actual_width = imgrgb.shape[:2]
    actual_heighttir, actual_widthtir = imgtir.shape[:2]
        # 检查尺寸是否与期望的尺寸不符
    if (actual_width, actual_height) != (actual_heighttir, actual_widthtir):
        imgrgb = cv2.resize(imgrgb, (actual_widthtir,actual_heighttir), interpolation=cv2.INTER_AREA)
        # print(actual_height, actual_width)
    return np.concatenate((imgrgb,imgtir), 2)