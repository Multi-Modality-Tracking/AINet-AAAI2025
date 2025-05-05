import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
from lib.test.evaluation.data import Sequence, Sequence_RGBT, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text

class DMET_testingSet(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None,data_fraction=None, attr=None):
        self.root = env_settings().dmet_dir if root is None else root
        super().__init__('DMET_testingSet', root, image_loader)

        # video_name for each sequence
        with open('evaluate_DMET/test_set.txt','r') as f:
            seq_list =  [line.strip() for line in f]
        # video_name for each sequence
        self.sequence_list = seq_list

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

    def get_sequence_list(self):
        # video_name for each sequence
        with open('evaluate_DMET/test_set.txt','r') as f:
            seq_list =  [line.strip() for line in f]
        sequence_info_list = []
        for i in range(len(seq_list)):
            sequence_info = {}
            sequence_info["name"] = seq_list[i] 
            sequence_info["path"] = '/data1/Datasets/Tracking/DMET/'+sequence_info["name"]
            #sequence_info["startFrame"] = int('1')
            #print(end_frame[i])
            #sequence_info["endFrame"] = end_frame[i]
                
            #sequence_info["nz"] = int('6')
            #sequence_info["ext"] = 'jpg'
            sequence_info["anno_path"] = sequence_info["path"]+'/init.txt'
            #sequence_info["object_class"] = 'person'
            sequence_info_list.append(sequence_info)
        self.sequence_list = sequence_info_list   
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        anno_path = sequence_info['anno_path']
        ground_truth_rect = load_text(str(anno_path), delimiter=['', '\t', ','], dtype=np.float64)
        img_list_v = sorted([p for p in os.listdir(os.path.join(sequence_path, 'up')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        frames_v = [os.path.join(sequence_path, 'up', img) for img in img_list_v]
        
        img_list_i = sorted([p for p in os.listdir(os.path.join(sequence_path, 'down')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        frames_i = [os.path.join(sequence_path, 'down', img) for img in img_list_i]
        # Convert gt
        if ground_truth_rect.shape[1] > 4:
            gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
            gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

            x1 = np.amin(gt_x_all, 1).reshape(-1,1)
            y1 = np.amin(gt_y_all, 1).reshape(-1,1)
            x2 = np.amax(gt_x_all, 1).reshape(-1,1)
            y2 = np.amax(gt_y_all, 1).reshape(-1,1)

            ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)
        return Sequence_RGBT(sequence_info['name'], frames_v, frames_i, 'lasher', ground_truth_rect)




    def get_name(self):
        return 'DMET_testingSet'

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, 'init.txt')
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)

    def get_sequence_info(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        seq_path = os.path.join(self.root, seq_name)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_v(self, seq_path, frame_id):
        frame_path_v = os.path.join(seq_path, 'up', sorted([p for p in os.listdir(os.path.join(seq_path, 'up')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        return self.image_loader(frame_path_v)
        
    def _get_frame_i(self, seq_path, frame_id):
        frame_path_i = os.path.join(seq_path, 'down', sorted([p for p in os.listdir(os.path.join(seq_path, 'down')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        return self.image_loader(frame_path_i)

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
        frame_list  = frame_list_v + frame_list_i # 6
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

        # if seq_name not in self.sequence_list:
        #     print('warning!!!'*100)
        # if anno is None:
        #     anno = self.get_sequence_info(seq_path)

        # anno_frames = {}
        # for key, value in anno.items():
        #     anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        # object_meta = OrderedDict({'object_class_name': None,
        #                            'motion_class': None,
        #                            'major_class': None,
        #                            'root_class': None,
        #                            'motion_adverb': None})

        # #return frame_list_v, frame_list_i, anno_frames, object_meta
        # return frame_list, anno_frames, object_meta
