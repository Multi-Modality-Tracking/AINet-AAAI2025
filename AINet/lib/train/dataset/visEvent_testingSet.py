import os
import os.path
import cv2 as cv
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from data.image_loader import jpeg4py_loader
from admin.environment import env_settings
from lib.utils.load_text import load_text


class visEvent_testingSet(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, data_fraction=None):
        self.root = env_settings().visevent_testingset_dir if root is None else root
        super().__init__('visEvent_testingSet', root, image_loader)

        # video_name for each sequence
        self.sequence_list = ['dvSave-2021_02_15_13_02_21_Chicken', 'video_0018', '00466_UAV_outdoor6', 'dvSave-2021_02_04_20_41_53', 'dvSave-2021_02_14_16_56_18_car2', 'dvSave-2021_02_06_09_09_44_person6', 'dvSave-2021_02_15_13_12_45_redcar', 'dvSave-2021_02_14_16_40_59_car1', '00325_UAV_outdoor5', '00340_UAV_outdoor6', 'dvSave-2021_02_15_12_58_05_personHead2', 'dvSave-2021_02_06_08_33_09_cat', 'dvSave-2021_02_06_17_20_28_personFootball', 'dvSave-2021_02_15_13_10_54_person', 'dvSave-2021_02_12_13_43_54', '00398_UAV_outdoor6', 'tennis_long_001', 'dvSave-2021_02_14_16_53_15_flag', 'dvSave-2021_02_14_16_45_13_car5', 'traffic_0070', 'dvSave-2021_02_14_16_37_15_motor2', 'dvSave-2021_02_06_09_36_44_Pedestrian', 'dvSave-2021_02_06_09_09_44_person7', 'dvSave-2021_02_15_13_05_43_Chicken', 'dvSave-2021_02_14_16_56_18_car6', 'dvSave-2021_02_14_16_26_44_girl', 'dvSave-2021_02_14_16_34_11_car3', 'dvSave-2021_02_06_15_08_41_flag', 'dvSave-2021_02_06_09_24_39_oldman1', 'traffic_0043', 'video_0056', 'dvSave-2021_02_16_17_23_10', 'traffic_0034', 'dvSave-2021_02_14_16_43_54_car4', 'dvSave-2021_02_06_15_16_07_car', 'tennis_long_003', 'traffic_0073', 'dvSave-2021_02_06_09_22_41_person1', 'dvSave-2021_02_14_16_43_23_car4', 'dvSave-2021_02_15_10_23_05_boyhead', 'roadLight_001', 'dvSave-2021_02_12_13_46_18', 'dvSave-2021_02_06_09_11_41_car1', 'video_0039', '00413_UAV_outdoor6', '00410_UAV_outdoor6', '00385_UAV_outdoor6', 'dvSave-2021_02_06_08_58_43_cat', 'dvSave-2021_02_06_09_17_11_person', 'dvSave-2021_02_08_21_17_43_car3', 'dvSave-2021_02_06_09_16_06_person', 'dvSave-2021_02_04_21_04_05', 'dvSave-2021_02_04_21_21_24', 'dvSave-2021_02_06_17_49_51_personBasketball', 'dvSave-2021_02_06_17_31_03_personBasketball', '00351_UAV_outdoor6', 'dvSave-2021_02_06_17_34_58_personBasketball', 'dvSave-2021_02_15_10_26_52_personHead1', 'basketball_0076', 'dvSave-2021_02_06_15_15_36_redcar', 'dvSave-2021_02_06_08_57_35_machineBrad', 'dvSave-2021_02_04_20_56_55', 'traffic_0013', 'video_0076', 'dvSave-2021_02_06_08_56_18_windowPattern', 'dvSave-2021_02_06_18_04_18_person1', 'video_0005', '00241_tennis_outdoor4', 'dvSave-2021_02_06_09_09_44_person3', '00335_UAV_outdoor5', 'dvSave-2021_02_08_21_15_49_car1', 'dvSave-2021_02_14_16_31_07_person1', 'dvSave-2021_02_06_09_10_52_car2', 'dvSave-2021_02_15_13_27_20_bottle', 'video_0032', 'traffic_0064', 'traffic_0067', 'dvSave-2021_02_06_09_10_52_car3', 'video_0009', 'video_0064', 'dvSave-2021_02_06_10_09_04_bottle', 'dvSave-2021_02_08_21_07_02_car2', 'dvSave-2021_02_15_12_53_54_personHead', '00236_tennis_outdoor4', 'dvSave-2021_02_06_09_33_23_person1', 'dvSave-2021_02_14_16_34_11_person1', 'dvSave-2021_02_14_16_31_07_whitecar4', 'dvSave-2021_02_06_15_14_26_blackcar', 'dvSave-2021_02_14_16_28_37_car2', 'dvSave-2021_02_08_21_15_49_car8', 'dvSave-2021_02_06_09_14_18_person5', '00419_UAV_outdoor6', 'dvSave-2021_02_06_09_11_41_person1', 'dvSave-2021_02_12_13_39_56', 'video_0049', '00141_tank_outdoor2', 'dvSave-2021_02_16_17_07_38', 'dvSave-2021_02_14_16_46_34_car5', 'dvSave-2021_02_15_10_26_52_basketball2', 'traffic_0006', 'dvSave-2021_02_06_09_58_27_DigitAI', 'dvSave-2021_02_14_16_46_34_car10', 'dvSave-2021_02_12_13_38_26', 'dvSave-2021_02_14_16_28_37_car3', '00297_tennis_outdoor4', 'dvSave-2021_02_14_16_56_59_car6', 'dvSave-2021_02_12_13_51_43', '00478_UAV_outdoor6', 'dvSave-2021_02_14_16_43_54_car2', 'dvSave-2021_02_06_17_53_39_personFootball', 'dvSave-2021_02_16_17_38_25', '00437_UAV_outdoor6', '00514_person_outdoor6', 'dvSave-2021_02_14_16_48_45_car8', 'dvSave-2021_02_06_09_13_36_person2', '00435_UAV_outdoor6', 'dvSave-2021_02_06_17_27_53_personFootball', 'video_0008', 'dvSave-2021_02_15_10_24_03_basketball', 'dvSave-2021_02_06_09_09_44_blackcar', 'tennis_long_005', '00483_UAV_outdoor6', 'dvSave-2021_02_16_17_20_20', 'dvSave-2021_02_15_13_24_03_girlhead', 'traffic_0049', 'dvSave-2021_02_06_15_17_48_whitecar', 'dvSave-2021_02_12_13_56_29', '00408_UAV_outdoor6', 'dvSave-2021_02_14_16_31_07_whitecar1', 'UAV_long_001', 'dvSave-2021_02_14_16_56_59_car4', 'dvSave-2021_02_14_16_30_05_car1', 'dvSave-2021_02_06_09_35_08_Pedestrian', 'dvSave-2021_02_14_16_29_49_car2', 'dvSave-2021_02_15_13_01_16_Duck', '00473_UAV_outdoor6', 'dvSave-2021_02_14_17_00_48', 'traffic_0058', 'traffic_0037', 'dvSave-2021_02_06_09_33_23_person5', '00282_tennis_outdoor4', 'dvSave-2021_02_14_16_34_48_person1', '00147_tank_outdoor2', '00503_UAV_outdoor6', 'dvSave-2021_02_08_21_06_03_car7', 'dvSave-2021_02_14_16_35_40_car2', 'dvSave-2021_02_14_16_56_01_house', 'dvSave-2021_02_04_20_49_43', 'dvSave-2021_02_14_16_56_59_car2', 'dvSave-2021_02_15_23_56_17', 'dvSave-2021_02_14_16_40_59_car7', '00471_UAV_outdoor6', 'dvSave-2021_02_14_16_37_15_car2', 'dvSave-2021_02_06_09_23_50_person1', 'dvSave-2021_02_08_21_07_52', 'dvSave-2021_02_06_09_21_53_car', '00197_driving_outdoor3', '00506_person_outdoor6', '00421_UAV_outdoor6', 'dvSave-2021_02_06_17_51_05_personBasketball', '00451_UAV_outdoor6', '00442_UAV_outdoor6', '00453_UAV_outdoor6', 'dvSave-2021_02_06_09_36_15_Pedestrian', 'dvSave-2021_02_08_21_02_13_motor2', 'dvSave-2021_02_15_12_58_56_person', 'dvSave-2021_02_14_16_37_15_person', 'dvSave-2021_02_14_16_46_34_car12', 'dvSave-2021_02_15_13_28_20_cash', 'dvSave-2021_02_14_16_40_59_blackcar1', 'dvSave-2021_02_14_16_22_06', 'dvSave-2021_02_15_12_56_56_Fish', 'dvSave-2021_02_14_16_43_23_car3', 'tennis_long_004', 'video_0058', '00430_UAV_outdoor6', 'video_0029', 'dvSave-2021_02_06_17_47_49_personBasketball', 'dvSave-2021_02_06_17_33_01_personBasketball', '00423_UAV_outdoor6', 'dvSave-2021_02_15_10_14_18_chicken', 'dvSave-2021_02_14_16_48_45_car5', '00508_person_outdoor6', 'dvSave-2021_02_08_21_15_49_car3', 'dvSave-2021_02_06_10_14_17_paperClip', 'dvSave-2021_02_06_18_04_18_person3', 'dvSave-2021_02_15_23_54_17', 'dvSave-2021_02_15_10_12_19_basketball', 'video_0026', 'traffic_0052', 'tennis_long_007', 'dvSave-2021_02_06_17_41_45_personBasketball', 'dvSave-2021_02_15_13_25_36_girlhead', 'dvSave-2021_02_14_16_51_21_car1', '00370_UAV_outdoor6', 'dvSave-2021_02_08_21_17_43_car5', 'video_0004', 'dvSave-2021_02_06_17_45_17_personBasketball', '00416_UAV_outdoor6', 'dvSave-2021_02_14_16_45_13_car1', '00355_UAV_outdoor6', 'dvSave-2021_02_08_21_06_03_car5', 'dvSave-2021_02_14_16_21_40', 'dvSave-2021_02_15_23_51_36', 'dvSave-2021_02_06_08_52_19_rotateball', 'video_0050', 'dvSave-2021_02_06_10_11_59_paperClips', 'dvSave-2021_02_14_16_55_35_person1', 'video_0070', 'dvSave-2021_02_14_16_56_18_car4', 'dvSave-2021_02_15_13_24_49_girlhead', 'dvSave-2021_02_06_09_14_18_whitecar1', '00425_UAV_outdoor6', 'dvSave-2021_02_06_17_36_49_personBasketball', 'dvSave-2021_02_06_17_23_26_personFootball', 'dvSave-2021_02_16_17_29_37', 'dvSave-2021_02_14_17_02_37_roadflag', 'dvSave-2021_02_08_21_05_56_motor', '00314_UAV_outdoor5', '00464_UAV_outdoor6', 'dvSave-2021_02_15_12_56_56_personHead', '00445_UAV_outdoor6', 'dvSave-2021_02_15_10_22_23_boyhead', 'video_0041', '00406_UAV_outdoor6', 'traffic_0055', '00458_UAV_outdoor6', 'dydrant_001', 'tennis_long_006', 'dvSave-2021_02_14_16_37_15_car5', 'tennis_long_002', 'dvSave-2021_02_16_17_42_50', 'traffic_0040', 'dvSave-2021_02_14_16_28_37_person1', 'video_0054', '00374_UAV_outdoor6', 'dvSave-2021_02_15_13_14_18_blackcar', 'dvSave-2021_02_14_16_42_14_car1', 'dvSave-2021_02_08_21_04_56_car6', 'dvSave-2021_02_14_16_48_45_car3', 'dvSave-2021_02_16_17_34_11', 'dvSave-2021_02_14_16_26_44_car3', 'dvSave-2021_02_15_10_26_11_chicken', 'dvSave-2021_02_14_16_40_59_motor1', 'dvSave-2021_02_15_13_08_12_blackcar', 'dvSave-2021_02_14_16_30_20_car2', 'traffic_0046', 'dvSave-2021_02_14_16_31_07_redtaxi01', 'dvSave-2021_02_15_10_22_23_basketball', 'dvSave-2021_02_06_09_16_35_car', 'video_0067', 'video_0079', 'dvSave-2021_02_06_17_57_54_personFootball', 'dvSave-2021_02_06_10_05_38_phone', 'dvSave-2021_02_06_17_16_26_whitecar', 'dvSave-2021_02_04_21_20_22', 'dvSave-2021_02_06_09_10_52_car1', '00404_UAV_outdoor6', '00462_UAV_outdoor6', '00292_tennis_outdoor4', '00449_UAV_outdoor6', 'traffic_0061', 'dvSave-2021_02_14_16_45_13_car7', 'dvSave-2021_02_08_21_06_03_motor2', 'dvSave-2021_02_15_10_24_03_boyhead', 'dvSave-2021_02_15_13_04_57_Duck', 'dvSave-2021_02_14_16_46_34_car3', '00447_UAV_outdoor6', 'traffic_0023', 'dvSave-2021_02_06_09_13_36_person0', '00331_UAV_outdoor5', 'dvSave-2021_02_06_08_56_40_windowPattern2', 'dvSave-2021_02_15_13_13_44_whitecar', '00490_UAV_outdoor6', 'dvSave-2021_02_06_10_03_17_GreenPlant', 'dvSave-2021_02_15_12_45_02_Duck', 'dvSave-2021_02_15_12_44_27_chicken', '00432_UAV_outdoor6', 'dvSave-2021_02_14_16_46_34_car8', 'video_0015', 'dvSave-2021_02_06_09_14_18_girl1', 'dvSave-2021_02_06_17_21_41_personFootball', 'dvSave-2021_02_06_17_15_20_whitecar', 'dvSave-2021_02_06_15_12_44_car', 'dvSave-2021_02_15_13_09_09_person', 'traffic_0019', 'dvSave-2021_02_14_16_34_48_car2', 'dvSave-2021_02_06_10_17_16_paperClips', '00510_person_outdoor6', '00345_UAV_outdoor6', 'traffic_0028', 'dvSave-2021_02_14_16_46_34_car16', 'dvSave-2021_02_08_21_04_56_car3', 'dvSave-2021_02_16_17_15_53', 'dvSave-2021_02_08_21_06_03_car3', 'dvSave-2021_02_04_21_18_52', '00433_UAV_outdoor6', 'basketball_0078', 'dvSave-2021_02_14_16_42_44_car6', 'dvSave-2021_02_14_16_40_59_car4', 'dvSave-2021_02_06_15_18_36_redcar', '00455_UAV_outdoor6', 'dvSave-2021_02_06_09_11_41_person4', 'dvSave-2021_02_14_16_26_44_person1', 'dvSave-2021_02_16_17_12_18', 'dvSave-2021_02_15_10_26_52_basketball1', '00428_UAV_outdoor6', 'dvSave-2021_02_14_16_51_21_motor1', 'video_0021', 'video_0060', 'dvSave-2021_02_08_21_02_13_car3', 'dvSave-2021_02_15_10_23_05_basketall', '00439_UAV_outdoor6', 'video_0045', '00511_person_outdoor6', 'dvSave-2021_02_14_16_31_07_blackcar2', 'video_0073', 'dvSave-2021_02_14_16_42_44_car3', 'dvSave-2021_02_06_09_24_26_Pedestrian1', 'dightNUM_001']
        
        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))
        
    def get_name(self):
        return 'visEvent_testingSet'

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, 'groundtruth.txt')
        # gt = np.loadtxt(bb_anno_file, delimiter=',', dtype=np.float32)
        gt = load_text(bb_anno_file, delimiter=[',', ' ', '\t'])
        return torch.tensor(gt)

    def get_sequence_info(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        seq_path = os.path.join(self.root, seq_name)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_v(self, seq_path, frame_id):
        frame_path_v = os.path.join(seq_path, 'vis_imgs', sorted([p for p in os.listdir(os.path.join(seq_path, 'vis_imgs')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        # return self.image_loader(frame_path_v)
        img=cv.imread(frame_path_v)
        return cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
    def _get_frame_i(self, seq_path, frame_id):
        frame_path_i = os.path.join(seq_path, 'event_imgs', sorted([p for p in os.listdir(os.path.join(seq_path, 'event_imgs')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        # return self.image_loader(frame_path_i)
        img=cv.imread(frame_path_i)
        return cv.cvtColor(img, cv.COLOR_BGR2RGB)

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_name = self.sequence_list[seq_id]
        seq_path = os.path.join(self.root, seq_name)
        frame_list_v = [self._get_frame_v(seq_path, f) for f in frame_ids] # 其中的元素都是用cv读取出的图片
        frame_list_i = [self._get_frame_i(seq_path, f) for f in frame_ids]

        frame_list  = frame_list_v + frame_list_i # 6
        if seq_name not in self.sequence_list:
            print('warning!!!'*100)
        if anno is None:
            anno = self.get_sequence_info(seq_path)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
            # anno_frames = {'bbox': [Tensor([x1,y1,h1,w1], Tensor([x2,y2,h2,w2])], 'valid': ..., 'visible': ...}

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        #return frame_list_v, frame_list_i, anno_frames, object_meta
        return frame_list, anno_frames, object_meta

    def get_modal(self, seq_id=None):
        return 'V','E'