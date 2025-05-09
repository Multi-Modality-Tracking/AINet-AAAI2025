import numpy as np
from lib.test.evaluation.data import Sequence, Sequence_RGBT, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os
from lib.utils.lmdb_utils import *

class RGBT234LmdbDataset(BaseDataset):
    # RGBt234 dataset
    def __init__(self, attr=None):
        super().__init__()
        self.base_path = self.env_settings.lmdb_path
        self.key_root = 'rgbt234.'
        self.sequence_list = self._get_sequence_list(attr=attr)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _read_bb_anno(self, seq_path):
        bb_anno_file = seq_path+'.init_lbl'
        gt_str_list = decode_str(self.base_path, bb_anno_file)  # the last line is empty
        gt_str_list = gt_str_list.split('\r\n')  if '\r\n' in gt_str_list else gt_str_list.split('\n')## the last line is empty
        while gt_str_list[-1]=='':
            del gt_str_list[-1]
        gt_list = [list(map(float, line.split(','))) for line in gt_str_list]
        # gt_list = [np.fromstring(line, sep=',') for line in gt_str_list]
        gt_arr = np.array(gt_list).astype(np.float32)
        return gt_arr
    

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        anno_path = sequence_info['anno_path']
        ground_truth_rect = self._read_bb_anno(sequence_path)
        frames_v = [[self.base_path, sequence_path+'.visible.'+str(i)] for i in range(ground_truth_rect.shape[0])]
        frames_i = [[self.base_path, sequence_path+'.infrared.'+str(i)] for i in range(ground_truth_rect.shape[0])]
        
        # Convert gt
        if ground_truth_rect.shape[1] > 4:
            gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
            gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

            x1 = np.amin(gt_x_all, 1).reshape(-1,1)
            y1 = np.amin(gt_y_all, 1).reshape(-1,1)
            x2 = np.amax(gt_x_all, 1).reshape(-1,1)
            y2 = np.amax(gt_y_all, 1).reshape(-1,1)

            ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)
        return Sequence_RGBT(sequence_info['name'], frames_v, frames_i, 'rgbt234', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_info_list)
        
    def _get_sequence_list(self, attr=None):
        sequence_list= ['blackwoman',
        'car3',
        'car4',
        'child1',
        'cycle1',
        'cycle2',
        'cycle3',
        'cycle4',
        'cycle5',
        'dog1',
        'man2',
        'manypeople1',
        'manypeople2',
        'orangeman1',
        'single1',
        'single3',
        'threepeople',
        'tricycle1',
        'twoman2',
        'twowoman1',
        'whiteman1',
        'woman1',
        'woman2',
        'woman3',
        'afterrain',
        'aftertree',
        'baby',
        'baginhand',
        'baketballwaliking',
        'balancebike',
        'basketball2',
        'bicyclecity',
        'bike',
        'bikeman',
        'bikemove1',
        'biketwo',
        'bluebike',
        'blueCar',
        'boundaryandfast',
        'bus6',
        'call',
        'car',
        'car10',
        'car20',
        'car37',
        'car41',
        'car66',
        'caraftertree',
        'carLight',
        'carnotfar',
        'carnotmove',
        'carred',
        'child',
        'child3',
        'child4',
        'children2',
        'children3',
        'children4',
        'crossroad',
        'crouch',
        'diamond',
        'dog',
        'dog10',
        'dog11',
        'elecbike',
        'elecbike2',
        'elecbike3',
        'elecbike10',
        'elecbikechange2',
        'elecbikeinfrontcar',
        'elecbikewithhat',
        'elecbikewithlight',
        'elecbikewithlight1',
        'face1',
        'floor-1',
        'flower1',
        'flower2',
        'fog',
        'fog6',
        'glass',
        'glass2',
        'graycar2',
        'green',
        'greentruck',
        'greyman',
        'greywoman',
        'guidepost',
        'hotglass',
        'hotkettle',
        'inglassandmobile',
        'jump',
        'kettle',
        'kite2',
        'kite4',
        'luggage',
        'man3',
        'man4',
        'man5',
        'man7',
        'man8',
        'man9',
        'man22',
        'man23',
        'man24',
        'man26',
        'man28',
        'man29',
        'man45',
        'man55',
        'man68',
        'man69',
        'man88',
        'manafterrain',
        'mancross',
        'mancross1',
        'mancrossandup',
        'mandrivecar',
        'manfaraway',
        'maninblack',
        'maninglass',
        'maningreen2',
        'maninred',
        'manlight',
        'manoccpart',
        'manonboundary',
        'manonelecbike',
        'manontricycle',
        'manout2',
        'manup',
        'manwithbag',
        'manwithbag4',
        'manwithbasketball',
        'manwithluggage',
        'manwithumbrella',
        'manypeople',
        'mobile',
        'night2',
        'nightcar',
        'nightrun',
        'nightthreepeople',
        'notmove',
        'oldman',
        'oldman2',
        'oldwoman',
        'people',
        'people1',
        'people3',
        'playsoccer',
        'push',
        'rainingwaliking',
        'raningcar',
        'redbag',
        'redcar',
        'redcar2',
        'redmanchange',
        'rmo',
        'run',
        'run1',
        'run2',
        'scooter',
        'shake',
        'shoeslight',
        'soccer',
        'soccer2',
        'soccerinhand',
        'straw',
        'stroller',
        'supbus',
        'supbus2',
        'takeout',
        'tallman',
        'threeman',
        'threeman2',
        'threewoman2',
        'together',
        'toy1',
        'toy3',
        'toy4',
        'tree2',
        'tree3',
        'tree5',
        'trees',
        'tricycle',
        'tricycle2',
        'tricycle6',
        'tricycle9',
        'tricyclefaraway',
        'tricycletwo',
        'twoelecbike',
        'twoelecbike1',
        'twoman',
        'twoman1',
        'twoperson',
        'twowoman',
        'walking40',
        'walking41',
        'walkingman',
        'walkingman1',
        'walkingman12',
        'walkingman20',
        'walkingman41',
        'walkingmantiny',
        'walkingnight',
        'walkingtogether',
        'walkingtogether1',
        'walkingtogetherright',
        'walkingwithbag1',
        'walkingwithbag2',
        'walkingwoman',
        'whitebag',
        'whitecar',
        'whitecar3',
        'whitecar4',
        'whitecarafterrain',
        'whitesuv',
        'woamn46',
        'woamnwithbike',
        'woman',
        'woman4',
        'woman6',
        'woman48',
        'woman89',
        'woman96',
        'woman99',
        'woman100',
        'womancross',
        'womanfaraway',
        'womaninblackwithbike',
        'womanleft',
        'womanpink',
        'womanred',
        'womanrun',
        'womanwithbag6',
        'yellowcar']

        if attr!=None:
            fp = os.path.join("dataset_attr/RGBT234_Attributes", attr+'.txt')
            with open(fp) as f:
                idx = f.read().split('\n')
            for i in range(len(sequence_list)-1, -1, -1):
                if '0' in idx[i]:
                    del sequence_list[i]


        sequence_info_list = []
        for i in range(len(sequence_list)):
            sequence_info = {}
            sequence_info["name"] = sequence_list[i] 
            sequence_info["path"] = self.key_root + sequence_info["name"]
            sequence_info["anno_path"] = sequence_info["path"]+'.init_lbl'
            #sequence_info["object_class"] = 'person'
            sequence_info_list.append(sequence_info)
        return sequence_info_list
    