class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data1/Code/luandong/WWY_code_data/Codes/AINet'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data1/Code/luandong/WWY_code_data/Codes/AINet/saved/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/data1/Code/luandong/WWY_code_data/Codes/AINet/pretrained_networks'

        self.lasot_dir = '/data/lasot'
        self.got10k_dir = '/data/got10k/train'
        self.got10k_val_dir = '/data/got10k/val'
        self.lasot_lmdb_dir = '/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/data/got10k_lmdb'
        self.trackingnet_dir = '/data/trackingnet'
        self.trackingnet_lmdb_dir = '/data/trackingnet_lmdb'
        self.coco_dir = '/data/coco'
        self.coco_lmdb_dir = '/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/data/vid'
        self.imagenet_lmdb_dir = '/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        
        self.lasher_dir = '/data1/Datasets/Tracking/LasHeR/'
        self.vtuav_dir = "/data1/Datasets/Tracking/VTUAV/"
        self.rgbt234_dir = '/data1/Datasets/Tracking/RGBT234/'
        self.visevent_trainingset_dir = '/data1/Datasets/Tracking/visevent/'
        self.visevent_testingset_dir = '/data1/Datasets/Tracking/visevent/'
        self.dmet_dir = '/data1/Datasets/Tracking/DMET/'
        