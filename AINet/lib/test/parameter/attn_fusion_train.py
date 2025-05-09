from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.attn_fusion_train.config import cfg, update_config_from_file


def parameters(yaml_name: str, checkpoint_path=None, update=[1., 0., 0.]):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/attn_fusion_train/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    if checkpoint_path!=None:
        params.checkpoint = checkpoint_path
    else:
        params.checkpoint = "../output/checkpoints/train/attn_fusion_train/%s/OSTrack_twobranch_ep%04d.pth.tar" % (yaml_name, cfg.TEST.EPOCH)

    params.update = update

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
