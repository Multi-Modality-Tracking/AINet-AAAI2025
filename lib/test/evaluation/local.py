from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data/got10k_lmdb'
    settings.got10k_path = '/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.gtot_path = '/data/GTOT/'
    settings.itb_path = '/data/itb'
    settings.lasot_extension_subset_path_path = '/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/data/lasot_lmdb'
    settings.lasot_path = '/data/lasot'
    settings.network_path = '/data1/Code/luandong/WWY_code_data/Codes/AINet/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data/nfs'
    settings.otb_path = '/data/otb'
    settings.prj_dir = '/data1/Code/luandong/WWY_code_data/Codes/AINet'
    settings.result_plot_path = '/data1/Code/luandong/WWY_code_data/Codes/AINet/output/test/result_plots'
    settings.results_path = '/data1/Code/luandong/WWY_code_data/Codes/AINet'    # Where to store tracking results
    settings.rgbt210_path = '/data/RGBT210/'
    settings.rgbt234_path = '/data/RGBT234/'
    settings.save_dir = '/data1/Code/luandong/WWY_code_data/Codes/AINet/output'
    settings.segmentation_path = '/data1/Code/luandong/WWY_code_data/Codes/AINet/output/test/segmentation_results'
    settings.tc128_path = '/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data/trackingnet'
    settings.uav_path = '/data/uav'
    settings.vot18_path = '/data/vot2018'
    settings.vot22_path = '/data/vot2022'
    settings.vot_path = '/data/VOT2019'
    settings.youtubevos_dir = ''

    settings.lasher_path = '/data1/Datasets/Tracking/LasHeR/'
    settings.rgbt234_path = '/data1/Datasets/Tracking/RGBT234/'
    settings.rgbt210_path = '/data1/Datasets/Tracking/RGBT210/'
    settings.gtot_path = '/data1/Datasets/Tracking/GTOT/'
    settings.vtuav_path = '/data1/Datasets/Tracking/VTUAV/'
    settings.visevent_testingset_dir = '/data1/Datasets/Tracking/visevent/'
    settings.dmet_path = '/data1/Datasets/Tracking/DMET/'

    return settings

