import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker
import time



def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                num_gpus=8, checkpoint_path=None, update=[1., 0., 0.]):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id, checkpoint_path=checkpoint_path, debug=debug, update=update)]

    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)

#nohup python -u /data1/Code/luandong/WWY_code_data/Codes/sigma_fusion/tracking/test.py > /data1/Code/luandong/WWY_code_data/Codes/sigma_fusion/test256.out 2>&1 &
def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--tracker_name',default='ostrack_twobranch', type=str, help='Name of tracking method.')
    parser.add_argument('--tracker_param',default='384', type=str, help='Name of config file.')
    parser.add_argument('--runid', type=int, default=145, help='The run id.')
    # 1:f2cross, 2:cross2f
    # 70:ab_modalfuse, 71:ab_crosslayer
    parser.add_argument('--dataset_name', type=str, default='lashertestingset', help='Name of dataset (lashertestingset, rgbt234, vtuav, viseventtestingset).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--vis_gpus', type=str, default='0,1,2,3')
    parser.add_argument('--checkpoint_path', type=str, default='/data1/Code/luandong/WWY_code_data/datasets/ainet_tmp/saved/v5/checkpoints/train/ostrack_twobranch/vitb_256_mae_ce_32x4_ep300/OSTrack_twobranch_ep0014.pth.tar')
    parser.add_argument('--update', type=float, default=[1.,0.,0.], nargs='+') # 模板更新【score阈值|RGB贡献值|TIR贡献值, 第1项设成1表示不更1】
    parser.add_argument('--wait', type=int, default=0)  # 定时【分钟】



    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    time.sleep(args.wait*60)
    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                args.threads, num_gpus=args.num_gpus, checkpoint_path=args.checkpoint_path, update=args.update)


if __name__ == '__main__':
    main()

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # run_tracker('ostrack_twobranch', 'vitb_256_mae_ce_32x4_ep300', 2535201, 'DMET_test', None, 0,
    #             0, num_gpus=1)