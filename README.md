# AINet
RGBT Tracking via All-layer Multimodal Interactions with Progressive Fusion Mamba

## Dataset
We use the LasHeR training set for training, GTOT, RGBT210, RGBT234, LasHeR testing set, VTUAVST for testing, and their project addresses are as follows:
* [GTOT](http://chenglongli.cn/code-dataset/)
* [RGBT210](http://chenglongli.cn/code-dataset/)
* [RGBT234](http://chenglongli.cn/code-dataset/)
* [LasHeR](https://github.com/BUGPLEASEOUT/LasHeR)
* [VTUAV](https://github.com/zhang-pengyu/DUT-VTUAV)

## Environment Preparation
1. Our code is trained and tested with Python == 3.8.13, PyTorch == 2.1.1 and CUDA == 11.8 on NVIDIA GeForce RTX 4090.
2. Install causal_conv1d and mamba from our repo.

## Training
1. We adopt [OSTrack](https://github.com/botaoye/OSTrack) as our base tracker. For our best result, we need to load the parameter from [DropMAE](https://github.com/jimmy-dq/DropTrack).
2. Modify the relevant dataset and pretrained model paths, Then run the following command.
```
python lib/train/run_training.py --script ostrack_twobranch --config 384 --save_dir your_save_dir
```

## Evaluation
Modify the relevant dataset and checkpoint paths, then run the following command. 
```
python tracking/test.py --tracker_name ostrack_twobranch --tracker_param 384 --checkpoint_path your_checkpoint_path
```

## Results and Models
| Model | RGBT210(PR/SR) | RGBT234(PR/SR) | LasHeR(PR/NPR/SR) | VTUAV(PR/SR) | Checkpoint | Raw Result |
|:-------:|:----------------:|:----------------:|:-------------------:|:--------------:|:--------------:|:--------------:|
| AFter |       |      |     |     | [download]() | [download]()
