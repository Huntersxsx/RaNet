# RaNet: Relation-aware Video Reading Comprehension for Temporal Language Grounding

## Introduction

This is an implementation repository for our work in EMNLP 2021.
**Relation-aware Video Reading Comprehension for Temporal Language Grounding**. [arxiv paper](https://arxiv.org/abs/2110.05717)

![](https://github.com/Huntersxsx/RaNet/blob/master/img/framework.png)

## Note:
Our pre-trained models are available at [SJTU jbox](https://jbox.sjtu.edu.cn/l/215Z2T) or [baiduyun, passcode:xmc0](https://pan.baidu.com/s/1CRojAlDURJ57tUprdNbfFg) or [Google Drive](https://drive.google.com/drive/folders/1AFdgfxFCA9ji36HaveL2dQ7wr7OjlHjb?usp=sharing).
<!-- The repository contains the development code. This preview is intended for the reviewers of our AAAI2022 submission.
The code provided allows for evaluating our pretrained models. We will release the final version of the code on our official GitHub repo soon.
We discourage the reviewers from distributing this repository to third party users. Please follow the instructions below for the installation and download of necessary data.  -->

## Installation

Clone the repository and move to folder:
```bash
git clone https://github.com/Huntersxsx/RaNet.git

cd RaNet
```

To use this source code, you need Python3.7+ and a few python3 packages:
- pytorch 1.1.0
- torchvision 0.3.0
- torchtext
- easydict
- terminaltables
- tqdm

## Data
We use the data offered by [2D-TAN](https://rochester.app.box.com/s/8znalh6y5e82oml2lr7to8s6ntab6mav).

</br>

The folder structure should be as follows:
```
.
├── checkpoints
│   ├── best
│   │    ├── TACoS
│   │    ├── ActivityNet
│   │    └── Charades
├── data
│   ├── TACoS
│   │    ├── tall_c3d_features.hdf5
│   │    └── ...
│   ├── ActivityNet
│   │    ├── sub_activitynet_v1-3.c3d.hdf5
│   │    └── ...
│   ├── Charades-STA
│   │    ├── charades_vgg_rgb.hdf5
│   │    └── ...
│
├── experiments
│
├── lib
│   ├── core
│   ├── datasets
│   └── models
│
└── moment_localization
```

## Train and Test
Please download the visual features from [box drive](https://rochester.app.box.com/s/8znalh6y5e82oml2lr7to8s6ntab6mav) and save it to the `data/` folder.

#### Training
Use the following commands for training:
- For TACoS dataset, run: 
```bash
    sh run_tacos.sh
```
- For ActivityNet-Captions dataset, run:
```bash
    sh run_activitynet.sh
```
- For Charades-STA dataset, run:
```bash
    sh run_charades.sh
```

#### Testing
Our trained model are provided in [SJTU jbox](https://jbox.sjtu.edu.cn/l/215Z2T) or [baiduyun, passcode:xmc0](https://pan.baidu.com/s/1CRojAlDURJ57tUprdNbfFg) or [Google Drive](https://drive.google.com/drive/folders/1AFdgfxFCA9ji36HaveL2dQ7wr7OjlHjb?usp=sharing). Please download them to the `checkpoints/best/` folder.
Use the following commands for testing:
- For TACoS dataset, run: 
```bash
    sh test_tacos.sh
```
- For ActivityNet-Captions dataset, run:
```bash
    sh test_activitynet.sh
```
- For Charades-STA dataset, run:
```bash
    sh test_charades.sh
```

## Main results:

| **TACoS** | Rank1@0.3 | Rank1@0.5 | Rank5@0.3 | Rank5@0.5 |
| ---- |:-------------:| :-----:|:-----:|:-----:|
| **RaNet** |  43.34 | 33.54 |  67.33 | 55.09 |
</br>

| **ActivityNet** | Rank1@0.5 | Rank1@0.7 | Rank5@0.6 | Rank5@0.7 |
| ---- |:-------------:| :-----:|:-----:|:-----:|
| **RaNet** | 45.59 | 28.67 | 75.93 | 62.97 |
</br>

| **Charades (VGG)**  | Rank1@0.5 | Rank1@0.7 | Rank5@0.5 | Rank5@0.7 |
| ---- |:-------------:| :-----:|:-----:|:-----:|
| **RaNet** | 43.87 | 26.83 | 86.67 | 54.22 |
</br>

| **Charades (I3D)**  | Rank1@0.5 | Rank1@0.7 | Rank5@0.5 | Rank5@0.7 |
| ---- |:-------------:| :-----:|:-----:|:-----:|
| **RaNet** | 60.40 | 39.65 | 89.57 | 64.54 |

## Acknowledgement

We greatly appreciate the [2D-Tan repository](https://github.com/microsoft/2D-TAN), [gtad repository](https://github.com/frostinassiky/gtad) and [CCNet repository](https://github.com/speedinghzl/CCNet). Please remember to cite the papers:

```

@article{gao2021relation,
  title={Relation-aware Video Reading Comprehension for Temporal Language Grounding},
  author={Gao, Jialin and Sun, Xin and Xu, Mengmeng and Zhou, Xi and Ghanem, Bernard},
  journal={arXiv preprint arXiv:2110.05717},
  year={2021}
}

@InProceedings{2DTAN_2020_AAAI,
author = {Zhang, Songyang and Peng, Houwen and Fu, Jianlong and Luo, Jiebo},
title = {Learning 2D Temporal Adjacent Networks forMoment Localization with Natural Language},
booktitle = {AAAI},
year = {2020}
} 

@InProceedings{Xu_2020_CVPR,
author = {Xu, Mengmeng and Zhao, Chen and Rojas, David S. and Thabet, Ali and Ghanem, Bernard},
title = {G-TAD: Sub-Graph Localization for Temporal Action Detection},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
url={https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_G-TAD_Sub-Graph_Localization_for_Temporal_Action_Detection_CVPR_2020_paper.pdf},
month = {June},
year = {2020}
}

@INPROCEEDINGS{9009011,
author={Huang, Zilong and Wang, Xinggang and Huang, Lichao and Huang, Chang and Wei, Yunchao and Liu, Wenyu},
booktitle={2019 IEEE/CVF International Conference on Computer Vision (ICCV)}, 
title={CCNet: Criss-Cross Attention for Semantic Segmentation}, 
year={2019},
volume={},
number={},
pages={603-612},
doi={10.1109/ICCV.2019.00069}
}

```