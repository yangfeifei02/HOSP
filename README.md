# HOSP
PyTorch code for paper: High-order structure preserving graph neural network for few-shot learning

Arxiv: https://arxiv.org/abs/2005.14415

# Requirements
Python 3.6

torch 1.2.0

numpy 1.18.4

# dataset

-[miniImagenet](https://drive.google.com/drive/folders/15WuREBvhEbSWo4fTr1r-vMY0C_6QWv4w)

-[FC-100](https://drive.google.com/drive/folders/1nz_ADBblmrg-qs-8zFU3v6C5WSwQnQm6)

-[tiered-imagenet](https://drive.google.com/file/d/1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07/view)

-[CIFAR-FS](https://drive.google.com/file/d/1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI/view)

The data folder should be organized as,
```
tt.arg.dataset_root = ./data/private/dataset/
```
# Train
```
python3 train.py

Modify the following parameters for different models

-tt.arg.dataset: mini / tiered / CIFARFS

-tt.arg.num_unlabeled : for semi-supervised learning

-tt.arg.meta_batch_size: batch size

-tt.arg.num_layers: 4 

-tt.arg.num_ways: N-way   # 5-way

-tt.arg.num_shots: K-shot   # 1-shot

-tt.arg.transductive: False
```

# eval
```
python3 eval.py

Modify tt.arg.test_model for different models

tt.arg.test_model = 'D-mini_N-5_K-5_U-0_L-4_B-40_T-False'
```

# Citation

If you find this code useful you can cite us using the following bibTex:
```
@article{lin2020high,
title={High-order structure preserving graph neural network for few-shot learning},
author={Guangfeng Lin, Ying Yang, Yindi Fan, Xiaobing Kang, Kaiyang Liao, Fan Zhao},
journal={arXiv preprint arXiv:2005.14415},
year={2020}
}
```
