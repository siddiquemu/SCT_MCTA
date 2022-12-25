# Tracking Passengers and Baggage Items using Multiple Overhead Cameras at Security Checkpoints.
(Published by IEEE Transactions on Systems, Man, and Cybernetics: Systems) [published paper](https://ieeexplore.ieee.org/document/9984680). Preprint version [arxiv paper](https://arxiv.org/abs/2007.07924)

# codeabse progress for Self-Supervised Learning (SSL) detector
- [ ] Prepare data for SSL training and testing
- [ ] Train initial Supervised Learning (SL) ResNet-50 ([PANet](https://github.com/ShuLiu1993/PANet)) model using train set
- [ ] Generate pseudo-labels for training ResNet-50 ([PANet](https://github.com/ShuLiu1993/PANet)) model
- [ ] Train iteratively using pseudo labels
- [ ] Evaluate the SSL detector model

# codeabse progress for Single-Camera Tracking (SCT)
- [ ] Finetune [Tracktor](https://github.com/phil-bergmann/tracking_wo_bnw) detector (ResNet-50 Faster-RCNN) using SSL or use SSL PANet for detection
- [ ] Evaluate the SCT model

# codeabse progress for Multi-Camera Tracklet Association (MCTA)
- [ ] Prepare the precomputed SCT tracklets (offline)
- [ ] Evaluate the MCTA model

### Requirements for PANet detector: ###
* Python 3.6
* Pytorch 0.4.0
* CUDA 9.2
* Pycocotools 2.0

### Requirements for Tracktor detector: ###
* Detectron2
* Python 3.8
* Pytorch 1.9
* CUDA 10.2
* Pycocotools 2.0

### [ ] Installation ###

1. clone this repository and go to root folder
```python
https://github.com/siddiquemu/SCT_MCTA.git
cd SCT_MCTA
```
2. create environment
```python
pip install -r panet_requirements.yml
```
3. This codebase is heavily based on instance segmentation network [PANet](https://github.com/facebookresearch/detectron2) and an SCT [Tracktor](https://github.com/phil-bergmann/tracking_wo_bnw). Install both in separate environment
```SCT_MCTA
```

### [ ] Data Preprocessing ###
1. Download the data from 
```
```
2. run the following script from root to generate the train/test split
```
```

### [ ] Test ###
1. To test the models, download models from

2. run the following script to evaluate the SSL detector models

```
```

### [ ] Train ###
[ ] To train the SL model using train set:
1. run the following script

```
```

[ ] To train the SSL model on the unlabeled data using pretrained model:

1.  go to root directory and run

```
```

### Citing SCT_MCTA ###
If you find this work helpful in your research, please cite using the following bibtex
```
@ARTICLE{siddiqueMulticamTSMC2022,
  author={Siddique, Abubakar and Medeiros, Henry},
  journal={IEEE Transactions on Systems, Man, and Cybernetics: Systems}, 
  title={Tracking Passengers and Baggage Items Using Multiple Overhead Cameras at Security Checkpoints}, 
  year={2022},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TSMC.2022.3225252}}


```
