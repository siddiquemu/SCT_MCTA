# Tracking Passengers and Baggage Items using Multiple Overhead Cameras at Security Checkpoints.
(Published by IEEE Transactions on Systems, Man, and Cybernetics: Systems) [published paper](https://ieeexplore.ieee.org/document/9984680). Preprint version [arxiv paper](https://arxiv.org/abs/2007.07924)
- A docker will be released soon for testing SSL, SCT, and MCTA
# codeabse progress for Self-Supervised Learning (SSL) detector
- [ ] Prepare data for SSL training and testing
- [ ] Train initial Supervised Learning (SL) ResNet-50 ([PANet](https://github.com/ShuLiu1993/PANet)) model using train set
- [ ] Generate pseudo-labels for training ResNet-50 ([PANet](https://github.com/ShuLiu1993/PANet)) model
- [ ] Train iteratively using pseudo labels
- [ ] Test the SSL detector model

# codeabse progress for Single-Camera Tracking (SCT)
- [x] Finetune [Tracktor](https://github.com/phil-bergmann/tracking_wo_bnw) detector (ResNet-50 Faster-RCNN) using SSL FRCNN or SSL PANet predictions
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
* CUDA 11.2
* Python 3.8
* Pytorch 1.9
* Pycocotools 2.0

### [ ] PANet Installation for SSL-PANet Training ###
1. clone this repository and go to root folder
```python
https://github.com/siddiquemu/SCT_MCTA.git
cd SCT_MCTA/SSL_PANet
```
2. create environment by following instance segmentation network [PANet](https://github.com/ShuLiu1993/PANet) (need torch 0.4.0)
```python
pip install -r panet_requirements.yml
```
3. Dataset preparation for running PANet-based SSL
   - For unlabeled data following command from SSL_PANet root folder
    ```
    python tools/clasp_unlabeled_json.py 
    ```
   - For labeled data run the following
    ```
    python tools/clasp_gt_annotations.py  
    ```
   - For joint labeled and unlabeled data preparation
    ```
    python tools/CLASP_SSL/b2mask_clasp1_annotations.py  
    ```
   - For labeled, unlabeled, and augmentded data preparation run the above two steps and then run the following
    ```
    for ITER in 4; do  bash tools/train_semi_iters_clasp1.sh SSL_pseudo_labels ${ITER} 10 2; done  
    ```
4. Start iterative training (including pseudo-label generation steps) using the following command
   ```
   for ITER in 4; do  bash tools/train_semi_iters_clasp1.sh SSL_aug_train ${ITER} 10 2; done  
   ```
### [x] Tracktor Installation for SSL-FRCNN Training ###

1. clone this repository and go to root folder
```python
https://github.com/siddiquemu/SCT_MCTA.git
cd SCT_MCTA/tracking_wo_bnw
```
2. Install tracktor similar to [Tracktor](https://github.com/phil-bergmann/tracking_wo_bnw) but we do not need to clone latest repo.
3. Run the command
```
cd SCT_MCTA/SSL_FRCNN
python clasp_det.py
```
3. Edit the configs for data and pretrained model directories
```
./cfg/SSL.yaml
clasp_det.py
```

### [x] Data Preprocessing ###
1. Collect data upon request at [alert-coe@northeastern.edu](https://alert.northeastern.edu/)

Folder structure for CLASP1 datasets:

```
CLASP1/train_gt
├── A_11
│   ├── gt
│   ├── img1
│   ├── test_frames
│   └── train_frames
├── A_9
│   ├── gt
│   ├── img1
│   ├── test_frames
│   └── train_frames
├── B_11
│   ├── gt
│   ├── img1
│   ├── test_frames
│   └── train_frames
├── B_9
│   ├── gt
│   ├── img1
│   ├── test_frames
│   └── train_frames
├── C_11
│   ├── gt
│   ├── img1
│   ├── test_frames
│   └── train_frames
├── C_9
│   ├── gt
│   ├── img1
│   ├── test_frames
│   └── train_frames
├── D_11
│   ├── gt
│   ├── img1
│   ├── test_frames
│   └── train_frames
├── D_9
│   ├── gt
│   ├── img1
│   ├── test_frames
│   └── train_frames
├── E_11
│   ├── gt
│   ├── img1
│   ├── test_frames
│   └── train_frames
└── E_9
    ├── gt
    ├── img1
    ├── test_frames
    └── train_frames
```

Folder structure for CLASP2 datasets:

```
CLASP2/train_gt/
├── G_11
│   ├── gt
│   ├── gt_sct
│   ├── img1
│   ├── test_frames
│   └── train_frames
├── G_9
│   ├── gt
│   ├── gt_sct
│   ├── img1
│   ├── test_frames
│   └── train_frames
├── H_11
│   ├── gt
│   ├── gt_sct
│   ├── img1
│   ├── test_frames
│   └── train_frames
├── H_9
│   ├── gt
│   ├── gt_sct
│   ├── img1
│   ├── test_frames
│   └── train_frames
├── I_11
│   ├── gt
│   ├── gt_sct
│   ├── img1
│   ├── test_frames
│   └── train_frames
└── I_9
    ├── gt
    ├── gt_sct
    ├── img1
    ├── test_frames
    └── train_frames
```


2. run the following script from root to generate the train/test split

```
```

### [ ] Test SSL_PANet ###

```
cd SCT_MCTA/SSL_PANet
```

1. To test the models, download models from

2. run the following script to evaluate the SSL detector models

```
bash run_ssl_panet.sh
```

### [ ] Train SSL_PANet ###
[x] To train the SL model using train set:

```
cd SCT_MCTA/SSL_PANet
```
1. run the following script to generate pseudo-labels for the unlabeled frames

```
for ITER in 1; do   bash train_semi_iters_clasp1.sh SSL_pseudo_labels ${ITER} 0 2; done
```
2. run the following to start training using the generated psudo-labels

```
for ITER in 1; do   bash train_semi_iters_clasp1.sh SSL_aug_train ${ITER} 0 2; done
```
### [ ] Test SCT ###
This codebase is heavily based on SCT [Tracktor](https://github.com/phil-bergmann/tracking_wo_bnw).

### [ ] Test MCTA ###
Will be updated soon...

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
