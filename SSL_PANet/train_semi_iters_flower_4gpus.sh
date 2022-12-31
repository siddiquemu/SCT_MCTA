#!/usr/bin/env bash
set -x

TYPE=$1 # SL or SSL or semi
ITER=$2
PERCENT=$3
GPUS=$2
PORT=${PORT:-29500}

#for ITER in 4; do   bash tools/train_semi_iters_clasp2.sh semi ${ITER} 10 2; done

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

if [[ ${TYPE} == 'SL' ]]; then
    python3 tools/train_net_step.py --ssl_iter ${ITER} --dataset flower_2021_labeled \
          --load_ckpt /media/abubakarsiddique/PANet_Models/flower/SL/ckpt/model_step9999.pth \
          --working_dir /media/abubakarsiddique/PANet_Models/flower/SL \
          --cfg configs/panet/e2e_panet_R-50-FPN_2x_mask_4gpus.yaml
else
    # semi
    # generate pseudo labels using 3*N processes in N GPUS
    python3 tools/pseudo_labels_ngpu_bash_flower.py --ssl_iter ${ITER} --database flower --label_percent ${PERCENT} \
              --cfg configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml
    # train using computed pseudo labels
    python3 tools/train_net_step.py --ssl_iter ${ITER} --dataset flower --label_percent ${PERCENT}\
          --working_dir /media/abubakarsiddique/PANet_Models/flower/modified_loss_semi \
          --cfg configs/panet/e2e_panet_R-50-FPN_2x_mask_4gpus.yaml

fi
