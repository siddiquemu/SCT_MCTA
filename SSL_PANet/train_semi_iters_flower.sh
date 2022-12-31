#!/usr/bin/env bash
set -x

TYPE=$1 # SL or SSL or semi
ITER=$2
PERCENT=$3
GPUS=$2
PORT=${PORT:-29500}

#for ITER in 4; do   bash tools/train_semi_iters_clasp2.sh semi ${ITER} 10 2; done
#python3 tools/train_net_step.py --ssl_iter ${ITER} --dataset flower_2021_labeled --load_ckpt /media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/PANet_Models/flower/modified_loss_semi/100_percent/iter2/ckpt/model_step19999.pth --working_dir /media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/PANet_Models/flower/SL --cfg /home/siddique/PANet/configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

if [[ ${TYPE} == 'SL' ]]; then

    python3 tools/train_net_step.py --ssl_iter ${ITER} --dataset flower_2021_labeled \
          --load_ckpt /media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/PANet_Models/flower/modified_loss_semi/100_percent/iter2/ckpt/model_step19999.pth \
          --working_dir /media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/PANet_Models/flower/SL \
          --cfg /home/siddique/PANet/configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml
else
    # semi
    # generate pseudo labels using 3*N processes in N GPUS
    python3 tools/pseudo_labels_ngpu_bash_flower.py --ssl_iter ${ITER} --database flower --label_percent ${PERCENT} \
              --cfg configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml
    # train using computed pseudo labels
    python3 tools/train_net_step.py --ssl_iter ${ITER} --dataset flower --label_percent ${PERCENT}\
          --working_dir /media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62/PANet_Models/flower/modified_loss_semi \
          --cfg configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml

fi