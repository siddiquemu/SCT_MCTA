#!/usr/bin/env bash
set -x

TYPE=$1 # SL or SSL or semi
ITER=$2
PERCENT=$3
GPUS=$2
PORT=${PORT:-29500}

#for ITER in 4; do   bash tools/train_semi_iters_clasp2.sh semi ${ITER} 10 2; done

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

if [[ ${TYPE} == 'baseline' ]]; then
    python3 tools/train_net_step.py --ssl_iter ${ITER} --dataset clasp2_2021_aug_score \
          --load_ckpt /home/siddique/PANet/models/R50-FRCNN/panet_det_step89999.pth.th \
          --cfg /home/siddique/PANet/configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml
else
    # semi
    # generate pseudo labels using 3*N processes in N GPUS
#    python3 tools/pseudo_labels_ngpu_bash.py --ssl_iter ${ITER} --database clasp2 --label_percent ${PERCENT} \
#          --cfg configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml
    # train using computed pseudo labels
     python3 tools/train_net_step.py --ssl_iter ${ITER} --dataset clasp2_2021_mixed_aug_score --label_percent ${PERCENT}\
           --working_dir /media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62/PANet_Models/clasp2/modified_loss_mixed_aug \
           --cfg configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml --learning_type ${TYPE}

fi