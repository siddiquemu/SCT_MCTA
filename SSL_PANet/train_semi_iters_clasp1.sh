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
    python3 tools/train_net_step.py --ssl_iter ${ITER} --dataset clasp1_2021_aug_score \
          --load_ckpt /media/abubakar/PhD_Backup/models/panet/panet_det_step89999.pth.th \
          --cfg ./configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml
elif [[ ${TYPE} == 'SSL_mixed_aug' ]]; then
    # semi
    # generate pseudo labels using 3*N processes in N GPUS
#    python3 tools/pseudo_labels_ngpu_bash.py --ssl_iter ${ITER} --database clasp1 --label_percent ${PERCENT} \
#          --cfg configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml
    # train using computed pseudo labels
#    python3 tools/train_net_step.py --ssl_iter ${ITER} --dataset clasp1_2021_aug_score --label_percent ${PERCENT}\
#          --working_dir /media/abubakar/PhD_Backup/models/clasp1/modified_loss \
#          --cfg configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml
    python3 tools/train_net_step.py --ssl_iter ${ITER} --dataset clasp1_2021_mixed_aug_score --label_percent ${PERCENT}\
          --working_dir /media/abubakar/PhD_Backup/models/clasp1/modified_loss_mixed_aug \
          --cfg configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml --learning_type ${TYPE}

elif [[ ${TYPE} == 'SSL_aug_train' ]]; then

    # generate pseudo labels using 3*N processes in N GPUS
   python3 tools/pseudo_labels_ngpu_bash.py --ssl_iter ${ITER} --database clasp1 --label_percent ${PERCENT} \
         --cfg ./configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml --models_dir /media/abubakar/PhD_Backup/models/clasp1 \
         --data_dir /media/abubakar/PhD_Backup/data/CLASP1 --model_type modified_loss

    # train using computed pseudo labels
   python3 tools/train_net_step.py --ssl_iter ${ITER} --dataset clasp1_2021_aug_score --label_percent ${PERCENT}\
         --working_dir /media/abubakar/PhD_Backup/models/clasp1/modified_loss \
         --cfg ./configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml

elif [[ ${TYPE} == 'SSL_pseudo_labels' ]]; then
    # semi
    # generate pseudo labels using 3*N processes in N GPUS
   python3 tools/pseudo_labels_ngpu_bash.py --ssl_iter ${ITER} --database clasp1 --label_percent ${PERCENT} \
         --cfg ./configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml --models_dir /media/abubakar/PhD_Backup/models/clasp1 \
         --data_dir /media/abubakar/PhD_Backup/data/CLASP1 --model_type modified_loss

fi