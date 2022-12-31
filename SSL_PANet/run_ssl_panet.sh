
#!/usr/bin/env bash
set -x

TYPE=$1 # SL or SSL or semi
ITER=$2
PERCENT=$3
GPUS=$2
PORT=${PORT:-29500}

#for ITER in 4; do   bash tools/train_semi_iters_clasp2_4gpus.sh semi ${ITER} 10 2; done

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

if [[ ${TYPE} == 'baseline' ]]; then
    python3 tools/train_net_step.py --ssl_iter ${ITER} --dataset clasp2_2021_aug_score \
          --load_ckpt /home/siddique/PANet/models/R50-FRCNN/panet_det_step89999.pth.th \
          --cfg /home/siddique/PANet/configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml
else
    python3 tools/infer_panet_clasp.py --data_dir /media/6TB_local/tracking_wo_bnw/data/CLASP/train_gt_all/PB_gt \
    --dataset clasp2 --ssl_iter 8 --model_dir /media/6TB_local/PANet_Models/clasp2/modified_loss

fi