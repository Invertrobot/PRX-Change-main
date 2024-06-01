#! /bin/bash
cd ..

python train_refine.py \
  --bash='train_dsifn_refine.sh' \
  --root='data/DSIFN/' \
  --best_name='dsifn_refine' \
  --epoch=800 \
  --lr=4e-4 \
  --weight_decay=2e-3 \
  --train_batchsize=16 \
  --color_setting='0. 0. 0. 0.' \
  --trainsize=256 \
  --cropsize=256 \
  --resume=True \
  --checkpoint_path='' \
  --save_path='' \
  --val_epoch=1 \
  --refine=True \
#  --cross=True
