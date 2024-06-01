#! /bin/bash
cd ..

python train.py \
   --bash='train_dsifn.sh' \
   --root='data/DSIFN/' \
   --best_name='dsifn' \
   --epoch=1000 \
   --lr=8e-4 \
   --weight_decay=2e-3 \
   --train_batchsize=16 \
   --color_setting='0. 0. 0. 0.' \
   --trainsize=256 \
   --cropsize=256 \
   --save_path='' \
   --val_epoch=1 \
