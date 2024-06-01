#! /bin/bash
cd ..

python train.py \
   --bash='train_s2looking.sh' \
   --root='data/S2Looking/' \
   --best_name='s2looking' \
   --epoch=1000 \
   --lr=6e-4 \
   --weight_decay=2e-3 \
   --train_batchsize=8 \
   --color_setting='0.2 0.2 0.1 0.1' \
   --trainsize=1024 \
   --cropsize=512 \
   --save_path='' \
   --val_epoch=10