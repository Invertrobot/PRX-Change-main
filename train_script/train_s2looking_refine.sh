#! /bin/bash
cd ..

python train_refine.py \
   --bash='train_s2looking_refine.sh' \
   --root='data/S2Looking/' \
   --best_name='s2looking_refine' \
   --epoch=500 \
   --lr=4e-4 \
   --weight_decay=2e-3 \
   --train_batchsize=8 \
   --color_setting='0.2 0.2 0.1 0.1' \
   --trainsize=1024 \
   --cropsize=512 \
   --resume=True \
   --checkpoint_path='' \
   --save_path='' \
   --val_epoch=1 \
   --refine=True \
#   --cross=True