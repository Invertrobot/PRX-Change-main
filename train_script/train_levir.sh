#! /bin/bash
cd ..

python train.py \
   --bash='train_levir.sh' \
   --root='/root/change/data/LEVIR-CD/' \
   --best_name='levir' \
   --epoch=1000 \
   --lr=6e-4 \
   --weight_decay=2e-3 \
   --train_batchsize=8 \
   --color_setting='0.2 0.2 0.1 0.1' \
   --trainsize=1024 \
   --cropsize=512 \
   --save_path='/root/change/checkpoints/run/levir/' \
   --val_epoch=1