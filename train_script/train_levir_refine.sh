#! /bin/bash
cd ..

python train_refine.py \
   --bash='train_levir_refine.sh' \
   --root='/root/change/data/LEVIR-CD/' \
   --best_name='levir_refine' \
   --epoch=800 \
   --lr=4e-4 \
   --weight_decay=2e-3 \
   --train_batchsize=8 \
   --color_setting='0.2 0.2 0.1 0.1' \
   --trainsize=1024 \
   --cropsize=512 \
   --resume=True \
   --checkpoint_path='/root/change/checkpoints/saves/TFN_18_4x_f1_channelshuffle_9160.pth' \
   --save_path='/root/change/checkpoints/run/levir/' \
   --val_epoch=1 \
   --refine=True \
   --cross=True
