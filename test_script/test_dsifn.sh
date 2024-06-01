#! /bin/bash
cd ..

python test.py \
   --bash='test_dsifn.sh' \
   --test_root='data/DSIFN/test/' \
   --test_checkpoint_path='checkpoints/dsifncd_best_f1_9644.pth'\
   --testsize=256 \
   --cross=True \
   --save_result=True \
   --save_result_path='results/dsifn/pred/'\
   --save_iou_map=True \
   --save_iou_map_path='results/dsifn/iou_map/'