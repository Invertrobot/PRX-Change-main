#! /bin/bash
cd ..

python test.py \
   --bash='test_levir.sh' \
   --test_root='data/LEVIR/test/' \
   --test_checkpoint_path='checkpoints/levircd_best_f1_9214.pth'\
   --testsize=1024 \
   --cross=True \
   --save_result=True \
   --save_result_path='results/levir/pred/'\
   --save_iou_map=True \
   --save_iou_map_path='results/levir/iou_map/'