#! /bin/bash
cd ..

python test.py \
   --bash='quick_start.sh' \
   --test_root='test_samples/' \
   --test_checkpoint_path='checkpoints/levircd_best_f1_9214.pth'\
   --testsize=1024 \
   --cross=True \
   --save_result=True \
   --save_result_path='results/qs/pred/'\
   --save_iou_map=True \
   --save_iou_map_path='results/qs/iou_map/'