#! /bin/bash
cd ..

python test.py \
   --bash='test_dsifn.sh' \
   --test_root='data/S2Looking/test/' \
   --test_checkpoint_path='checkpoints/s2looking_best_f1_6650.pth'\
   --testsize=1024 \
   --cross=True \
   --save_result=True \
   --save_result_path='results/s2looking/pred/'\
   --save_iou_map=True \
   --save_iou_map_path='results/s2looking/iou_map/'