import argparse

'''Train parameter'''
parser = argparse.ArgumentParser()

""" base setting """
parser.add_argument('--bash', default='train.sh', help='epoch number')
parser.add_argument('--device', default='0', help='gpu device number')
parser.add_argument('--mixed_precision_training', default=True, help='enable mixed precision training')

""" network setting """
parser.add_argument('--backbone', default='18', help='set backbone in {18, 34, 50, 101}')
parser.add_argument('--cross', default=False, help='cross attention')
parser.add_argument('--refine', default=False, help='Hard feature refinement')

""" hyper parameter """
parser.add_argument('--k', default=2, help='sampling parameter k')
parser.add_argument('--beta', default=0.75, help='sampling parameter beta')
parser.add_argument('--l1', default=5.0, help='loss weight lambda 1')
parser.add_argument('--l2', default=1.0, help='loss weight lambda 2')
parser.add_argument('--power', default=1.0, help='lr decay parameter')

""" train parameter """
parser.add_argument('--root', default='/root/change/data/LEVIR-CD/', help='root dir')

parser.add_argument('--trainsize', type=int, default=1024, help='training data size')
parser.add_argument('--cropsize', type=int, default=512, help='training random crop size')
parser.add_argument('--color_setting', default="0.2 0.2 0.1 0.1", help='color jitter setting')

parser.add_argument('--epoch', type=int, default=800, help='set epoch number')
parser.add_argument('--lr', type=float, default=6e-4, help='set learning rate')
parser.add_argument('--weight_decay', type=float, default=2e-3, help='set weight decay')

parser.add_argument('--train_batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--val_batchsize', type=int, default=4, help='val batch size')
parser.add_argument('--val_epoch', type=int, default=10, help='every n epochs do evaluation')
parser.add_argument('--decay_epoch', type=int, default=1, help='every n epochs decay learning rate')


parser.add_argument('--best_f1', type=float, default=0.0, help='record best f1')
parser.add_argument('--best_name', default='', help='best ckpt name')
parser.add_argument('--save_path', default='', help='checkpoint save dir')
parser.add_argument('--save_feq', default=100, help='checkpoint save dir')
parser.add_argument('--checkpoint_path', default='', help='checkpoint path for resume')
parser.add_argument('--resume', default=False,
                    help='resume checkpoint (set to False will load model pretrained on imagenet)')


""" test parameter """
parser.add_argument('--testsize', type=int, default=1024, help='testing size')
parser.add_argument('--save_result', default=False, help='test root dir')
parser.add_argument('--save_iou_map', default=False, help='test root dir')
parser.add_argument('--save_result_path', default='result/pred/levir/', help='pred result save path')
parser.add_argument('--save_iou_map_path', default='result/iou_map/levir/', help='iou map save path')
parser.add_argument('--test_checkpoint_path',
                    default='checkpoints/levircd_best_f1_9214.pth',
                    help='checkpoint path for test')
parser.add_argument('--test_root', default='/root/change/data/LEVIR-CD/test/', help='test root dir')

opt = parser.parse_args()
