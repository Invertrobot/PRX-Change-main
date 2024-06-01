import os
import numpy as np
import argparse
from thop import profile, clever_format
from utils.prx_change_model import *
from dataloader.change_dataloader import test_dataset
from tqdm import tqdm
import cv2
from config import opt
from visualize.tSNE import *
from PIL import Image


def save_mask(res, save_path, name):
    res = torch.where(res <= 0.5, 0., 255.)
    img = np.uint8(res.squeeze().data.cpu().numpy())
    img = Image.fromarray(img)
    img.save(save_path + name)


def save_iou_map(iou_map, size, path, index):
    img = torch.zeros([size, size, 3])
    img[(iou_map == 4).nonzero()[:, 0], (iou_map == 4).nonzero()[:, 1], 0] = 200.
    img[(iou_map == 6).nonzero()[:, 0], (iou_map == 6).nonzero()[:, 1], 1] = 200.
    img[(iou_map == 3).nonzero()[:, 0], (iou_map == 3).nonzero()[:, 1], 2] = 200.
    img = np.uint8(img.data.cpu().numpy())
    img = Image.fromarray(img)
    img.save(path + index)


def test_and_eval():
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device

    '''Bulid model and load checkpoint'''
    model = Resnet_CD(backbone=opt.backbone)
    print(model)
    print('Test with feat1-cross state: {}'.format(opt.cross))
    model.load_state_dict(torch.load(opt.test_checkpoint_path), strict=False)
    model.cuda()
    model.eval()

    x = torch.randn(1, 3, 512, 512).cuda()
    macs, params = profile(model, inputs=(x, x))
    macs, params = clever_format([macs, params], "%.3f")
    print('The number of MACs is %s' % macs)
    print('The number of params is %s' % params)

    if not os.path.exists(opt.save_result_path) and opt.save_result:
        os.makedirs(opt.save_result_path)
    if not os.path.exists(opt.save_iou_map_path) and opt.save_iou_map:
        os.makedirs(opt.save_iou_map_path)

    '''Load test data'''
    image_before, image_after, gt_root = opt.test_root + 'A/', opt.test_root + 'B/', opt.test_root + 'label/'
    test_loader = test_dataset(image_before, image_after, gt_root, opt.testsize)

    '''Test'''
    with torch.no_grad():
        TP, TN, FP, FN = 0, 0, 0, 0
        for i in tqdm(range(test_loader.size)):
            before, after, gt, name = test_loader.load_data()
            before = before.cuda()
            after = after.cuda()
            gt = gt.cuda()

            pred, results = model(before, after)
            pred = pred.sigmoid()
            mask = torch.where(pred <= 0.5, 0., 1.)
            mask = mask + 2  # (2, 3)
            gt_ = gt + 1  # (1, 2)
            iou_map = gt_ * mask

            TP += len((iou_map == 6).nonzero())  # change predict to change -> 2 * 3 -> TP
            FN += len((iou_map == 4).nonzero())  # change predict to unchange -> 2 * 2 -> FN
            FP += len((iou_map == 3).nonzero())  # unchange predict to change -> 1 * 3 -> FP
            TN += len((iou_map == 2).nonzero())  # unchange predict to unchange -> 1 * 2 -> TN

            if opt.save_result:
                save_mask(pred, opt.save_result_path, name)
            if opt.save_iou_map:
                save_iou_map(iou_map.squeeze(), iou_map.size()[-1], opt.save_iou_map_path, name)

    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    iou = TP / (TP + FP + FN + 1e-10)
    F1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    OA = (TP + TN) / (TP + TN + FP + FN + 1e-10)

    print('precision: %.2f %%' % (precision * 100))
    print('   recall: %.2f %%' % (recall * 100))
    print('      iou: %.2f %%' % (iou * 100))
    print('       OA: %.2f %%' % (OA * 100))
    print('       F1: %.2f %%' % (F1 * 100))


if __name__ == '__main__':
    test_and_eval()
