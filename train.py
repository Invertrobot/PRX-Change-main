from datetime import datetime
import shutil
from torch.autograd import Variable
import time
import os, argparse
from config import opt
from utils.prx_change_model import *
from dataloader.change_dataloader import get_loader
from visualize.process_bar import process_bar
from utils.lr_decay import adjust_lr
from thop import profile, clever_format
from utils.focal_loss import *


def train(train_loader, model, optimizer, epoch):
    model.train()
    total_step = len(train_loader)
    loss_sum, avg_loss, process_step, ref_loss_sum = 0, 0, 0, 0

    if opt.mixed_precision_training:
        scaler = torch.cuda.amp.GradScaler()

    global_start = time.perf_counter()
    for i, pack in enumerate(train_loader, start=1):
        start = time.perf_counter()

        optimizer.zero_grad()
        before, after, gts = pack

        '''Data process'''
        before = Variable(before)
        after = Variable(after)
        gts = Variable(gts)

        before = before.cuda()
        after = after.cuda()
        gts = gts.cuda()

        '''Calculate loss and backward'''
        if opt.mixed_precision_training:
            with torch.cuda.amp.autocast():
                '''Get predict mask'''
                pred, results = model(before, after)

            loss = BCE(pred, gts)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            '''Get predict mask'''
            pred, results = model(before, after)
            loss = BCE(pred, gts)
        '''Training progress visualize'''
        step_loss = loss.cpu().item()
        loss_sum += step_loss
        avg_loss = loss_sum / i

        '''Process bar'''
        process_step += 100 / total_step

        '''calc run time'''
        end = time.perf_counter()
        run_time = end - start
        last_time = "%d:%d" % (run_time * (total_step - i) // 60, run_time * (total_step - i) % 60)
        past_time = "%d:%d" % ((end - global_start) // 60, (end - global_start) % 60)
        time_str = "[{}/{}]".format(past_time, last_time)

        process_bar(process_step, epoch, avg_loss, step_loss, optimizer.param_groups[0]['lr'], time_str)
        '''Training progress visualize'''

    '''Training progress visualize'''
    print(' Epoch ' + str(epoch) + ' train loss: \033[34m%.4f\033[0m' % avg_loss)

    """log loss"""
    with open(opt.save_path + 'log.txt', 'a') as file:
        file.write("Train Epoch：{} loss: {}\n".format(epoch, avg_loss))

    '''Save checkpoints'''
    if epoch % opt.save_feq == 0:
        torch.save(model.state_dict(), opt.save_path + 'Epoch_%d_loss_%.4f_CD.pth' % (epoch, avg_loss))


def val(val_loader, model, epoch):
    model.eval()
    total_step = len(val_loader)
    loss_sum, process_step = 0, 0
    TP, TN, FP, FN = 0, 0, 0, 0
    with torch.no_grad():
        for i, pack in enumerate(val_loader, start=1):
            before, after, gts = pack
            '''Data process'''
            before = Variable(before)
            after = Variable(after)
            gts = Variable(gts)

            before = before.cuda()
            after = after.cuda()
            gts = gts.cuda()

            '''Get predict mask'''
            pred, results = model(before, after)

            '''Calculate loss and backward'''
            loss = BCE(pred, gts)

            '''evaluation'''
            dets = pred.sigmoid()
            dets = torch.where(dets <= 0.5, 0., 1.)
            dets = dets + 2  # (2 , 3)
            gts = gts + 1  # (1 , 2)
            iou_map = gts * dets

            TP += len((iou_map == 6).nonzero())  # change predict to change -> 2 * 3 -> TP
            FN += len((iou_map == 4).nonzero())  # change predict to unchange -> 2 * 2 -> FN
            FP += len((iou_map == 3).nonzero())  # unchange predict to change -> 1 * 3 -> FP
            TN += len((iou_map == 2).nonzero())  # unchange predict to unchange -> 1 * 2 -> TN

            '''Val progress visualize'''
            loss_sum += loss.cpu().item()
            avg_loss = loss_sum / i
            '''Process bar'''
            process_step += 100 / total_step
            process_bar(process_step, epoch, avg_loss, training=False)
            '''Training progress visualize'''

    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    iou = TP / (TP + FP + FN + 1e-10)
    F1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    OA = (TP + TN) / (TP + TN + FP + FN)

    '''Val progress visualize'''
    print(' Epoch ' + str(epoch) + '   val loss: %.4f pre: %.4f recall: %.4f iou:%.4f,F1:%.4f, OA:%.4f Best:%.4f' % (
        avg_loss, precision, recall, iou, F1, OA, opt.best_f1))

    """log loss and metrics"""
    with open(opt.save_path + 'log.txt', 'a') as file:
        file.write("Val Epoch：{} loss: {} F1: {}\n".format(epoch, avg_loss, F1))

    '''Save checkpoints'''
    if F1 > opt.best_f1:
        opt.best_f1 = F1
        torch.save(model.state_dict(), opt.save_path + '{}.pth'.format(opt.best_name))
        with open(opt.save_path + 'metrics.txt', 'w') as file:
            file.write("F1: {}\n OA: {}\n IoU:{}\n Pre:{}\n Recall:{}\n".format(F1, OA, iou, precision, recall))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device

    '''Build models'''
    ''' backbone in {18, 34, 50, 101}  set backbone volume
        x_cross in {True, False} set with cross attention or not
        operation_type in {cat, sub} set difference operation '''
    model = Resnet_CD(backbone=opt.backbone)
    print(model)
    model.cuda()
    params = model.parameters()
    optimizer = torch.optim.AdamW(params, opt.lr, eps=1e-3, weight_decay=opt.weight_decay)

    if opt.color_setting is not None:
        opt.color_setting = ''.join(opt.color_setting)
        opt.color_setting = opt.color_setting.split(' ')

    '''Resume model'''
    model_path = opt.checkpoint_path
    if opt.resume:
        print('Loading weights ' + model_path.split('/')[-1] + ' into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print('Finished!')
    else:
        pretrained_path = 'checkpoints/base/resnet{}.pth'.format(opt.backbone)
        if os.path.exists(pretrained_path):
            backbone_dict = torch.load(pretrained_path)
            model.backbone.load_state_dict(backbone_dict, strict=False)
            print('Load model pretrained on imagenet, Finished!')

    '''Load training data'''
    root = opt.root
    train_before, train_after, train_gt = root + 'train/A/', root + 'train/B/', root + 'train/label/'
    val_before, val_after, val_gt = root + 'val/A/', root + 'val/B/', root + 'val/label/'

    train_loader = get_loader(train_before, train_after, train_gt, batchsize=opt.train_batchsize,
                              trainsize=opt.trainsize, color_setting=opt.color_setting, shuffle=True, num_workers=8)
    val_loader = get_loader(val_before, val_after, val_gt, batchsize=opt.val_batchsize, trainsize=opt.trainsize,
                            color_setting=opt.color_setting, shuffle=False, advance=False)

    '''Loss function'''
    BCE = nn.BCEWithLogitsLoss()
    # BCE = BCEFocalLoss()

    opt.save_path = opt.save_path + '{}_{}/'.format(
        datetime.now(),
        opt.best_name,
    )

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
        shutil.copy('train_script/' + opt.bash, opt.save_path + opt.bash)

    '''Start Training'''
    print('Learning Rate: {} Total Epoch: {}'.format(opt.lr, opt.epoch))
    print('Train with Hard Feature Refinement: {}'.format(opt.refine))
    print('Train with cross: {}'.format(opt.cross))
    print('Train with k: {}, beta: {}, l1:{}, l2:{}'.format(opt.k, opt.beta, opt.l1, opt.l2))
    if opt.color_setting is not None:
        print('Train with color setting: {}'.format(opt.color_setting))
    print('Lets go!')

    for epoch in range(1, opt.epoch + 1):
        train(train_loader, model, optimizer, epoch)
        if epoch % opt.val_epoch == 0:
            val(val_loader, model, epoch)
        if epoch % opt.decay_epoch == 0:
            adjust_lr(optimizer, opt.lr, epoch, opt.epoch, power=opt.power)
