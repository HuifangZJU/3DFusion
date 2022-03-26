from __future__ import print_function

import os
import json
import argparse
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import math
from tensorboardX import SummaryWriter
import logging
import datetime
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import mayavi.mlab as mlab

from lib.dataset.coop_dataset import CooperativeDataset, DataLoader
from lib.models.voxelnet import Voxelnet
from lib.functions import log_helper
from lib.functions import bbox_helper
from lib.functions import anchor_projector
from lib.functions import box_3d_encoder
from lib.functions import load_helper
from lib.evaluator import metrics
from lib.mayavi.viz_util import draw_scene


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config/tjunc.json', required=False,
                    help='hyperparameter of voxelnet in json format')
parser.add_argument('--resume', default='/mnt/tank/shaocheng/3Dfusion/saved_models_kl_limits/checkpoint_e30.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
args = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
if cuda == True:
    torch.cuda.set_device(2)

def load_config(config_path):
    assert(os.path.exists(config_path))
    cfg = json.load(open(config_path, 'r'))
    for key in cfg.keys():
        if key != 'shared':
            cfg[key].update(cfg['shared'])
    return cfg

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_data_loader(cfg):
    logger = logging.getLogger('global')
    Dataset = CooperativeDataset 
    Dataloader = DataLoader

    scales = cfg['shared']['scales']
    max_size = cfg['shared']['max_size']
    ext = torch.FloatTensor(cfg['shared']['area_extents']).view(3,2)
    ref = cfg['shared']['reference_loc']
    voxsize = cfg['shared']['voxel_size']
    maxpts = cfg['shared']['number_T']
    train_path = cfg['shared']['train_data']
    batch_size = cfg['train']['batch_size']
    workers = cfg['train']['workers']

    train_dataset = Dataset(train_path, ref, ext, voxsize, maxpts, augment=True)
    train_loader = Dataloader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=False)
    logger.info('build dataloader done')
    return train_dataset, train_loader

def main():
    cfg = load_config(args.config)
    args.save_dir = cfg['train']['save_dir']
    args.seed = cfg['train']['seed']
    args.lr = cfg['train']['lr']
    args.momentum = cfg['train']['momentum']
    args.weight_decay = cfg['train']['weight_decay']
    args.epochs = cfg['train']['epochs']
    args.step_epochs = [cfg['train']['step_epochs']]
    args.start_epoch = 0
    
    # save intermediate information: e.g. loss
    log_helper.init_log('global', args.save_dir, logging.INFO)
    logger = logging.getLogger('global')
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    logger.info('Save loss curve to {}'.format(args.save_dir+'/tensorboard'))

    #load data
    train_dataset, train_loader = build_data_loader(cfg)

    #initialize model
    model = Voxelnet(cfg=cfg)
    #logger.info(model)
    # initialize optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), '{} is not a valid file'.format(args.resume)
        model, optimizer, args.start_epoch = load_helper.restore_from(model, optimizer, args.resume)

    model.cuda()

    runname = datetime.datetime.now().strftime('%b%d_%H-%M')
    writer = SummaryWriter(log_dir=args.save_dir+'/tensorboard'+runname)
    #begin training
    for epoch in range(args.start_epoch, args.epochs):
        # adjust learning rate
        if epoch+1 in args.step_epochs:
            lr = adjust_learning_rate(optimizer, 0.1, gradual= True)
        # main function
        train(train_loader, model, optimizer, epoch+1, cfg, writer)
        save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        }, True,
        os.path.join(args.save_dir, 'checkpoint_e%d.pth' % (epoch + 1)))
    writer.close()


def train(dataloader, model, optimizer, epoch, cfg, writer, warmup=False):
    logger = logging.getLogger('global')

    model.cuda()
    model.train()

    t0 = time.time()
    # collect data
    for iter, _input in enumerate(dataloader):
        lr = adjust_learning_rate(optimizer, 1, gradual=True)
        img_ids = _input[10]
        x = {
            'cfg': cfg,
            # 'image': torch.autograd.Variable(_input[0]).cuda(),
            'points': _input[1],
            'indices': _input[2],
            'num_pts': _input[3],
            'leaf_out': _input[4],
            'voxel_indices': _input[5],
            'voxel_points': torch.autograd.Variable(_input[6]).cuda(),
            'ground_plane': _input[7],
            'gt_bboxes_2d': _input[8],
            'gt_bboxes_3d': _input[9],
            'num_divisions': _input[11]
        }
        if x['gt_bboxes_3d'].cpu().numpy().shape[0] == 0:
            continue
        t1 = time.time()
        outputs = model(x)
        rpn_cls_loss = outputs['losses'][0]
        kl_loss = outputs['losses'][1]
        rpn_accuracy = outputs['accuracy'][0][0] / 100.
        #TODO: weights of losses need be adjusted
        loss = rpn_cls_loss + kl_loss

        t2 = time.time()
        # update optimizer
        optimizer.zero_grad()
        loss.backward(torch.ones_like(loss))
        #loss.reduce().backward()
        optimizer.step()

        t3 = time.time()
        # print('loss shape:', loss.size(), loss[0].size())
        # print('rpn_accuracy:', rpn_accuracy.size())
        #logger.info('Epoch: [%d][%d/%d]  Loss: %0.5f (cls: %.5f loc: %.5f img:%s acc: %.5f)'%
                    #(epoch, iter, len(dataloader), loss.cpu().item(), rpn_cls_loss.cpu().item(), rpn_loc_loss.cpu().item(),img_ids,rpn_accuracy.cpu().data.numpy()))
        i = (epoch - 1) * len(dataloader) + iter + 1
        i_time = t3 - t0
        n = args.epochs * len(dataloader)
        average_time = i_time
        remaining_time = (n - i) * average_time
        remaining_day = math.floor(remaining_time / 86400)
        remaining_hour = math.floor(remaining_time / 3600 - remaining_day * 24)
        remaining_min = math.floor(remaining_time / 60 - remaining_day * 1440 - remaining_hour * 60)
        sys.stdout.write("\r" + "Epoch: [%d][%d/%d]  Loss: %0.5f (cls: %.5f loc_kl: %.5f img:%s acc: %.5f) Progress: [%d%%], ETA %d:%02d:%02d" %
                    (epoch, iter, len(dataloader), loss.cpu().item(), rpn_cls_loss.cpu().item(),
                     kl_loss.cpu().item(), img_ids, rpn_accuracy.cpu().data.numpy(),i / n * 100, remaining_day, remaining_hour, remaining_min))

        #log_helper.print_speed((epoch - 1) * len(dataloader) + iter + 1, t3 - t0, args.epochs * len(dataloader))
        writer.add_scalar('total_loss', loss.cpu().item(), (epoch - 1) * len(dataloader) + iter + 1)
        writer.add_scalar('rpn_cls_loss', rpn_cls_loss.cpu().item(), (epoch - 1) * len(dataloader) + iter + 1)
        writer.add_scalar('kl_loss', kl_loss.cpu().item(), (epoch - 1) * len(dataloader) + iter + 1)
        t0 = t3

def validate(dataset, dataloader, model, cfg, epoch=-1):
    # switch to evaluate mode
    logger = logging.getLogger('global')
    model.cuda()
    model.eval()

    total_rc = 0
    total_gt = 0
    area_extents = np.asarray(cfg['shared']['area_extents']).reshape(-1, 2)
    bev_extents = area_extents[[0, 2]]

    score_threshold = cfg['test_rpn_proposal_cfg']['score_threshold']
    valid_samples = 0

    iou_threshold = 0.7
    evaluator = metrics.MetricsCalculator(iou_threshold=iou_threshold, bv=False)

    logger.info('start validate')
    for iter, _input in tqdm(enumerate(dataloader)):
        gt_boxes = _input[9]
        voxel_with_points = _input[6]
        batch_size = voxel_with_points.shape[0]
        # assert batch_size == 1
        img_ids = _input[10]

        x = {
            'cfg': cfg,
            'image': _input[0],
            'points': _input[1],
            'indices': _input[2],
            'num_pts': _input[3],
            'leaf_out': _input[4],
            'voxel_indices': _input[5],
            'voxel_points': torch.autograd.Variable(_input[6]).cuda(),
            'ground_plane': _input[7],
            'gt_bboxes_2d': _input[8],
            'gt_bboxes_3d': _input[9],
            'num_divisions': _input[11]
        }

        t0=time.time()
        outputs = model(x)
        outputs = outputs['predict']
        t2 =time.time()
        proposals = outputs[0].data.cpu().numpy()

        if torch.is_tensor(gt_boxes):
            gt_boxes = gt_boxes.cpu().numpy()

        for b_ix in range(batch_size):
            rois_per_points_cloud = proposals[proposals[:, 0] == b_ix]
            if gt_boxes.shape[0] != 0:
                gts_per_points_cloud = gt_boxes[b_ix]
                gts_per_points_cloud = gts_per_points_cloud[gts_per_points_cloud[:,3]>0] #Filter empty boxes (from batch)

                rois_per_points_cloud_anchor = box_3d_encoder.box_3d_to_anchor(rois_per_points_cloud[:, 1:1 + 7])
                gts_per_points_cloud_anchor = box_3d_encoder.box_3d_to_anchor(gts_per_points_cloud)
                rois_per_points_cloud_bev, _ = anchor_projector.project_to_bev(rois_per_points_cloud_anchor, bev_extents)
                gts_per_points_cloud_bev, _ = anchor_projector.project_to_bev(gts_per_points_cloud_anchor, bev_extents)

                # rpn recall
                num_rc, num_gt = bbox_helper.compute_recall(rois_per_points_cloud_bev, gts_per_points_cloud_bev)
                total_gt += num_gt
                total_rc += num_rc

                #Filter predictions by score (should add NMS)
                score_filter = rois_per_points_cloud[:, -1]>score_threshold
                filteredPred = rois_per_points_cloud[score_filter, 1:]

                #accumulate metrics
                evaluator.accumulate(filteredPred, gts_per_points_cloud)

                # visualisation
                if args.visual:
                    fig = draw_scene(x['points'][b_ix,:,0:3].numpy(), filteredPred, gts_per_points_cloud)

                    if args.save_as_figure:
                        mlab.view(41.362850002505866, 63.78581064281706, 103.86320312989403, [0.78470593, 1.99756785, 5.53537035])
                        mlab.roll(-0.4413787228555281)
                        mlab.savefig(filename=f'fig{iter*batch_size+b_ix}.png', figure=fig)
                        #  mlab.show()
                        #  input()
                        #  print(mlab.view())
                        #  print(mlab.roll())
                        mlab.close()
                        #Can generate gifs with
                        #ffmpeg -f image2 -framerate 2 -i fig%d.png out.gif
                    else:
                        input()

    prec, rec = evaluator.PR() 
    ap = evaluator.AP(prec, rec)
    logger.info(f'Test AP w/IoU {iou_threshold}: {ap.cpu().item()}')
    plt.plot(rec.cpu().numpy(), prec.cpu().numpy())
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title(f'Test AP w/IoU {iou_threshold}: {ap.cpu().item()}')
    plt.savefig('pr.pdf')

    #  logger.info('rpn300 recall=%f'% (total_rc/total_gt))

    return total_rc/total_gt

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    #if is_best:
    #    shutil.copyfile(filename, 'model_best.pth')

def adjust_learning_rate(optimizer, rate, gradual = True):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = None
    for param_group in optimizer.param_groups:
        if gradual:
            param_group['lr'] *= rate
        else:
            param_group['lr'] = args.lr * rate
        lr = param_group['lr']
    return lr

if __name__ == "__main__":
    main()
