import os
import gc
import json
import argparse
import sys
import numpy as np
import torch
from tqdm import tqdm
import mayavi.mlab as mlab
from lib.dataset.coop_dataset import CooperativeDataset, DataLoader, Transform
from lib.models.voxelnet import Voxelnet
from lib.functions import load_helper
from lib.functions.nms import nms
from lib.evaluator.iou import iou

def load_config(config_path):
    assert(os.path.exists(config_path))
    cfg = json.load(open(config_path, 'r'))
    for key in cfg.keys():
        if key != 'shared':
            cfg[key].update(cfg['shared'])
    return cfg


def build_data_loader(cfg, args):
    ext = torch.FloatTensor(cfg['shared']['area_extents']).view(3,2)
    ref = cfg['shared']['reference_loc']
    voxsize = cfg['shared']['voxel_size']
    maxpts = cfg['shared']['number_T']
    test_path = cfg['shared']['test_data']
    test_dataset = CooperativeDataset(test_path, ref, ext, voxsize, maxpts, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.batch_size, shuffle=False, pin_memory=False)
    return test_dataset, test_loader


@torch.no_grad()
def detectForCam(cfg, _input, model):

    detections = []
    gts = []
    gt_boxes = _input[9]
    voxel_with_points = _input[6]
    batch_size = voxel_with_points.shape[0]

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
    outputs = model(x)
    outputs = outputs['predict']
    proposals = outputs[0].data.cpu().numpy()

    if torch.is_tensor(gt_boxes):
        gt_boxes = gt_boxes.cpu().numpy()
    b_ix = 0
    rois_per_points_cloud = proposals[proposals[:, 0] == b_ix]
    score_threshold = cfg['test_rpn_proposal_cfg']['score_threshold']
    if gt_boxes.shape[0] != 0:
        gts_per_points_cloud = gt_boxes[b_ix]
        gts_per_points_cloud = gts_per_points_cloud[gts_per_points_cloud[:,3] > 0] #Filter empty boxes (from batch)

        #Filter predictions by score
        score_filter = rois_per_points_cloud[:, -1] > score_threshold

        filteredPred = rois_per_points_cloud[score_filter, 1:]

        if gts_per_points_cloud.shape[0] == 0:
            return x['points'],[],[]

        if filteredPred.shape[0] == 0:
            filteredPred = np.zeros((1,8))

        #accumulate metrics
        detections.append(filteredPred)
        gts.append(gts_per_points_cloud)

    return x['points'], detections, gts

def getAnchRepr(bboxes):
    xyz = bboxes[:, :3]
    extensions = bboxes[:,3:6]/2
    pry = np.zeros(xyz.shape)
    pry[:,1] = bboxes[:, 6]
    anchRepr = np.hstack((xyz,pry))
    anchRepr = np.hstack((anchRepr,extensions))
    return anchRepr

def getBBpts(repr):
    '''Given a representation get 8 BB edge points in world reference. Out shape [8,3]'''

    x,y,z = repr[0],repr[1],repr[2]
    pitch,roll,yaw = repr[3],repr[4],repr[5]
    trActor = Transform(x,y,z,yaw,pitch,roll)

    #BB Extensions (half length for each dimension)
    ex, ey, ez = repr[6],repr[7],repr[8]

    #Get 8 coordinates
    bbox_pts = np.array([
    [  ex,   ey,   ez],
    [- ex,   ey,   ez],
    [- ex, - ey,   ez],
    [  ex, - ey,   ez],
    [  ex,   ey, - ez],
    [- ex,   ey, - ez],
    [- ex, - ey, - ez],
    [  ex, - ey, - ez]
    ])
    #Transform the bbox points from actor ref frame to global reference
    bbox_pts = trActor.transform_points(bbox_pts)
    return bbox_pts

def drawBox(b,fig,color,line_width):
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                    line_width=line_width, figure=fig)

        i, j = k + 4, (k + 1) % 4 + 4
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                    line_width=line_width, figure=fig)

        i, j = k, k + 4
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                    line_width=line_width, figure=fig)
def drawActors(reprs, fig, color, line_width=1):

    '''Given array of n actor repr, generate vertices and plot lines in 3D vis'''
    for r in reprs:
        b = getBBpts(r)
        drawBox(b,fig,color,line_width)
def drawActorswithScores(reprs, fig, color, score, iou, line_width=1):

    '''Given array of n actor repr, generate vertices and plot lines in 3D vis'''
    for r,s,u in zip(reprs,score,iou):
        b = getBBpts(r)
        e = np.random.randint(0,3,1)[0]
        mlab.text3d(b[e,0],b[e,1],b[e,2],str(s)[:5]+'  '+str(u)[:5],color=color,figure=fig,line_width=line_width,scale=0.3)
        drawBox(b, fig, color, line_width)

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config/tjunc.json', required=False,
                    help='path to config file')
parser.add_argument('--checkpoint', default='assets/saved_models/checkpoint_TJunc.pth', type=str,
                    required=False,
                    help='path to model checkpoint')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    help='mini-batch size (default: 4)')
parser.add_argument('--cameras', type=str, default='012345',
                    help='string containing the ids of cameras to be used in the evaluation. E.g. 037 fuses data from cameras 0,3,7. Default: all available sensors are used')
args = parser.parse_args()
cfg = load_config(args.config)


test_dataset0, test_loader0 = build_data_loader(cfg, args)
test_dataset0.selectedCameras = [0]
test_iter0 = iter(test_loader0)
test_dataset1, test_loader1 = build_data_loader(cfg, args)
test_dataset1.selectedCameras = [1]
test_iter1 = iter(test_loader1)
test_dataset2, test_loader2 = build_data_loader(cfg, args)
test_dataset2.selectedCameras = [2]
test_iter2 = iter(test_loader2)
test_dataset3, test_loader3 = build_data_loader(cfg, args)
test_dataset3.selectedCameras = [3]
test_iter3 = iter(test_loader3)
test_dataset4, test_loader4 = build_data_loader(cfg, args)
test_dataset4.selectedCameras = [4]
test_iter4 = iter(test_loader4)
test_dataset5, test_loader5 = build_data_loader(cfg, args)
test_dataset5.selectedCameras = [5]
test_iter5 = iter(test_loader5)

def chooseDataloader(cam):
    if cam == '0':
        return test_iter0
    if cam == '1':
        return test_iter1
    if cam == '2':
        return test_iter2
    if cam == '3':
        return test_iter3
    if cam == '4':
        return test_iter4
    if cam == '5':
        return test_iter5

# available cameras
cameras = args.cameras if len(args.cameras) > 0 else [0, 1, 2, 3, 4, 5]
# load model checkpoint
device = torch.device("cuda:0")
model = Voxelnet(cfg=cfg)
assert os.path.isfile(args.checkpoint), '{} is not a valid file'.format(args.checkpoint)
model = load_helper.load_checkpoint(model, args.checkpoint)
model.cuda()
model.eval()

print('Show late fusion predictions')
detectionsLate = []
gts = []

for i in range(0,1000):
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000))
    mlab.clf(figure=fig)
    for cam in cameras:
        #Set camera to use
        dataloder = chooseDataloader(cam)
        _input = next(dataloder)
        #Get predictions and gt
        points, dets, gts = detectForCam(cfg, _input, model)
        pts = points[0]
        dets = np.concatenate(dets)
        dets = nms(dets, 0.1, 0.01)
        if dets.size == 0:
            continue
        #get detection scores
        detection_scores = dets[:,-1]
        #get ious
        # Ignore score
        dets = dets[:, :7]
        # Calculates IoU of all pred and all gts
        predR = np.repeat(dets,gts[0].shape[0],axis=0).reshape(-1,7)
        gtR = np.repeat(gts,dets.shape[0],axis=0).reshape(-1,7)
        predR = torch.from_numpy(predR).float()
        gtR = torch.from_numpy(gtR).float()
        ious = iou(predR, gtR, bv=True).reshape(dets.shape[0], gts[0].shape[0])  # Results in vector [n_pred, n_gt] of IOUs
        # Get max IOU for each pred
        mIOU, gIdx = torch.max(ious, dim=1)
        mIOU = mIOU.numpy()
        # Seed for random colours
        color = tuple(np.random.rand(3).tolist())
        # Plot points
        detections = getAnchRepr(dets)
        groundtruth = getAnchRepr(gts[0])
        mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='point', color=color, figure=fig)
        drawActorswithScores(detections, fig, color, detection_scores,mIOU)
    drawActors(groundtruth, fig, (1,0,0), line_width= 2)
    try:
        mlab.show()
    except KeyboardInterrupt:
        continue
















