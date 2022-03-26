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
from lib.functions.nms import nmswithVar
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
    uncertainties = []
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
    variations = outputs[1].data.cpu().numpy()

    if torch.is_tensor(gt_boxes):
        gt_boxes = gt_boxes.cpu().numpy()
    b_ix = 0
    rois_per_points_cloud = proposals[proposals[:, 0] == b_ix]
    vars_per_points_cloud = variations[variations[:,0] == b_ix]
    score_threshold = cfg['test_rpn_proposal_cfg']['score_threshold']
    if gt_boxes.shape[0] != 0:
        gts_per_points_cloud = gt_boxes[b_ix]
        gts_per_points_cloud = gts_per_points_cloud[gts_per_points_cloud[:,3] > 0] #Filter empty boxes (from batch)

        #Filter predictions by score
        score_filter = rois_per_points_cloud[:, -1] > score_threshold

        filteredPred = rois_per_points_cloud[score_filter, 1:]
        filteredVar = vars_per_points_cloud[score_filter, 1:]

        if gts_per_points_cloud.shape[0] == 0:
            return x['points'],[],[]

        if filteredPred.shape[0] == 0:
            filteredPred = np.zeros((1,8))
            filteredVar = np.zeros((1,7))
        #accumulate metrics
        detections.append(filteredPred)
        uncertainties.append(filteredVar)
        gts.append(gts_per_points_cloud)
    return x['points'], detections, uncertainties, gts

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
def drawfusedbox(limits,fig,color,line_width=1):
    for m in limits:
        r = getCenterRep(m.reshape(1, -1))
        r = getAnchRepr(r)
        b = getBBpts(r.reshape(-1))
        drawBox(b, fig,color, line_width)

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
def computeGaussian(mean, delta, newvalue):
    numerator = np.exp(-0.5*np.power((newvalue-mean)/delta,2))
    denominator = delta*np.power(2*np.pi,0.5)
    return  numerator/denominator

def genGaussian(m,v):
    mean = m
    delta = v
    newvalue = v*np.random.rand(1)
    newvalue = m -0.5*v +newvalue
    possibility = computeGaussian(mean, delta, newvalue)
    return newvalue,possibility


def drawActorswithUncertainty(reprs, limits, fig, color, vars, score, iou, line_width=4):
    '''Given array of n actor repr, generate vertices and plot lines in 3D vis'''
    #plot center box
    sample_num = 5
    for r,mean,v,s,u in zip(reprs,limits,vars,score,iou):

        b_center = getBBpts(r)
        e = np.random.randint(0, 3, 1)[0]
        mlab.text3d(b_center[e, 0], b_center[e, 1], b_center[e, 2], str(s)[:5] + '  ' + str(u)[:5], color=color, figure=fig,
                    line_width=line_width, scale=0.3)
        drawBox(b_center, fig, color, line_width)
        #generate random edges

        for i in range(sample_num):
            m = np.zeros(mean.shape)
            m[:] = mean[:]
            #random pick a coordinat to change
            adjustID = np.random.randint(0,6,1)[0]
            gaussian_mean = m[adjustID]
            gaussian_var = v[adjustID]
            newvalue, possibility = genGaussian(gaussian_mean,gaussian_var)
            m[adjustID] = newvalue
            r = getCenterRep(m.reshape(1,-1))
            r = getAnchRepr(r)
            b = getBBpts(r.reshape(-1))
            # print(b)
            # test = input()
            newr = color[0] * possibility
            newg = color[1] * possibility
            newb = color[2] * possibility
            newcolor = (newr[0],newg[0],newb[0])
            drawBox(b, fig, newcolor, line_width*possibility)


def nmsProcessing(dets,gts,vars):
    iouthreshould = 0.1
    scores = dets[:,7]
    dets = dets[:, :7]
    # Calculates IoU of all pred and all gts
    predR = np.repeat(dets, gts[0].shape[0], axis=0).reshape(-1, 7)
    gtR = np.repeat(gts, dets.shape[0], axis=0).reshape(-1, 7)
    predR = torch.from_numpy(predR).float()
    gtR = torch.from_numpy(gtR).float()
    ious = iou(predR, gtR, bv=True).reshape(dets.shape[0], gts[0].shape[0])  # Results in vector [n_pred, n_gt] of IOUs
    # Get max IOU for each pred
    mIOU, vehicleID = torch.max(ious, dim=1)
    mIOU = np.asarray(mIOU)
    vehicleID = np.asarray(vehicleID)
    index = mIOU > iouthreshould
    if mIOU.shape[0] == 1:
        if mIOU[0] < iouthreshould:
            return [],[],[],[],[]
        else:
            index=[0]
    dets = dets[index]
    mIOU = mIOU[index]
    vehicleID = vehicleID[index]
    vars = vars[index,:]
    scores = scores[index]
    return dets, scores, mIOU, vehicleID, vars

def getDeltas(dets,gts,vehicleID):
    vehicleID = vehicleID.tolist()
    gts = gts[0]
    gts = gts[vehicleID,:]
    deltas = dets-gts
    return deltas


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config/tjunc.json', required=False,
                    help='path to config file')
parser.add_argument('--checkpoint', default='/mnt/tank/shaocheng/3Dfusion/saved_models_kl_limits/checkpoint_e30.pth', type=str,
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
test_dataset_all, test_loader_all = build_data_loader(cfg, args)
test_iter_all = iter(test_loader_all)
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

def get2points(boxes):
    xyz = boxes[:, :3]
    l_g = boxes[:, 3]
    w_g = boxes[:, 4]
    h_g = boxes[:, 5]
    yaw = boxes[:, 6]

    ey = h_g / 2
    ex = l_g / 2
    ez = w_g / 2

    cy = np.cos(yaw)
    sy = np.sin(yaw)

    extentions = np.zeros(xyz.shape)
    extentions[:, 0] = sy * ez + cy * ex
    extentions[:, 1] = ey
    extentions[:, 2] = cy * ez - sy * ex

    points_all_p = xyz + extentions
    points_all_n = xyz - extentions
    newboxes = np.concatenate((points_all_p, points_all_n, boxes[:, 6:]), axis=1)
    return newboxes
    
def getCenterRep(boxes_limits):
    # convert boxes in 2-points format to [x,y,z,w,l,h]
    xyz = boxes_limits[:, :3] + boxes_limits[:, 3:6]
    xyz = 0.5 * xyz
    delta = boxes_limits[:, :3] - boxes_limits[:, 3:6]
    yaw = boxes_limits[:, 6]
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    h = delta[:, 1]
    w = delta[:, 0] * sy + delta[:, 2] * cy
    l = delta[:, 0] * cy - delta[:, 2] * sy
    boxes_3d = np.concatenate((xyz, l.reshape((-1, 1)), w.reshape((-1, 1)), h.reshape((-1, 1)),
                               boxes_limits[:,6:]), axis=1)
    return boxes_3d

def gaussionfusion(miu0,miu1,delta0,delta1):
    square_d0 = np.power(delta0, 2)
    square_d1 = np.power(delta1, 2)
    new_miu = square_d1 * miu0 + square_d0 * miu1
    new_miu = new_miu/(square_d0 + square_d1)

    new_delta = delta0 * delta1
    new_delta = new_delta/np.power(square_d0+square_d1,0.5)
    return new_miu,new_delta

def getfusedboxes(fusionvehicles, fusionlimits, fusionvariation, gts):

    fused_points = np.zeros(gts[0].shape)
    fused_vars = np.zeros(gts[0].shape)
    num_boxes = np.zeros([1, gts[0].shape[0]])
    for vehicleID, limit_points, variations in zip(fusionvehicles, fusionlimits, fusionvariation):
        for v, p, delta in zip(vehicleID, limit_points, variations):
            if num_boxes[0, v] == 0:
                fused_points[v,:] = p
                fused_vars[v,:] = delta
                num_boxes[0, v] = num_boxes[0, v] + 1
            else:
                print("Fusion : ")
                print(fused_points[v,:])
                print(p)
                fused_points[v,:], fused_vars[v,:] = gaussionfusion(fused_points[v,:], p,fused_vars[v,:],delta)
                print(fused_points[v,:])
                num_boxes[0, v] = num_boxes[0, v] + 1
    idx = num_boxes[0,:]>1
    fused_points = fused_points[idx,:]
    fused_vars = fused_vars[idx,:]
    return  fused_points, fused_vars
# available cameras
cameras = args.cameras if len(args.cameras) > 0 else [0, 1, 2, 3, 4, 5]
# load model checkpoint
#device = torch.device("cuda:2")
cuda = True if torch.cuda.is_available() else False
if cuda == True:
    torch.cuda.set_device(2)

model = Voxelnet(cfg=cfg)
assert os.path.isfile(args.checkpoint), '{} is not a valid file'.format(args.checkpoint)
model = load_helper.load_checkpoint(model, args.checkpoint)
model.cuda()
model.eval()

# print('Show late fusion predictions')
detectionsLate = []
gts = []
root = '/home/shaoche/code/coop-3dod-infra/earlyoffsets/'
plot1 = True
plot2 = False
for i in range(2,1000,100):
    print(i)
    if plot1:
        fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000))
        mlab.clf(figure=fig)
    if plot2:
        fig2 = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000))
        mlab.clf(figure=fig2)
    fusionvehicles=[]
    fusionlimits=[]
    fusionvariation=[]
    for cam in cameras:
        #Set camera to use
        dataloder = chooseDataloader(cam)
        _input = next(dataloder)
        #Get predictions and gt
        points, dets, vars, gts = detectForCam(cfg, _input, model)

        pts = points[0]
        dets = np.concatenate(dets)
        vars = np.concatenate(vars)
        vars = np.exp(vars)
        vars = np.sqrt(vars)
        #change dets, vars to center form in order for nms calculation

        dets = getCenterRep(dets)


        gts[0]= getCenterRep(gts[0])
        # dets = nms(dets,0.1,0.01)
        dets, vars = nmswithVar(dets, vars, 0.1, 0.01)

        if dets.size == 0:
            continue

        #Get detection scores

        dets, detection_scores, mIOU, vehicleID,vars = nmsProcessing(dets,gts,vars)
        if len(dets) == 0:
            continue
    #     # Write to file
    #     for vid, delta in zip(vehicleID, deltas):
    #         print(vid)
    #         f.write(str(cam) + ' ' + str(vid))
    #         for element in delta:
    #             f.write(' ' + str(element))
    #         f.write('\n')
    # f.close()
        # Visualization
        color = tuple(np.random.rand(3).tolist())
        # Plot points
        detections = getAnchRepr(dets)
        limit_points = get2points(dets)
        fusionlimits.append(limit_points)
        fusionvariation.append(vars)
        fusionvehicles.append(vehicleID)
        groundtruth = getAnchRepr(gts[0])
        if plot1:
            mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='point', color=color, figure=fig)
            # color = tuple(np.random.rand(3).tolist())
            drawActorswithUncertainty(detections,limit_points, fig, color, vars, detection_scores,mIOU)

        if plot2:
            drawActorswithScores(detections, fig2, color, detection_scores, mIOU)
    fboxes, fvars = getfusedboxes(fusionvehicles, fusionlimits, fusionvariation, gts)
    if plot1:
        drawActors(groundtruth, fig, (1,0,0), line_width= 3)
        drawfusedbox(fboxes, fig, (0,1,0), line_width=3)
        try:
            mlab.show()
        except KeyboardInterrupt:
            continue
    if plot2:
        drawActors(groundtruth, fig2, (1, 0, 0), line_width=3)
        drawfusedbox(fboxes, fig2, (0, 1, 0), line_width=3)

    #f.close()















