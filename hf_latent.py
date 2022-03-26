import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import tensorflow as tf
from logger import Logger
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="T-junction-2cam-exchange-8fe", help='name of the dataset')
parser.add_argument('--pretrained_name', type=str, default="none", help='load pretrained model')
parser.add_argument('--model_name', type=str, default="render", help='name of the model')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--test_batch_size', type=int, default=4, help='size of the batches')
parser.add_argument('--lr', type=float, default=1e-5, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--img_height', type=int, default=32, help='size of image height')
parser.add_argument('--img_width', type=int, default=48, help='size of image width')
parser.add_argument('--sample_interval', type=int, default=100, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=10, help='interval between model checkpoints')
parser.add_argument('--root_dir', type=str, default='/mnt/tank/shaocheng/data_association', help='root directory')
opt = parser.parse_args()
print(opt)

os.makedirs('%s/%s/%s/images' % (opt.root_dir, opt.dataset_name, opt.model_name), exist_ok=True)
os.makedirs('%s/%s/%s/saved_models' % (opt.root_dir, opt.dataset_name, opt.model_name), exist_ok=True)
os.makedirs('%s/%s/%s/logs' % (opt.root_dir, opt.dataset_name, opt.model_name), exist_ok=True)
cuda = True if torch.cuda.is_available() else False
if cuda == True:
    torch.cuda.set_device(1)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# Loss functions
criterion_ae = torch.nn.MSELoss()
criterion_regu = torch.nn.MSELoss()
#latent_uv_regu = torch.nn.L1Loss()

# camera intrinsics
def get_Camera_parameters():
    root = '/home/shaoche/code/coop-3dod-infra/images/cameras/'
    path_intrinsic = root + 'cam_intrinsic.txt'
    intrinsic = np.loadtxt(path_intrinsic)
    intrinsic = torch.from_numpy(intrinsic)
    intrinsic = Variable(intrinsic.type(Tensor))
    return intrinsic
def get_projCam_parameters(vfov=45,hfov=60,pixel_width=opt.img_width,pixel_height=opt.img_height):
    camera_intrinsics = np.zeros((3,4))
    camera_intrinsics[2,2] = 1
    camera_intrinsics[0,0] = (pixel_width/2.0)/math.tan(math.radians(hfov/2.0))
    camera_intrinsics[0,2] = pixel_width/2.0
    camera_intrinsics[1,1] = (pixel_height/2.0)/math.tan(math.radians(vfov/2.0))
    camera_intrinsics[1,2] = pixel_height/2.0
    camera_intrinsics = camera_intrinsics[:3, :3]
    camera_intrinsics = torch.from_numpy(camera_intrinsics)
    camera_intrinsics = Variable(camera_intrinsics.type(Tensor))
    return camera_intrinsics
render_intrinsic = get_projCam_parameters()
orig_intrinsic = get_Camera_parameters()
# Initialize model
#autoencoder = encode3Dmodel(intrinsic=camera_intrinsics, img_height=opt.img_height, img_width=opt.img_width)
autoencoder = rendermodel(intrinsic=render_intrinsic, img_height=opt.img_height, img_width=opt.img_width)
IMG_FULL_HEIGHT = 300
IMG_FULL_WIDTH = 400
if cuda:
    autoencoder = autoencoder.cuda()
    criterion_ae.cuda()
    criterion_regu.cuda()

if opt.epoch != 0:
    # Load pretrained models
    autoencoder.load_state_dict(torch.load('%s/%s/%s/saved_models/ae_%d.pth' % (opt.root_dir, opt.pretrained_name, opt.model_name,opt.epoch)))
else:
    #autoencoder.apply(weights_init_normal)
    init_weights(autoencoder, init_type='normal', init_gain=0.0005)

# Optimizers
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
transforms_ = [ transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.ToTensor(),transforms.Normalize(mean=(0.5,0.5,0.5),std=(1,1,1))]


traindataloader = DataLoader(coopDataset('/home/shaoche/code/coop-3dod-infra/images/train_image_list.txt',
                                          transforms_=transforms_,phase="train"), batch_size=opt.batch_size, shuffle=True)
testdataloader = DataLoader(coopDataset('/home/shaoche/code/coop-3dod-infra/images/test_image_list.txt',
                                          transforms_=transforms_,phase="test"), batch_size=opt.test_batch_size, shuffle=True)


def rotation_angle(diff_r):
    diff_r = diff_r.cpu().detach().numpy()
    trace_r = np.trace(diff_r)
    cos_value = (trace_r-1)*0.5
    cos_value = min(1.0,cos_value)
    cos_value = max(-1.0,cos_value)
    angle = math.acos(cos_value)
    angle = angle*180/math.pi
    return angle


def pose_evaluate(real_rot_a2b, real_trans_a2b, pred_rot_a2b, pred_trans_a2b):
    diff_r = torch.bmm(pred_rot_a2b.transpose(1,2), real_rot_a2b)
    angle = 0
    traslation=0
    for i in range(opt.batch_size):
        diff_r_single = diff_r[i]
        angle += rotation_angle(diff_r_single)
        traslation += (real_trans_a2b[i]-pred_trans_a2b[i]).norm(2)
    return angle/opt.batch_size, traslation/opt.batch_size


def estimate_pairwise_pose(latent_3d_a,latent_3d_b):
    #Bx3xN
    latent_3d_a = latent_3d_a.cpu().detach()
    latent_3d_b = latent_3d_b.cpu().detach()

    #compute centroids of point sets
    #Bx3x1
    center_a = torch.mean(latent_3d_a, dim=-1)
    center_b = torch.mean(latent_3d_b, dim=-1)

    #compute centered vectors
    centered_latent_a = latent_3d_a - torch.unsqueeze(center_a,dim=-1)
    centered_latent_b = latent_3d_b - torch.unsqueeze(center_b,dim=-1)

    #compute the covariance matrix
    covariance_matrix = torch.bmm(centered_latent_a, centered_latent_b.transpose(1,2))

    #compute the svd decomposition
    batch_size = covariance_matrix.shape[0]
    #compute pairwise RT
    u, sigma, v_transpose = torch.svd(covariance_matrix[0])
    u_transpose = u.transpose(0,1)
    v = v_transpose.transpose(0,1)
    detflag = torch.eye(3)
    detflag[-1, -1] = torch.det(v.mm(u_transpose))

    u_transpose = torch.unsqueeze(u_transpose,dim=0)
    v = torch.unsqueeze(v,dim=0)
    detflag = torch.unsqueeze(detflag,dim=0)
    for i in range(1, batch_size):
        u_single, sigma_single, v_transpose_single = torch.svd(covariance_matrix[i])
        v_single = v_transpose_single.transpose(0,1)
        u_transpose_single = u_single.transpose(0,1)
        detflag_single = torch.eye(3)
        detflag_single[-1, -1] = torch.det(v_single.mm(u_transpose_single))

        u_transpose = torch.cat((u_transpose,torch.unsqueeze(u_transpose_single,dim=0)),dim=0)
        v = torch.cat((v,torch.unsqueeze(v_single,dim=0)),dim=0)
        detflag = torch.cat((detflag,torch.unsqueeze(detflag_single,dim=0)),dim=0)

    r_estimate = torch.bmm(v, detflag)
    r_estimate = torch.bmm(r_estimate, u_transpose)
    t_estimate = torch.unsqueeze(center_b,dim=2) - torch.bmm(r_estimate,torch.unsqueeze(center_a,dim=2))

    return r_estimate, t_estimate.transpose(1, 2)


def relativeRT(batch_size, extrinsic_a, extrinsic_b):

    rotation_a = extrinsic_a[:, :, :3]
    trans_a = extrinsic_a[:, :, -1]

    rotation_b = extrinsic_b[:, :, :3]
    trans_b = extrinsic_b[:, :, -1]

    cam2world_a = rotation_a.transpose(1, 2).view((batch_size, 3, 3)).float()
    world2cam_b = rotation_b.view((batch_size, 3, 3)).float()

    # determine crop1 to crop2
    rotation_a2b = torch.bmm(world2cam_b, cam2world_a)

    trans_a = trans_a.view((batch_size, 1, 3)).float()
    trans_b = trans_b.view((batch_size, 1, 3)).float()
    trans_a2b = torch.bmm(cam2world_a,trans_a.transpose(1,2))#Bx3x1
    trans_a2b = torch.bmm(world2cam_b,trans_a2b)#Bx3x1

    trans_a2b = trans_b - trans_a2b.transpose(1,2)
    return rotation_a2b, trans_a2b


def sample_images(batches_done):

    """Saves a generated sample from the validation set"""

    test_samples = iter(testdataloader)
    test_batch = next(test_samples)

    input_x = test_batch['img_crop']
    extrinsic = test_batch['extrinsic']
    extrinsic_a = extrinsic[:, :3, :]
    extrinsic_b = extrinsic[:, 3:, :]
    center = test_batch['center']
    scale = test_batch['scale']
    scale[:, 0, :] = scale[:, 0, :] / opt.img_width
    scale[:, 1, :] = scale[:, 1, :] / opt.img_height
    real_rot_a2b, real_trans_a2b = relativeRT(opt.test_batch_size, extrinsic_a, extrinsic_b)

    input_x = Variable(input_x.type(Tensor))
    extrinsic = Variable(extrinsic.type(Tensor))
    output_type = ['output', 'latent_fg_a', 'latent_fg_b', 'render_marker']
    output_config=autoencoder(input_x, extrinsic, center, scale, "test",output_type)
    #pred_rot_a2b, pred_trans_a2b = estimate_pairwise_pose(output_config['latent_3d_a'], output_config['latent_3d_b'])


    input_a = input_x[:, :3, :, :]
    input_b = input_x[:, 3:, :, :]
    output_a = output_config['output'][:, :3, :, :]
    output_b = output_config['output'][:, 3:, :, :]

    img_sample = torch.cat((input_a.data, input_b.data), -2)
    pre_sample = torch.cat((output_a.data, output_b.data), -2)

    save_image(img_sample, '%s/%s/%s/images/%s_img.png' % (opt.root_dir, opt.dataset_name, opt.model_name, batches_done), nrow=4, normalize=True)
    save_image(pre_sample, '%s/%s/%s/images/%s_imgpre.png' % (opt.root_dir, opt.dataset_name, opt.model_name, batches_done), nrow=4, normalize=True)
    save_image(output_config['render_marker'], '%s/%s/%s/images/%s_latent.png' % (opt.root_dir, opt.dataset_name, opt.model_name, batches_done), nrow=4, normalize=True)

# ----------
#  Training
# ----------


prev_time = time.time()
logger = SummaryWriter('%s/%s/%s/logs' % (opt.root_dir, opt.dataset_name, opt.model_name))
regularization = False
for epoch in range(opt.epoch, opt.n_epochs+1):
    for i, batch in enumerate(traindataloader):

        input_x = batch['img_crop']
        img = batch['img_crop']
        extrinsic = batch['extrinsic']
        center = batch['center']
        scale = batch['scale']
        scale[:, 0, :] = scale[:, 0, :]/opt.img_width
        scale[:, 1, :] = scale[:, 1, :]/ opt.img_height
        # extrinsic_a = extrinsic[:, :3, :]
        # extrinsic_b = extrinsic[:, 3:, :]
        #gt_rot_a2b, gt_trans_a2b = relativeRT(opt.batch_size, extrinsic_a, extrinsic_b)

        input_x = Variable(input_x.type(Tensor))
        img = Variable(img.type(Tensor))
        extrinsic = Variable(extrinsic.type(Tensor))
        output_type = ['output', 'latent_3d_a', 'latent_3d_b', 'latent_3d_a2b', 'latent_3d_b2a', 'mse_uv', 'mse_depth', 'mse_fg']
        output_config = autoencoder(input_x, extrinsic, center, scale, "train", output_type)
        #for tag, value in autoencoder.named_parameters():
        #    print(tag)
        #    print(value)

        # ---------------------
        #  pose accuracy
        # --------------------
        pred_rot_a2b, pred_trans_a2b = estimate_pairwise_pose(output_config['latent_3d_a'], output_config['latent_3d_b'])
        #rot_error, trans_error = pose_evaluate(gt_rot_a2b, gt_trans_a2b, pred_rot_a2b, pred_trans_a2b)

        # ---------------------
        #  Train autoencoder
        # --------------------
        optimizer.zero_grad()
        # loss
        pred = output_config['output']
        #img = img[:,:,2:-2,:]
        pred_loss = criterion_ae(pred, img)

        if regularization :
            regu_loss_uv = output_config['mse_uv']
            if not regu_loss_uv:
                loss = pred_loss
                regu_loss = torch.Tensor([0])
            else:
                regu_loss_depth = output_config['mse_depth']
                regu_loss_fg = output_config['mse_fg']
                regu_loss = 0.5*regu_loss_uv + regu_loss_depth + regu_loss_fg
                loss = pred_loss + lambda_3D*regu_loss
        else:
            loss = pred_loss
            regu_loss = pred_loss
        loss.backward()
        optimizer.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(traindataloader) + i
        batches_left = opt.n_epochs * len(traindataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write("\r" + opt.dataset_name + "-"+ opt.model_name+ "[Epoch %d/%d] [Batch %d/%d] [Overall loss: %f] [P loss: %f, R loss: %f] ETA: %s" %
                                                        (epoch, opt.n_epochs,
                                                        i, len(traindataloader),
                                                        loss.item(), pred_loss.item(),
                                                        regu_loss.item(),time_left))
        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)
        #--------------tensor board--------------------------------#
        if batches_done % 10 == 0:
            #info = {'loss': loss.item(), 'img_loss': pred_loss.item(), 'regu_loss' : regu_loss.item(), 'rot_error' : rot_error, 'trans_error':trans_error}
            info = {'loss': loss.item(), 'img_loss': pred_loss.item(), 'regu_loss': regu_loss.item()}
            for tag, value in info.items():
                logger.add_scalar(tag, value, batches_done)
            for tag, value in autoencoder.named_parameters():
                tag = tag.replace('.','/')
                logger.add_histogram(tag, value.data.cpu().numpy(),batches_done)
                #logger.add_histogram(tag+'grad', value.grad.data.cpu().numpy(),batches_done+1)
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(autoencoder.state_dict(), '%s/%s/%s/saved_models/ae_%d.pth' %
                   (opt.root_dir, opt.dataset_name, opt.model_name, epoch))

