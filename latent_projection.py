import torch
from torch import nn
import numpy as np
import math





class ProjectPoint2ImageSoft(nn.Module):
  def __init__(self, K, uv_only=False, transform_feat=False):
    super(ProjectPoint2ImageSoft, self).__init__()
    self.K = K
    im_width = int(K[0,2]*2)
    im_height = int(K[1,2]*2)
    self.im_height = im_height 
    self.im_width = im_width 
    self.transform_feat = transform_feat
    
    ui, vi = np.meshgrid(range(im_width), range(im_height))
    grid = np.hstack((vi.reshape(-1,1), ui.reshape(-1,1))).astype(np.float32)
    self.grid = torch.tensor(grid).to(K.device)

    # params of gaussian kernel at every projected point
    #self.sigma = 1.0
    #self.sigma = K[0,0].item()/64
    self.sigma = K[0,0].item()/256

    # if uv_only then return the re-projected uv coordinates, not the image
    self.uv_only = uv_only



  def forward_single(self, RT, pts_3d, pts_feat, pts_scale):
    R = RT[:3,:3]
    T = RT[:3, 3]

    # transform points from world coordinate to camera coordinate 
    points_local = torch.t(torch.mm(R, torch.t(pts_3d))) + T
    points_local = torch.t(points_local)

    # perspective projection
    points_proj = torch.mm(self.K[:3,:3], points_local)
    points_mask = points_proj[2,:]>0.1
    points_proj = points_proj[points_mask]
    u = points_proj[0,:]/points_proj[2,:]
    v = points_proj[1,:]/points_proj[2,:]
    uv = torch.cat((v.reshape(-1,1), u.reshape(-1,1)),dim=1)

    # project points to image plance with soft weights
    # to differientiate to the geometry
    distance = uv.view(-1,1,2) - self.grid.view(1,-1,2)  # N x (HxW) x 2
    distance_sq = distance[:,:,0]**2 + distance[:,:,1]**2 # N x (HxW) 

    weight = torch.exp(-distance_sq / (pts_scale.view(-1,1) * self.sigma * self.sigma))

    # concatenate the depth as an additional channel
    # TODO: discuss other options instead of soft projection of depth
    pts_feat = torch.cat((pts_feat, points_proj[2:3,:].view(-1,1)), dim=1)

    # sum up features from all 3d points for each grid point
    img = torch.mm(torch.t(pts_feat), weight) #  (C x N) x (N x (HxW)) --> C x (HxW)
    img = img.view(1, -1, self.im_height, self.im_width) 
    return img

  def forward_batch(self, RT, center_3d, scale, pts_3d, pts_feat, pts_scale, pts_marker):
    # params:
    #   RT: Bx3x4
    #   pts_3d: Bx3xN
    #   pts_feat: BxCxN
    #   pts_scale: BxN
    #import pdb;pdb.set_trace()
    bs = RT.shape[0]
    R = RT[:, :3,:3]
    T = RT[:, :3, 3]

    if pts_3d.shape[1]==1: # larger blob if there is single point
        self.sigma = self.K[0,0].item()/16.
    else:
        self.sigma = self.K[0,0].item()/32.

    # transform points from world coordinate to camera coordinate
    points_local_render = torch.bmm(R,pts_3d) + T.view(bs,3,1)-center_3d #Bx3xN
    points_local = torch.ones_like(points_local_render)
    points_local[:, 0, :] = points_local_render[:, 0, :] / scale[:, :, 0]
    points_local[:, 1, :] = points_local_render[:, 1, :] / scale[:, :, 1]
    points_local[:, 2, :] = points_local_render[:, 2, :]
    # perspective projection
    K_expand = torch.unsqueeze(self.K,dim=0).expand(points_local.shape[0],3,3)
    #points_proj = self.K.unsqueeze(0) @ points_local # Bx3xN
    points_proj = torch.bmm(K_expand,points_local) # Bx3xN

    points_mask = points_proj[:,2]>0.1 #BxN


    u = points_proj[:,0,:]/points_proj[:,2,:].clamp(min=0.1)
    v = points_proj[:,1,:]/points_proj[:,2,:].clamp(min=0.1)
    uv = torch.cat((v.reshape(bs,-1,1), u.reshape(bs,-1,1)),dim=2)# BxNx2

    if self.uv_only:
      uvz = torch.cat((uv, points_proj[:,2,:].reshape(bs,-1,1)),dim=2)
      return uvz
    # project points to image plane with soft weights
    # to differientiate to the geometry
    distance = uv.view(bs,-1,1,2) - self.grid.view(1,1,-1,2).expand(bs,-1,-1,-1)  # B x N x (HxW) x 2

    distance_sq = distance[...,0]**2 + distance[...,1]**2 # B x N x (HxW)
    pts_scale = pts_scale.clamp(min=0.01)
    weight = torch.exp(-distance_sq / (pts_scale.view(bs,-1,1) * self.sigma * self.sigma))# B x N x (HxW)
    weight = weight * points_mask.view(bs,-1,1).float()
    # concatenate the depth as an additional channel
    # TODO: discuss other options instead of soft projection of depth
    # pts_feat = torch.cat((pts_feat, points_proj[:,2:3,:].view(bs,-1,1)), dim=2)

    # sum up features from all 3d points for each grid point
    img = pts_feat @ weight #  (B x C x N) x (B x N x (HxW)) --> B x C x (HxW)

    marker = pts_marker @ weight #  (B x 3 x N) x (B x N x (HxW)) --> B x C x (HxW)

    img = img.view(bs, -1, self.im_height, self.im_width)
    marker = marker.view(bs, -1, self.im_height, self.im_width)

    return img, marker

  def forward(self, RT, center_3d, scale, pts_3d, pts_feat, pts_scale, pts_maker):
    # params:
    #   RT: Bx3x4
    #   pts_3d: Bx3xN
    #   pts_feat: BxCxN
    #   pts_scale: BxN
    return self.forward_batch(RT, center_3d, scale, pts_3d, pts_feat, pts_scale, pts_maker)
