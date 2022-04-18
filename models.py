import torch.nn as nn
import torch
from torch.nn import init
from torch.autograd import Variable
import latent_projection
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

##############################
#       pix2pix U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [  nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
                    nn.InstanceNorm2d(out_size),
                    nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x



class UNetUpNoSkip(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUpNoSkip, self).__init__()
        layers = [  nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
                    nn.InstanceNorm2d(out_size),
                    nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)

        return x


##############################
#       latent3D U-NET
##############################
class unetDownRohbin(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, padding = 1):
        super(unetDownRohbin, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, padding),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, padding),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(),)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, padding),
                                       nn.ReLU(),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, padding),
                                       nn.ReLU(),)
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)

        return outputs


class unetUpRohbin(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, padding=1):
        super(unetUpRohbin, self).__init__()
        self.conv = unetDownRohbin(in_size, out_size, False, padding)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                             nn.Conv2d(in_size, out_size, 3, stride=1, padding=1),
                             nn.BatchNorm2d(out_size),
                             nn.ReLU()
                             )

    def forward(self, x, skip_input):
        x = self.up(x)
        x = torch.cat((x, skip_input), 1)
        return self.conv(x)


class unetUpNoSKipRohbin(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, padding):
        super(unetUpNoSKipRohbin, self).__init__()
        self.conv = unetDownRohbin(out_size, out_size, False, padding) # note, changed to out_size, out_size for no skip
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            #self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            self.up = nn.Sequential(
                             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                             nn.Conv2d(in_size, out_size, 3, stride=1, padding=1),
                             nn.BatchNorm2d(out_size),
                             nn.ReLU()
                             )

    def forward(self, inputs2):
        outputs2 = self.up(inputs2)
        return self.conv(outputs2)


##############################
#       Encoder - Decoder
##############################
class pix2pixEncoder(nn.Module):
    def __init__(self, in_channels=6, out_channels=128):
        super(pix2pixEncoder, self).__init__()
        self.down1 = UNetDown(in_channels, 32, True)
        self.down2 = UNetDown(32, 64, True)
        self.down3 = UNetDown(64, out_channels, True)
#        self.down4 = UNetDown(128, out_channels, True)


    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        #d4 = self.down4(d3)
        return d3 # 256*16*16


class rohbinEncoder(nn.Module):
    def __init__(self, in_channels=6, out_channels=256):
        super(rohbinEncoder, self).__init__()
        self.down1 = unetDownRohbin(in_channels, 32, True, padding=1)
        self.down2 = unetDownRohbin(32, 64, True, padding=1)
        self.down3 = unetDownRohbin(64, 128, True, padding=1)
        self.down4 = unetDownRohbin(128, out_channels, True, padding=1)
        self.pooling = nn.MaxPool2d(kernel_size=2)


    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d1 = self.pooling(d1)
        d2 = self.down2(d1)
        d2 = self.pooling(d2)
        d3 = self.down3(d2)
        d3 = self.pooling(d3)
        d4 = self.down4(d3)
        return d4 # 256*16*16


class rohbinDecoder(nn.Module):
    def __init__(self, in_channels=256, out_channels=3):
        super(rohbinDecoder, self).__init__()

        #self.combine = latentCombine(num_3d_features=600,num_output_features=32768)
        self.up1 = unetUpNoSKipRohbin(int(in_channels), int(in_channels/2), False, padding=1)
        self.up2 = unetUpNoSKipRohbin(int(in_channels/2), int(in_channels/4), False, padding=1)
        self.up3 = unetUpNoSKipRohbin(int(in_channels/4), int(in_channels/8), False, padding=1)
        self.final = nn.Conv2d(int(in_channels/8), out_channels, 1)


    def forward(self, d4): #256*16*16= 65536
        d3 = self.up1(d4)
        d2 = self.up2(d3)
        d1 = self.up3(d2)
        return self.final(d1)

##############################
#      latent transformation
##############################
class latentParitionLinear(nn.Module): # linear change
    def __init__(self, num_input_features, num_fg_features, num_3d_features):
        super(latentParitionLinear,self).__init__()
        self.to_fg = nn.Sequential(nn.Linear(num_input_features, num_fg_features),
                                   #nn.Dropout(inplace=True, p=0.3),
                                   nn.Sigmoid(),
                                   )
        self.to_3d = nn.Sequential(nn.Linear(num_input_features, num_3d_features),
                                   #nn.Dropout(inplace=True, p=0.3)  # removing dropout degrades results
                                   nn.Sigmoid(),
                                   )
        self.to_fg_conv = nn.Sequential(nn.Conv2d(256, 3, 1),
                                        nn.Dropout(inplace=True, p=0.3),
                                        nn.ReLU(inplace=False))
        self.to_3d_size = nn.Sequential(nn.Linear(num_input_features, out_features=int(num_3d_features/3)),
                                        nn.Sigmoid(),
                                               #nn.Dropout(inplace=True, p=0.3)  # removing dropout degrades results
                                               )

    def forward(self, input_x, max_3d_scale, min_3d_scale): #65536->128,3*200

        center_flat = input_x.view(input_x.size()[0], -1)
        #latent_fg ranges from (0,100)

        latent_fg = self.to_fg(center_flat) * 10.
        latent_3d = self.to_3d(center_flat)
        latent_3d = latent_3d.view(latent_3d.size()[0], -1, 3)

        #set scale of camera points
        latent_3d_scale = torch.ones_like(latent_3d)
        max_3d_scale = torch.unsqueeze(max_3d_scale,dim=1).expand(latent_3d.size())
        min_3d_scale = torch.unsqueeze(min_3d_scale,dim=1).expand(latent_3d.size())
        latent_3d_scale[:,:,:2] = max_3d_scale[:,:,:2] - min_3d_scale[:,:,:2]
        latent_3d_scale[:,:,2] = 5

        latent_3d = latent_3d * latent_3d_scale

        latent_3d_offset = min_3d_scale
        latent_3d_offset[:,:,2] = 0.5
        latent_3d += latent_3d_offset
        latent_3d_size = self.to_3d_size(center_flat)
        return latent_fg, latent_3d, latent_3d_size


class latentParitionconv(nn.Module): # conv change
    def __init__(self, num_input_channels, num_fg_channels=3, num_3d_channels=3):
        super(latentParitionconv,self).__init__()
        self.to_fg = nn.Sequential(nn.Conv2d(num_input_channels, num_fg_channels,1),
                                   nn.Dropout(inplace=True, p=0.3),
                                   nn.ReLU(inplace=False))
        self.to_3d = nn.Sequential(nn.Conv2d(num_input_channels, num_3d_channels,1),
                                   nn.Dropout(inplace=True, p=0.3)  # removing dropout degrades results

                                   )

    def forward(self, x): #256*16*16 -> 3*16*16
        latent_fg = self.to_fg(x)
        latent_3d = self.to_3d(x)
        return latent_fg, latent_3d


class latentCombineLinear(nn.Module):
    def __init__(self, num_3d_features, num_output_features):
        super(latentCombineLinear,self).__init__()
        self.from_3d = nn.Sequential(nn.Linear(num_3d_features, num_output_features),
                                     nn.Dropout(inplace=True, p=0.3),
                                     nn.ReLU(inplace=False))

    def forward(self, x_fg, x_3d, bottleneck_width):
        x_fg = x_fg.view(x_fg.size()[0], x_fg.size()[1], 1, 1)
        x_fg = x_fg.expand(x_fg.size()[0], x_fg.size()[1], bottleneck_width, bottleneck_width)

        x_3d = self.from_3d(x_3d)
        x_3d = x_3d.view(x_3d.size()[0], -1, bottleneck_width, bottleneck_width)
        output = torch.cat((x_fg, x_3d), 1) # the same as unet input
        return output


class latentCombineconv(nn.Module):
    def  __init__(self, num_channels=3, num_output_channels=128, bottle_neck=16):
        super(latentCombineconv,self).__init__()
        self.bottle_neck = bottle_neck
        self.from_3d = nn.Sequential(nn.Conv2d(num_channels, num_output_channels, 3, 1, 1),
                                     nn.Dropout(inplace=True, p=0.3),
                                     nn.ReLU(inplace=False))

        self.from_fg = nn.Sequential(nn.Conv2d(num_channels, num_output_channels, 3, 1, 1),
                                     nn.Dropout(inplace=True, p=0.3),
                                     nn.ReLU(inplace=False))

    def forward(self, x_fg, x_3d):
        x_fg = self.from_fg(x_fg)

        x_3d = x_3d.transpose(1,2)
        x_3d = x_3d.view(x_3d.size()[0], x_3d.size()[1], self.bottle_neck, self.bottle_neck)
        x_3d = self.from_3d(x_3d)
        output = torch.cat((x_fg, x_3d), 1) # the same as unet input
        return output




##############################
#          model
##############################
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

    trans_a2b = trans_b.transpose(1,2) - trans_a2b
    return rotation_a2b, trans_a2b

def camera2world(batch_size, extrinsic_a):
    world2cam_rot_a = extrinsic_a[:, :, :3]
    world2cam_trans_a = extrinsic_a[:, :, -1]
    cam2world_rot_a = world2cam_rot_a.transpose(1,2)
    # determine crop1 to crop2
    world2cam_trans_a = world2cam_trans_a.view((batch_size, 1, 3)).float()
    cam2world_trans_a = -torch.bmm(cam2world_rot_a, world2cam_trans_a.transpose(1, 2))
#    cam2world = torch.cat((cam2world_rot_a,cam2world_trans_a),dim=2)
    return cam2world_rot_a,cam2world_trans_a



class encode3Dmodel(nn.Module):
    def __init__(self, intrinsic, img_height=300, img_width=400, input_channels=3, hidden_channels = 256, output_channels=3):
        super(encode3Dmodel, self).__init__()
        self.point_num = 12
        self.fg_feature_num =3
        self.img_height = img_height
        self.img_width = img_width
        self.intrinsic = intrinsic
        self.input_channels = input_channels
        self.bottleneck_features = int(hidden_channels*img_width*img_height/64)

        self.encoder = rohbinEncoder(in_channels=input_channels, out_channels=hidden_channels)
        self.partition = latentParitionLinear(num_input_features=self.bottleneck_features, num_fg_features=(self.point_num*self.fg_feature_num), num_3d_features=int(3*self.point_num))
        self.combine = latentCombineconv(3, int(hidden_channels/2), bottle_neck=8)
        self.decoder = rohbinDecoder(in_channels=hidden_channels, out_channels=output_channels)

    def forward(self, x, extrinsic, phase, output_types):

        #####################################
        #         Reading input data and RT's
        #####################################
        batch_size = x.size()[0]
        crop_a = x[:, :self.input_channels, :, :]
        crop_b = x[:, self.input_channels:, :, :]
        extrinsic_a = extrinsic[:, :3, :]
        extrinsic_b = extrinsic[:, 3:, :]
        #####################################
        #     constrain scale of 3d points
        #####################################
        inv_intrinsic = torch.inverse(self.intrinsic)
        uvz_corner = torch.tensor([[1., 1., 1.], [self.img_width, self.img_height, 1.]])
        uvz_corner = uvz_corner.type(extrinsic_a.dtype).to(extrinsic_a.device)
        uvz_corner = uvz_corner.transpose(0, 1)
        uvz_corner = torch.mm(inv_intrinsic, uvz_corner)
        uv_scale_max = uvz_corner[:, 1]
        uv_scale_min = uvz_corner[:, 0]
        uv_scale_max = torch.unsqueeze(uv_scale_max,dim=0).expand(batch_size,3)
        uv_scale_min = torch.unsqueeze(uv_scale_min,dim=0).expand(batch_size,3)
        scale_max = 4.0 * uv_scale_max #Bx3
        scale_min = 4.0 * uv_scale_min #Bx3
        #####################
        #        encoder
        #####################
        encoded_a = self.encoder(crop_a)
        encoded_b = self.encoder(crop_b)
        #####################
        #        parition
        #####################
        latent_fg_a, latent_3d_a, latent_3d_a_size = self.partition(encoded_a, scale_max, scale_min)
        latent_fg_b, latent_3d_b, latent_3d_b_size = self.partition(encoded_b, scale_max, scale_min)
        latent_3d_a = latent_3d_a.transpose(1,2) #bsx3*n
        latent_3d_b = latent_3d_b.transpose(1,2)
        rot_a2b, trans_a2b = relativeRT(batch_size, extrinsic_a, extrinsic_b)
        rot_b2a, trans_b2a = relativeRT(batch_size, extrinsic_b, extrinsic_a)
        transform_a2b = torch.cat((rot_a2b,trans_a2b), dim=-1)
        transform_b2a = torch.cat((rot_b2a,trans_b2a), dim=-1)
        latent_3d_a2b = torch.bmm(rot_a2b, latent_3d_a) + trans_a2b.expand(latent_3d_a.size())
        latent_3d_b2a = torch.bmm(rot_b2a, latent_3d_b) + trans_b2a.expand(latent_3d_b.size())

        latent_fe_a = self.combine(latent_fg_b, latent_3d_b2a)
        latent_fe_b = self.combine(latent_fg_a, latent_3d_a2b)

        output_a = self.decoder(latent_fe_a)
        output_b = self.decoder(latent_fe_b)
        output = torch.cat((output_a, output_b), dim=1)
        ###############################################
        # Select the right output
        output_dict_all = {'output': output, 'latent_3d_a': latent_3d_a, 'latent_3d_b': latent_3d_b,
                           'latent_3d_a2b': latent_3d_a2b, 'latent_3d_b2a': latent_3d_b2a,
                           'latent_fg_a': latent_fg_a, 'latent_fg_b': latent_fg_b,
                          }
        output_dict = {}
        for key in output_types:
            output_dict[key] = output_dict_all[key]
        return output_dict


class rendermodel(nn.Module):
    def __init__(self, intrinsic, img_height=300, img_width=400, input_channels=3, hidden_channels = 256, output_channels=3):
        super(rendermodel, self).__init__()
        self.regu_dis = 8
        self.point_num = 12
        self.fg_feature_num =8
        self.img_height = img_height
        self.img_width = img_width
        self.intrinsic = intrinsic
        self.input_channels = input_channels
        # self.bottleneck_features = int(hidden_channels*img_width*img_height/64)
        #self.bottleneck_features = int(473600)
        self.bottleneck_features = int(6144)

        self.encoder = rohbinEncoder(in_channels=input_channels, out_channels=hidden_channels)
        self.partition = latentParitionLinear(num_input_features=self.bottleneck_features, num_fg_features=(self.point_num*self.fg_feature_num), num_3d_features=int(3*self.point_num))
        self.latent_img_renderer = latent_projection.ProjectPoint2ImageSoft(K=intrinsic)
        self.render_encoder = rohbinEncoder(in_channels=self.fg_feature_num, out_channels=hidden_channels)
        self.decoder = rohbinDecoder(in_channels=hidden_channels, out_channels=output_channels)

    #project pts_3d to camera uvs with given externalrt and intrinsic matrix
    def projector(self, externalrt, pts_3d):
        bs = pts_3d.shape[0]
        if externalrt.shape[0] == 1:
            points_local = pts_3d
        else:
            R = externalrt[:, :3, :3]
            T = externalrt[:, :3, 3]
            # transform points from world coordinate to camera coordinate
            points_local = torch.bmm(R, pts_3d) + T.view(bs, 3, 1)  # Bx3xN
        # perspective projection
        k_expand = torch.unsqueeze(self.intrinsic, dim=0).expand(points_local.shape[0], 3, 3)
        # points_proj = self.K.unsqueeze(0) @ points_local # Bx3xN
        points_proj = torch.bmm(k_expand, points_local)  # Bx3xN

        points_mask = points_proj[:, 2] > 0.1  # BxN
        u = points_proj[:, 0, :] / points_proj[:, 2, :].clamp(min=0.1)
        v = points_proj[:, 1, :] / points_proj[:, 2, :].clamp(min=0.1)
        u = u * points_mask.float()
        v = v * points_mask.float()
        uv = torch.cat((v.reshape(bs, -1, 1), u.reshape(bs, -1, 1)), dim=2)  # BxNx2
        return uv

    def getUVdistance(self, batch_size, latent_3d_a, latent_3d_b,transform_b2a):
        uv_a = self.projector(torch.tensor([0]), latent_3d_a)#BxNx2
        uv_b2a = self.projector(transform_b2a,latent_3d_b)#BxNx2
        uv_distance = uv_a.view(batch_size, 1, -1, 2) - uv_b2a.view(batch_size, -1, 1, 2)  # BxNxNx2
        uv_distance = uv_distance[..., 0] ** 2 + uv_distance[..., 1] ** 2  # B xN*N
        return  uv_distance

    def getvalid_mask(self, batch_size, uv_distance):
        constrain_neibor = uv_distance < self.regu_dis ** 2  # B x N*N
        constrained_uv_distance_sq = uv_distance * constrain_neibor.float()  # B xNxN
        max_dis, no_use_index = torch.max(constrained_uv_distance_sq, dim=-1)  # B x N
        valid_a = max_dis > 0  # BxN (index of point a)
        valid_a = valid_a.float()
        num_pairs = torch.sum(valid_a, dim=-1)  # B
        index_intersected_samples = num_pairs > 0  # B, if exists non-overlapping samples
        num_samples = torch.sum(index_intersected_samples)  # 1 value torch<B
        if num_samples == 0:
            # print("no overlap with the defined uv_range!")
            flag = 0
            mask = []
            index_intersected_trainable_samples = []
            num_pairs = []
        else:
            min_dis, index_b = torch.min(uv_distance, dim=-1)  # B x N
            valid_b = min_dis > 0
            valid_b = valid_b.float()
            num_trainable_pairs = torch.sum(valid_b, dim=-1)  # B
            index_trainable_samples = num_trainable_pairs > 0  # B

            index_intersected_trainable_samples = index_intersected_samples * index_trainable_samples
            num_pairs = num_pairs[index_intersected_trainable_samples]
            valid_a = torch.unsqueeze(valid_a, dim=-1)
            valid_a = valid_a.expand(batch_size, self.point_num, self.point_num)  # BxNxN
            min_dis = torch.unsqueeze(min_dis, dim=-1)
            min_dis = min_dis.expand(batch_size, self.point_num, self.point_num)
            valid_b = uv_distance == min_dis
            valid_b = valid_b.float()
            mask = torch.ones_like(uv_distance)
            mask = mask * valid_a * valid_b  # BxNxN
            flag = 1
        return flag, mask, index_intersected_trainable_samples, num_pairs

    def projection_error(self,batch_size,uv_distance_b2a, uv_distance_a2b, geo_distance,fg_distance):

        flag1, mask1, trainable_samples1, num_pairs1 = self.getvalid_mask(batch_size, uv_distance_b2a)
        flag2, mask2, trainable_samples2, num_pairs2 = self.getvalid_mask(batch_size, uv_distance_a2b)
        if not flag1 or not flag2:
            mse_uv = []
            mse_depth = []
            mse_fg = []
        else:
            mask = mask1 * mask2
            #trainable_samples = trainable_samples1 * trainable_samples2 # not applicable, as the same sample can have different pairs of points
            #num_pairs = num_pairs1 * num_pairs2
            num_pairs = torch.sum(mask, dim=-1)  # BxN
            num_pairs = torch.sum(num_pairs,dim=-1)
            trainable_samples = num_pairs > 0
            num_pairs = num_pairs[trainable_samples]
            num_samples = torch.sum(trainable_samples)  # 1 value torch<B
            if num_samples == 0:
                mse_uv = []
                mse_depth = []
                mse_fg = []
            else:
                valid_uv_distance = uv_distance_b2a * mask
                valid_geo_distance = geo_distance * mask
                valid_fg_distance = fg_distance * mask

                valid_uv_distance = valid_uv_distance[trainable_samples]
                valid_geo_distance = valid_geo_distance[trainable_samples]
                valid_fg_distance = valid_fg_distance[trainable_samples]
                mse_uv = torch.mean(torch.sum(valid_uv_distance.view(num_samples, -1), dim=-1) / num_pairs)
                mse_depth = torch.mean(torch.sum(valid_geo_distance.view(num_samples, -1), dim=-1) / num_pairs)
                mse_fg = torch.mean(torch.sum(valid_fg_distance.view(num_samples, -1), dim=-1) / num_pairs)

        return mse_uv, mse_depth, mse_fg

    def forward(self, x, extrinsic, center_world, scale, phase, output_types):

        ###########################################
        #         Reading input data and RT's
        ###########################################
        batch_size = x.size()[0]
        crop_a = x[:, :self.input_channels, :, :]
        crop_b = x[:, self.input_channels:, :, :]
        extrinsic_a = extrinsic[:, :3, :]
        extrinsic_b = extrinsic[:, 3:, :]
        scale_a = scale[:, 0, :]
        scale_b = scale[:, 1, :]
        # use identity R and zero T as we want to predict the 3d features in the camera coordinate
        identity = torch.cat((torch.eye(3), torch.zeros(3,1)), dim=1)
        identity = identity.view(1,3,4).repeat(extrinsic_a.shape[0],1,1)
        identity = identity.type(extrinsic_a.dtype).to(extrinsic_a.device)
        ######################################################
        #     constrain scale of 3d points
        ######################################################
        inv_intrinsic = torch.inverse(self.intrinsic)
        uvz_corner = torch.tensor([[1., 1., 1.], [self.img_width, self.img_height, 1.]])
        uvz_corner = uvz_corner.type(extrinsic_a.dtype).to(extrinsic_a.device)
        uvz_corner = uvz_corner.transpose(0, 1)
        uvz_corner = torch.mm(inv_intrinsic, uvz_corner)
        uv_scale_max = uvz_corner[:, 1]
        uv_scale_min = uvz_corner[:, 0]
        uv_scale_max = torch.unsqueeze(uv_scale_max,dim=0).expand(batch_size,3)
        uv_scale_min = torch.unsqueeze(uv_scale_min,dim=0).expand(batch_size,3)
        scale_max = 4.0 * uv_scale_max #Bx3
        scale_min = 4.0 * uv_scale_min #Bx3
        #####################
        #        encoder
        #####################
        encoded_a = self.encoder(crop_a)
        encoded_b = self.encoder(crop_b)
        ##############################
        #        parition
        ##############################
        latent_fg_a, latent_3d_a_render, latent_3d_a_size = self.partition(encoded_a, scale_max, scale_min)
        latent_fg_b, latent_3d_b_render, latent_3d_b_size = self.partition(encoded_b, scale_max, scale_min)
        latent_3d_a = torch.ones_like(latent_3d_a_render)
        latent_3d_b = torch.ones_like(latent_3d_b_render)
        # move to full field of view coordinate
        scale_a_expand = torch.unsqueeze(scale_a, dim=1).expand(scale_a.shape[0], self.point_num, scale_a.shape[1])
        scale_a_expand = scale_a_expand.type(latent_3d_a.dtype).to(latent_3d_a.device)
        latent_3d_a[:, :, 0] = latent_3d_a_render[:, :, 0] * scale_a_expand[:, :, 0]
        latent_3d_a[:, :, 1] = latent_3d_a_render[:, :, 1] * scale_a_expand[:, :, 1]
        latent_3d_a[:, :, 2] = latent_3d_a_render[:, :, 2]
        scale_b_expand = torch.unsqueeze(scale_b, dim=1).expand(scale_b.shape[0], self.point_num, scale_b.shape[1])
        scale_b_expand = scale_b_expand.type(latent_3d_b.dtype).to(latent_3d_b.device)
        latent_3d_b[:, :, 0] = latent_3d_b_render[:, :, 0] * scale_b_expand[:, :, 0]
        latent_3d_b[:, :, 1] = latent_3d_b_render[:, :, 1] * scale_b_expand[:, :, 1]
        latent_3d_b[:, :, 2] = latent_3d_b_render[:, :, 2]

        latent_3d_a = latent_3d_a.transpose(1,2) #bsx3*n
        latent_3d_b = latent_3d_b.transpose(1,2)
        center_world = torch.unsqueeze(center_world,dim=2)
        center_world = center_world.type(extrinsic_a.dtype).to(extrinsic_a.device)
        #compute transformation matrix
        rot_a2b, trans_a2b = relativeRT(batch_size, extrinsic_a, extrinsic_b)
        rot_b2a, trans_b2a = relativeRT(batch_size, extrinsic_b, extrinsic_a)
        transform_a2b = torch.cat((rot_a2b,trans_a2b), dim=-1)
        transform_b2a = torch.cat((rot_b2a,trans_b2a), dim=-1)
        latent_3d_a2b = torch.bmm(rot_a2b, latent_3d_a) + trans_a2b.expand(latent_3d_a.size())
        latent_3d_b2a = torch.bmm(rot_b2a, latent_3d_b) + trans_b2a.expand(latent_3d_b.size())

        latent_fg_a_flattened = latent_fg_a.view(batch_size, self.fg_feature_num, -1)#bs*c*n
        latent_fg_b_flattened = latent_fg_b.view(batch_size, self.fg_feature_num, -1)#bs*c*n
        #####################
        #       applyRT
        #####################
        cam2world_rot_a, cam2world_trans_a = camera2world(batch_size,extrinsic_a)
        cam2world_rot_b, cam2world_trans_b = camera2world(batch_size,extrinsic_b)


        # use a renderer model, project key points on to new image
        # latent_3d_a/b is in render-camera coordinate, should move to real camera coordinate
        # here, we suppose render-camera is approximately similar in position with vehicles, and infrastructure is far from observations
        Ra = extrinsic_a[:, :3, :3]
        Ta = extrinsic_a[:, :3, 3]
        Ta = Ta.view(Ta.shape[0],3,1)
        center_3d_a = torch.bmm(Ra,center_world) + Ta.expand(center_world.size())
        Rb = extrinsic_b[:, :3, :3]
        Tb = extrinsic_b[:, :3, 3]
        Tb = Tb.view(Tb.shape[0], 3, 1)
        center_3d_b = torch.bmm(Rb, center_world) + Tb.expand(center_world.size())
        #move rendered small model to real vehicle position
        latent_3d_a = latent_3d_a + center_3d_a.expand(latent_3d_a.size())
        latent_3d_b = latent_3d_b + center_3d_b.expand(latent_3d_b.size())
        latent_3d_a_world = torch.bmm(cam2world_rot_a, latent_3d_a) + cam2world_trans_a.expand(latent_3d_a.size())
        latent_3d_b_world = torch.bmm(cam2world_rot_b, latent_3d_b) + cam2world_trans_b.expand(latent_3d_b.size())
        # select the top 3 channel of features bs*3*n
        latent_fg_marker_a, _ = torch.topk(latent_fg_a_flattened, 3, dim=1)
        latent_fg_marker_b, _ = torch.topk(latent_fg_b_flattened, 3, dim=1)
        ########### project points onto new images###########
        #use extrinsic matrics
        latent_img_a, marker_a = self.latent_img_renderer(extrinsic_a, center_3d_a, scale_a_expand, latent_3d_b_world, latent_fg_b_flattened,
                                                          latent_3d_b_size, latent_fg_marker_b)
        latent_img_b, marker_b = self.latent_img_renderer(extrinsic_b, center_3d_b, scale_b_expand, latent_3d_a_world, latent_fg_a_flattened,
                                                          latent_3d_a_size, latent_fg_marker_a)
        #only use intrinsic matrics
        # latent_img_a, marker_a = self.latent_img_renderer(identity, latent_3d_a, latent_fg_a_flattened,
        #                                                   latent_3d_a_size, latent_fg_marker_a)
        # latent_img_b, marker_b = self.latent_img_renderer(identity, latent_3d_b, latent_fg_b_flattened,
        #                                                   latent_3d_b_size, latent_fg_marker_b)
        latent_fe_a = self.render_encoder(latent_img_a)
        latent_fe_b = self.render_encoder(latent_img_b)
        # marker for visualization
        render_marker = torch.cat((marker_a, marker_b), dim=-2)

        #### estimate regularizers
        fg_distance = latent_fg_a_flattened.transpose(1, 2).view(batch_size, 1, self.point_num, self.fg_feature_num) - \
              latent_fg_b_flattened.transpose(1, 2).view(batch_size, -1, self.point_num,
                                                         self.fg_feature_num)  # BxNxNxC
        fg_distance = fg_distance ** 2  # BxNxNxC squred features
        fg_distance = torch.sum(fg_distance, dim=-1)  # BxNxN sumed squred features
        geo_distance = latent_3d_a.transpose(1, 2).view(batch_size, 1, self.point_num, 3) - \
                       latent_3d_b.transpose(1, 2).view(batch_size, self.point_num, 1, 3)
        geo_distance = geo_distance ** 2
        geo_distance = geo_distance[..., -1]  # BxNxN

        uv_distance_b2a = self.getUVdistance(batch_size, latent_3d_a, latent_3d_b,transform_b2a)
        uv_distance_a2b = self.getUVdistance(batch_size, latent_3d_b, latent_3d_a,transform_a2b)

        mse_uv, mse_depth, mse_fg = self.projection_error(batch_size, uv_distance_b2a, uv_distance_a2b, geo_distance, fg_distance)



        output_a = self.decoder(latent_fe_a)
        output_b = self.decoder(latent_fe_b)
        output = torch.cat((output_a, output_b), dim=1)
        ###############################################
        # Select the right output
        output_dict_all = {'output': output, 'latent_3d_a': latent_3d_a, 'latent_3d_b': latent_3d_b,'latent_3d_a2b': latent_3d_a2b,
                           'latent_3d_b2a': latent_3d_b2a, 'latent_fg_a': latent_fg_a_flattened, 'latent_fg_b': latent_fg_b_flattened,
                           'render_marker': render_marker,'mse_uv':mse_uv, 'mse_depth':mse_depth,'mse_fg':mse_fg}
        output_dict = {}
        for key in output_types:
            output_dict[key] = output_dict_all[key]
        return output_dict

























