# The gridsample of our model.

import math
import torch
import torch.nn as nn
from liga.utils import calibration_kitti
from tools.point_sample import bilinear_grid_sample_test
import torch.nn.functional as F

def project_pseudo_lidar_to_rectcam(pts_3d):
    xs, ys, zs = pts_3d[..., 0], pts_3d[..., 1], pts_3d[..., 2]
    return torch.stack([-ys, -zs, xs], dim=-1)

def project_rect_to_image(pts_3d_rect, P):
    n = pts_3d_rect.shape[0]
    ones = torch.ones((n, 1), device=pts_3d_rect.device)
    pts_3d_rect = torch.cat([pts_3d_rect, ones], dim=1)
    pts_2d = torch.mm(pts_3d_rect, torch.transpose(P, 0, 1))  # nx3   
    #pts_2d[:, 0] /= pts_2d[:, 2]
    #pts_2d[:, 1] /= pts_2d[:, 2] # changed by me
    pts_2d = torch.stack((pts_2d[:,0]/pts_2d[:, 2], pts_2d[:,1]/pts_2d[:, 2], pts_2d[:,2]),dim=1)
    pts_2d = torch.where(torch.abs(pts_2d) > 10000, torch.tensor(-1., device = pts_2d.device), pts_2d.clone().detach())
    return pts_2d[:, 0:3]

class GridSampleSimple(nn.Module):
    def __init__(self, model_cfg, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.prepare_coordinates_3d(point_cloud_range, voxel_size, grid_size)
        self.init_params()


    def prepare_coordinates_3d(self, point_cloud_range, voxel_size, grid_size, sample_rate=(1, 1, 1)):
        self.X_MIN, self.Y_MIN, self.Z_MIN = point_cloud_range[:3]
        self.X_MAX, self.Y_MAX, self.Z_MAX = point_cloud_range[3:]
        self.VOXEL_X_SIZE, self.VOXEL_Y_SIZE, self.VOXEL_Z_SIZE = voxel_size
        self.GRID_X_SIZE, self.GRID_Y_SIZE, self.GRID_Z_SIZE = grid_size.tolist()

        self.CV_DEPTH_MIN = point_cloud_range[0]
        self.CV_DEPTH_MAX = point_cloud_range[3]

        self.VOXEL_X_SIZE /= sample_rate[0]
        self.VOXEL_Y_SIZE /= sample_rate[1]
        self.VOXEL_Z_SIZE /= sample_rate[2]

        self.GRID_X_SIZE *= sample_rate[0]
        self.GRID_Y_SIZE *= sample_rate[1]
        self.GRID_Z_SIZE *= sample_rate[2]

        zs = torch.linspace(self.Z_MIN + self.VOXEL_Z_SIZE / 2., self.Z_MAX - self.VOXEL_Z_SIZE / 2.,
                            self.GRID_Z_SIZE, dtype=torch.float32)
        ys = torch.linspace(self.Y_MIN + self.VOXEL_Y_SIZE / 2., self.Y_MAX - self.VOXEL_Y_SIZE / 2.,
                            self.GRID_Y_SIZE, dtype=torch.float32)
        xs = torch.linspace(self.X_MIN + self.VOXEL_X_SIZE / 2., self.X_MAX - self.VOXEL_X_SIZE / 2.,
                            self.GRID_X_SIZE, dtype=torch.float32)
        zs, ys, xs = torch.meshgrid(zs, ys, xs)
        coordinates_3d = torch.stack([xs, ys, zs], dim=-1)
        self.coordinates_3d = coordinates_3d.float()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[
                    2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, batch_dict):

        calib = batch_dict['calib']
        # calib = [calibration_kitti.Calibration("data/zjdata/training/calib/0823_001000.txt")]
        N = batch_dict['batch_size']

        coordinates_3d = self.coordinates_3d.cuda()
        batch_dict['coord'] = coordinates_3d
        norm_coord_imgs = {}
        coord_imgs = {}
        for image_id in batch_dict['images_id']:
            image_id = str(image_id.item())
            coord_imgs['coord_img' + image_id] = []
            norm_coord_imgs['norm_coord_img' + image_id] = []

        for i in range(N):
            c3d = coordinates_3d.view(-1, 3)

            if 'random_T' in batch_dict:
                random_T = batch_dict['random_T'][i]
                c3d = torch.matmul(c3d, random_T[:3, :3].T) + random_T[:3, 3]
            # in pseudo lidar coord
            c3d = project_pseudo_lidar_to_rectcam(c3d)

            for image_id in batch_dict['images_id']:
                image_id = str(image_id.item())
                image_name = 'image_' + image_id
                if type(calib) == dict:
                    calibs_P = calib['P' + image_id]
                    calibs_P = torch.as_tensor(calibs_P, dtype=torch.float32, device=batch_dict[image_name].device)
                    calibs_R = calib['R' + image_id]
                    calibs_R = torch.as_tensor(calibs_R, dtype=torch.float32, device=batch_dict[image_name].device)
                    calibs_T = calib['T' + image_id]
                    calibs_T = torch.as_tensor(calibs_T, dtype=torch.float32, device=batch_dict[image_name].device)
                else:
                    calibs_P = getattr(calib[i], 'P' + image_id)
                    calibs_P = torch.as_tensor(calibs_P, dtype=torch.float32, device=batch_dict[image_name].device)
                    calibs_R = getattr(calib[i], 'R' + image_id)
                    calibs_R = torch.as_tensor(calibs_R, dtype=torch.float32, device=batch_dict[image_name].device)
                    calibs_T = getattr(calib[i], 'T' + image_id)
                    calibs_T = torch.as_tensor(calibs_T, dtype=torch.float32, device=batch_dict[image_name].device)
                
                c3d_image =  torch.mm(
                    torch.cat((c3d,torch.ones((c3d.shape[0],1),device=c3d.device)), dim=1), \
                    torch.transpose(torch.cat((calibs_R, calibs_T.unsqueeze(1)), dim=1),1,0)
                )
                coord_img = project_rect_to_image(c3d_image, calibs_P.float().cuda())
                # coord_img = torch.cat([coord_img, c3d[..., 2:]], dim=-1)
                coord_img = coord_img.view(*self.coordinates_3d.shape[:3], 3)
                coord_imgs['coord_img' + image_id].append(coord_img)

                crop_x1, crop_x2 = 0, batch_dict[image_name].shape[3]
                crop_y1, crop_y2 = 0, batch_dict[image_name].shape[2]
                norm_coord_img = \
                    (coord_img - torch.as_tensor([crop_x1, crop_y1, self.CV_DEPTH_MIN],  device=coord_img.device)) /\
                    torch.as_tensor(
                        [crop_x2 - 1 - crop_x1, crop_y2 - 1 - crop_y1, self.CV_DEPTH_MAX - self.CV_DEPTH_MIN],\
                        device=coord_img.device)
                norm_coord_img = norm_coord_img * 2. - 1.
                norm_coord_imgs['norm_coord_img' + image_id].append(norm_coord_img)       

        Voxels = []
        for image_id in batch_dict['images_id']:
            image_id = str(image_id.item())
            norm_coord_imgs['norm_coord_img' + image_id] = torch.stack(norm_coord_imgs['norm_coord_img' + image_id], dim=0)
            coord_imgs['coord_img' + image_id] = torch.stack(coord_imgs['coord_img' + image_id], dim=0)
            norm_coord_imgs_2d = norm_coord_imgs['norm_coord_img' + image_id].clone().detach()
            norm_coord_imgs_2d = torch.cat((norm_coord_imgs_2d[...,:2],torch.zeros(N, 20, 304, 388, 1).cuda()), dim=-1)
            stereo_feat = batch_dict['stereo_feat_' + image_id]
            Voxels.append(F.grid_sample(stereo_feat.unsqueeze(2), norm_coord_imgs_2d, align_corners=True))
            # Voxels.append(bilinear_grid_sample_test(stereo_feat.unsqueeze(2), norm_coord_imgs_2d, align_corners=True))
            # Voxels.append(bilinear_grid_sample_test(stereo_feat.unsqueeze(2)*depth.unsqueeze(1), norm_coord_imgs_2d, align_corners=True))

        # Voxel = torch.cat(Voxels, dim=1)
        Voxel = torch.stack(Voxels, dim=1)
        batch_dict['voxel_feature'] = Voxel
        
        return batch_dict
