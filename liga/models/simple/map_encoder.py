import sys

if '/home/zjlab/wsq/liga_test' in sys.path:
    sys.path.remove('/home/zjlab/wsq/liga_test')
sys.path.append('/home/zjlab/psl/liga_test')

import numpy as np
import torch
import torch.nn as nn
from spconv.pytorch import SparseConv3d, SubMConv3d, SparseSequential, SparseConvTensor
from mmcv.cnn import build_conv_layer, build_norm_layer
import logging
from mmcv.runner import load_checkpoint

class SpMiddleFHD(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="SpMiddleFHD", **kwargs
    ):
        super(SpMiddleFHD, self).__init__()
        self.name = name

        # self.dcn = None
        # self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        num_input_features = 4
        self.middle_conv = SparseSequential(
            SubMConv3d(num_input_features, 8, 3, bias=False),
            build_norm_layer(norm_cfg, 8)[1],
            nn.ReLU(),
            SubMConv3d(8, 8, 3, bias=False),
            build_norm_layer(norm_cfg, 8)[1],
            nn.ReLU(),
            SubMConv3d(8, 16, 3, bias=False),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(),
            SubMConv3d(16, 16, 3, bias=False),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(),
            SubMConv3d(16, 32, 3, bias=False),
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SubMConv3d(32, 32, 3, bias=False),
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SubMConv3d(32, 64, 3, bias=False),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, bias=False),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SparseConv3d(64, 64, (1, 3, 3), (1, 2, 2), bias=False),  # [20, 609, 777] -> [20, 304, 388]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 32, 3, bias=False),
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SubMConv3d(32, 32, 3, bias=False),
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SubMConv3d(32, 16, 3, bias=False),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(),
            SubMConv3d(16, 16, 3, bias=False),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(),            
            SubMConv3d(16, 8, 3, bias=False),
            build_norm_layer(norm_cfg, 8)[1],
            nn.ReLU(),
            SubMConv3d(8, 8, 3, bias=False),
            build_norm_layer(norm_cfg, 8)[1],
            nn.ReLU(),
         )
        # self.middle_conv = SparseSequential(
        #     SubMConv3d(num_input_features, 8, 3, bias=False),
        #     build_norm_layer(norm_cfg, 8)[1],
        #     nn.ReLU(),
        #     SubMConv3d(8, 8, 3, bias=False),
        #     build_norm_layer(norm_cfg, 8)[1],
        #     nn.ReLU(),
        #     SubMConv3d(8, 16, 3, bias=False),
        #     build_norm_layer(norm_cfg, 16)[1],
        #     nn.ReLU(),
        #     SparseConv3d(16, 16, (1, 3, 3), (1, 2, 2), bias=False),  # [20, 609, 777] -> [20, 304, 388]
        #     build_norm_layer(norm_cfg, 16)[1],
        #     nn.ReLU(),         
        #     SubMConv3d(16, 8, 3, bias=False),
        #     build_norm_layer(norm_cfg, 8)[1],
        #     nn.ReLU(),
        #     SubMConv3d(8, 8, 3, bias=False),
        #     build_norm_layer(norm_cfg, 8)[1],
        #     nn.ReLU(),
        #  )

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m, 1)

            # if self.dcn is not None:
            #     for m in self.modules():
            #         if isinstance(m, Bottleneck) and hasattr(m, "conv2_offset"):
            #             nn.init.constant_(m.conv2_offset, 0)

            # if self.zero_init_residual:
            #     for m in self.modules():
            #         if isinstance(m, Bottleneck):
            #             nn.init.constant_(m.norm3, 0)
            #         elif isinstance(m, BasicBlock):
            #             nn.init.constant_(m.norm2, 0)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, voxel_features, coors, batch_size, input_shape):

        # input: # [20, 609, 777]
        sparse_shape = np.array(input_shape[::-1]) + [0, 1, 1]
        # sparse_shape = np.array(input_shape[::-1])
        coors = coors.int()


        # voxel_features = voxel_features[:, [1, 0, 2, 3]]
        # coors = coors[:, [0, 1, 3, 2]]
        # sparse_shape = sparse_shape[[0, 2, 1]]

        ret = SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)
        # print(ret.features.shape)
        ret = self.middle_conv(ret)
        ret = ret.dense()

        # pooling, because input is z y x, output is N C D W H
        ret = ret.permute(0, 1, 2, 4, 3).contiguous()
        # N, C, D, W, H = ret.shape
        # N, C, D, H, W = ret.shape
        # ret = ret.view(N, C, D, H*W)        
        # ret = self.pooling(ret)
        # ret = ret.view(N, C, H, W)


        # N, C, D, H, W = ret.shape
        # ret = ret.view(N, C * D, H, W)


        return ret



class MapEncoder(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.cfg = model_cfg
        self.backbone_3d = SpMiddleFHD()

    def forward(self, batch_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                ...
        Returns:
            data_dict:
                points: (N, 3 + C_out),
                use_lead_xyz: whether to use xyz as point-wise features
                ...
        """
        points = batch_dict['lidar_map']

        voxels = batch_dict['voxels']
        coors = batch_dict['voxel_coords']
        num_points_per_voxel = batch_dict['voxel_num_points']

        input_features = points.type_as(voxels).contiguous()  
        # print(input_features.shape)     [max_voxels, 4]
        # print(coors.shape)      #[max_voxels, 3]
        # print('batch_size:', batch_dict['batch_size'])   # 1
        # print(torch.max(coors, dim = 0))
        spatial_features = self.backbone_3d(input_features.cuda(), coors, batch_dict['batch_size'], [776, 608, 20])
        B, C, Z, X, Y = spatial_features.shape    # [1, 8, 20, 388, 304]
        batch_dict['lidar_map_features'] = spatial_features.reshape((B, C, Z, Y, X))
        # print(spatial_features.shape)    [1, 8, 20, 388, 304]

        return batch_dict







