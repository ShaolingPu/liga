import sys

if '/home/zjlab/wsq/liga_test' in sys.path:
    sys.path.remove('/home/zjlab/wsq/liga_test')
sys.path.append('/home/zjlab/psl/liga_test')

import numpy as np
import torch
import torch.nn as nn
from .voxel_generator import *

class PointMapEncoder(object):
    def __init__(self, config):
        super().__init__()
        voxel_size = config['voxel_size']
        point_cloud_range = config['point_cloud_range']
        max_num_points = config['max_num_points']
        self.generator = VoxelGenerator(voxel_size, point_cloud_range, max_num_points, max_voxels=20000)

    def forward(self, data_dict):
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
        points = data_dict['lidar_map']

        voxels, coors, num_points_per_voxel = self.generator.generate(points)
        # print('The max points number is: ', np.max(num_points_per_voxel))
        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coors
        data_dict['voxel_num_points'] = num_points_per_voxel

        input_density = np.minimum(1.0, np.log(num_points_per_voxel + 1) / np.log(64)).reshape(-1, 1) 
        input_xyz = np.sum(voxels, axis=1) / num_points_per_voxel.astype(voxels.dtype).reshape(-1, 1) 
        input_features = np.hstack([input_xyz, input_density])  

        input_features = input_features.astype(voxels.dtype)
        data_dict['lidar_map'] = input_features
        return data_dict

