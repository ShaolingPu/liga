# The voxel_feature and bev_feature of our model.

import math
import torch
import torch.nn as nn

from liga.models.backbones_3d_stereo.submodule import convbn, hourglass2d

class VoxelFeatureSimple(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.input_channels = model_cfg.NUM_BEV_FEATURES
        self.num_channels = model_cfg.num_channels
        self.GN = model_cfg.GN
        self.map_fusion = model_cfg.map_fusion
    
        # self.rpn3d_conv2 = nn.Sequential(
        #     convbn(self.input_channels, self.num_channels, 3, 1, 1, 1, gn=self.GN),
        #     nn.ReLU(inplace=True))
        self.rpn3d_conv2 = nn.Sequential(
            convbn(self.input_channels, 160, 3, 1, 1, 1, gn=self.GN),
            nn.ReLU(inplace=True),
            convbn(160, 128, 3, 1, 1, 1, gn=self.GN),
            nn.ReLU(inplace=True),
            convbn(128, self.num_channels, 3, 1, 1, 1, gn=self.GN),
            nn.ReLU(inplace=True))
        self.rpn3d_conv3 = hourglass2d(self.num_channels, gn=self.GN)   

        self.init_params()
        
        # if self.map_fusion:
        #     in_channel = 32 * 4 + 8
        # else:
        #     in_channel = 32 * 4
        in_channel = 32 * 4
        self.height_compress = nn.Sequential(
                    nn.Conv2d(in_channel, 32, kernel_size=(3,1), stride=1, padding=(1,0)),
                    # nn.Conv2d(32*4, 32, kernel_size=(3,1), stride=1, padding=(1,0)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 16, kernel_size=(3,1), stride=1, padding=(1,0)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 8, kernel_size=(3,1), stride=1, padding=(1,0)),
                    nn.ReLU(inplace=True),
                    )
        # self.compute_depth = nn.Sequential(
        #             nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0),
        #             # nn.Sigmoid(),
        #             # nn.Softmax(),
        #             )
        self.softmax = nn.Softmax(3)

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

    # def forward(self, batch_dict):
        
    #     Voxel = batch_dict['voxel_feature']
    #     B, N, C, D, H, W = Voxel.shape
    #     # print(Voxel.shape)    [1, 4, 32, 20, 304, 388]  
    #     # Voxel = Voxel.view(B, N*C, D, -1)   
    #     # Voxel = torch.max(Voxel,dim=1)[0]
    #     # Voxel = Voxel.view(B, C, D, -1)
    #     Voxel = Voxel.view(B, N*C, D, -1)


    #     spatial_features = self.height_compress(Voxel) 
        
        
    #     if self.map_fusion:
    #         map_Voxel = batch_dict['lidar_map_features'].view(B, -1, D, H*W)
            
    #         # import cv2
    #         # for i in range(spatial_features.shape[2]):
    #         #     t = spatial_features[0,:,i].view(-1,H,W).sum(0).detach().cpu().numpy()
    #         #     t = t / t.max() * 255.0
    #         #     cv2.imwrite("spatial_features" + str(i) + ".jpg", t)
    #         # for i in range(map_Voxel.shape[2]):
    #         #     t = map_Voxel[0,:,i].view(-1,H,W).sum(0).detach().cpu().numpy()
    #         #     t = t / t.max() * 255.0
    #         #     cv2.imwrite("map_Voxel" + str(i) + ".jpg", t)
                
    #         spatial_features = torch.cat([spatial_features,map_Voxel],dim=1)

    #     # else:
    #     #     spatial_features = Voxel
    #     # depth_mask = self.compute_depth(spatial_features)
    #     # depth_mask = depth_mask.view(B,D,H,W)
    #     # depth_mask = self.softmax(depth_mask)
        
    #     spatial_features = spatial_features.view(*spatial_features.shape[:3], H, W)
    #     # spatial_features = spatial_features*depth_mask 

    #     # spatial_features = spatial_features.permute(0, 1, 2, 4, 3).contiguous()

    #     N, C, D, H, W = spatial_features.shape


    #     spatial_features = spatial_features.view(N, C * D, H, W)
    #     # spatial_features*
    #     x = self.rpn3d_conv2(spatial_features)
    #     batch_dict['spatial_features_2d_prehg'] = x
    #     x = self.rpn3d_conv3(x, None, None)[0]
    #     # print(x.shape) [1, 64, 304, 388]
    #     batch_dict['spatial_features_2d'] = x

    #     return batch_dict


    def forward(self, batch_dict):
        
        # Voxel = batch_dict['voxel_feature']
        B, N, C, D, H, W = [1, 4, 32, 20, 304, 388]
        # print(Voxel.shape)    [1, 4, 32, 20, 304, 388]  
        # Voxel = Voxel.view(B, N*C, D, -1)   
        # Voxel = torch.max(Voxel,dim=1)[0]
        # Voxel = Voxel.view(B, C, D, -1)
        # Voxel = Voxel.view(B, N*C, D, -1)


        # spatial_features = self.height_compress(Voxel) 
        
        
        # if self.map_fusion:
        spatial_features = batch_dict['lidar_map_features'].view(B, -1, D, H*W)
        # print(spatial_features.shape)
            
            # import cv2
            # for i in range(spatial_features.shape[2]):
            #     t = spatial_features[0,:,i].view(-1,H,W).sum(0).detach().cpu().numpy()
            #     t = t / t.max() * 255.0
            #     cv2.imwrite("spatial_features" + str(i) + ".jpg", t)
            # for i in range(map_Voxel.shape[2]):
            #     t = map_Voxel[0,:,i].view(-1,H,W).sum(0).detach().cpu().numpy()
            #     t = t / t.max() * 255.0
            #     cv2.imwrite("map_Voxel" + str(i) + ".jpg", t)
                
            # spatial_features = torch.cat([spatial_features,map_Voxel],dim=1)

        # else:
        # spatial_features = map_Voxel
        # depth_mask = self.compute_depth(spatial_features)
        # depth_mask = depth_mask.view(B,D,H,W)
        # depth_mask = self.softmax(depth_mask)
        
        spatial_features = spatial_features.view(*spatial_features.shape[:3], H, W)
        # spatial_features = spatial_features*depth_mask 

        # spatial_features = spatial_features.permute(0, 1, 2, 4, 3).contiguous()

        N, C, D, H, W = spatial_features.shape


        spatial_features = spatial_features.view(N, C * D, H, W)
        # spatial_features*
        x = self.rpn3d_conv2(spatial_features)
        batch_dict['spatial_features_2d_prehg'] = x
        x = self.rpn3d_conv3(x, None, None)[0]
        # print(x.shape) [1, 64, 304, 388]
        batch_dict['spatial_features_2d'] = x

        return batch_dict