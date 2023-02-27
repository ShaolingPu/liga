# The backbone of our model.
from email.mime import image
import numpy as np
import math
import torch
import torch.nn as nn
import torch.utils.data
import cv2

from .resnet import resnet34
from .submodule import feature_extraction_neck
from mmdet.models.builder import build_neck


class BackBoneSimple(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.feature_backbone = resnet34(pretrained=False)
        pretrain_model = torch.load('./models/resnet34-333f7ec4.pth')
        model_dict = self.feature_backbone.state_dict()
        state_dict = {k:v for k,v in pretrain_model.items() if k in model_dict.keys() and v.shape == model_dict[k].shape}
        model_dict.update(state_dict)
        self.feature_backbone.load_state_dict(model_dict)
        self.feature_neck = feature_extraction_neck(model_cfg.feature_neck)
        if getattr(model_cfg, 'sem_neck', None):
            self.sem_neck = build_neck(model_cfg.sem_neck)
        else:
            self.sem_neck = None

        self.init_params()
        self.maxpool = nn.MaxPool2d((2, 2), (2, 2), 0)
        # self.compute_depth =  nn.Sequential(
        #         nn.Conv2d(32,70,
        #               kernel_size=1,
        #               padding=0,
        #               stride=1,
        #               bias=False),
        #         nn.Softmax())

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

        for image_id in batch_dict['images_id']:
            image_id = str(image_id.item())
            image_name = 'image_' + image_id
            features = self.feature_backbone(self.maxpool(batch_dict[image_name]))
            features = [self.maxpool(batch_dict[image_name])] + list(features)

            # features = self.feature_backbone(batch_dict[image_name])
            # features = [batch_dict[image_name]] + list(features)
            stereo_feat, sem_feat = self.feature_neck(features)
            batch_dict['stereo_feat_' + image_id] = stereo_feat
            batch_dict['rpn_feature_' + image_id] = sem_feat
            batch_dict['sem_features_' + image_id] = self.sem_neck([sem_feat])
            

        # if type(batch_dict) == list:
        #     left = batch_dict[0]
        #     right = batch_dict[1]
        # else:
        #     left = batch_dict['left_img']
        #     right = batch_dict['right_img']
        
        # '’’Backbone and Neck'''
        # left_features = self.feature_backbone(left)
        # left_features = [left] + list(left_features)
        # right_features = self.feature_backbone(right)
        # right_features = [right] + list(right_features)
        # left_stereo_feat, left_sem_feat = self.feature_neck(left_features)
        # right_stereo_feat, _ = self.feature_neck(right_features)

        # '''Use for train, for 2D detec train'''
        # if 'test_one' not in batch_dict.keys(): #add by me
        #     if self.sem_neck is not None:  
        #         batch_dict['sem_features'] = self.sem_neck([left_sem_feat])
        #     else:
        #         batch_dict['sem_features'] = [left_sem_feat]

        # batch_dict['rpn_feature'] = left_sem_feat
        # batch_dict['left_stereo_feat'] = left_stereo_feat
        # batch_dict['right_stereo_feat'] = right_stereo_feat

        return batch_dict
