# 2D detection head based on mmdetection.


import numpy as np
import torch
import torch.nn as nn

from mmdet.models.builder import build_head
from mmdet.core import bbox2result


class MMDet2DHead(nn.Module):
    def __init__(self, model_cfg):
        super(MMDet2DHead, self).__init__()
        self.bbox_head = build_head(model_cfg.cfg)
        self.bbox_head.init_weights()
        self.use_3d_center = model_cfg.use_3d_center

        # if getattr(model_cfg, 'load_from') is not None:
        #     from mmcv.runner.checkpoint import load_state_dict
        #     state_dict = torch.load(model_cfg.load_from, map_location='cpu')['state_dict']
        #     print('loading mmdet head from ', model_cfg.load_from)
        #     load_state_dict(self.bbox_head, {k[10:]: v for k, v in state_dict.items() if k.startswith("bbox_head.")}, strict=False, logger=None)

    def get_loss(self, data_dict, tb_dict):
        losses = {}
        for image_id_ in data_dict['images_id']:

            image_id = str(image_id_.item())
            image = 'image_' + image_id
            img_metas = [{
                "image": data_dict[image][i],  # for debug
                "img_shape": list(data_dict[image][i].shape[1:3]) + [3],
                "pad_shape": list(data_dict[image][i].shape[1:3]) + [3]}
                for i in range(len(data_dict[image]))]
            if self.use_3d_center:
                gt_boxes_2d = \
                    torch.cat([data_dict['gt_boxes_2d_' + image_id], data_dict['gt_centers_2d_' + image_id]], dim=-1)
                # gt_boxes_2d = torch.cat([data_dict['bbox' + image_id], data_dict['gt_centers_2d' + image_id]], dim=-1)
            else:
                gt_boxes_2d = data_dict['gt_boxes_2d']

            gt_boxes_2d = torch.unbind(gt_boxes_2d)
            gt_boxes_3d = data_dict['gt_boxes_no3daug']
            gt_labels = torch.unbind(gt_boxes_3d[:, :, 7].long() - 1)  # a list of [N] tensors
            gt_bboxes_2d_ignore = torch.unbind(data_dict['gt_boxes_2d_ignored']) if 'gt_boxes_2d_ignored' in data_dict else None  # a list of [N, 4] tensors

            lossestmp = self.bbox_head.forward_train(data_dict['sem_features_' + image_id], \
                                          img_metas, gt_boxes_2d, gt_labels, gt_bboxes_2d_ignore)
            if image_id_ not in data_dict['cur_id']:
                for key in lossestmp:
                    for id in range(len(lossestmp[key])):
                        lossestmp[key][id] = lossestmp[key][id] * 0
            for k,v in lossestmp.items():
                lossestmp[k] = sum(_loss for _loss in v)
                losses[k] = losses.get(k,torch.tensor([0.],device=lossestmp[k].device)) + lossestmp[k]

        for k, v in losses.items():
            if not isinstance(v, (list, tuple)) and len(v.shape) == 0:
                _sum_loss = v
            else:
                _sum_loss = sum(_loss for _loss in v)
            assert len(_sum_loss.shape) == 0
            # if k != 'loss_bbox':
            #     assert len(_sum_loss.shape) == 0
            # else:
            #     assert len(_sum_loss.shape) in [0, 1]
            #     if len(_sum_loss.shape) == 1:
            #         assert _sum_loss.shape[0] < 10
            # if len(_sum_loss.shape) == 1:
            #     for i in range(_sum_loss.shape[0]):
            #         tb_dict['rpn2d_' + k + '_' + str(i)] = _sum_loss[i].item()
            losses[k] = _sum_loss.sum()
            tb_dict['rpn2d_' + k] = losses[k].item()
        loss_sum = sum([v for _, v in losses.items()])
        return loss_sum, tb_dict

    def forward(self, data_dict):
        return data_dict
        if 'test_one' in data_dict.keys():
            return data_dict
        if self.training:
            return data_dict
        else:
            img_metas = [{
                "img_shape": list(data_dict['left_img'][i].shape[1:3]) + [3],
                "pad_shape": list(data_dict['left_img'][i].shape[1:3]) + [3],
                "scale_factor": 1.0}  # TODO: scale factor from dataset
                for i in range(len(data_dict['left_img']))]
            outs = self.bbox_head(data_dict['sem_features'])
            data_dict['head_outs'] = outs

            try:
                bbox_list = self.bbox_head.get_bboxes(*outs, img_metas, rescale=False)  # TODO: rescale

                bbox_results = [
                    bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                    for det_bboxes, det_labels in bbox_list
                ]

                data_dict['boxes_2d_pred'] = []
                for bbox_result in bbox_results:
                    pred_dict = {}
                    pred_dict['pred_boxes_2d'] = np.concatenate([x[:, :-1] for x in bbox_result])
                    pred_dict['pred_scores_2d'] = np.concatenate([x[:, -1] for x in bbox_result])
                    pred_dict['pred_labels_2d'] = np.concatenate([[cls_id + 1] * len(x) for cls_id, x in enumerate(bbox_result)]).astype(np.int64)
                    data_dict['boxes_2d_pred'].append(pred_dict)
            except NotImplementedError:
                print("not implemented get_bboxes, skip")

            return data_dict
