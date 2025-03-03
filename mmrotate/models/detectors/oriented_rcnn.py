# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import ROTATED_DETECTORS
from .two_stage import RotatedTwoStageDetector
from mmrotate.core import rbbox2roi, build_assigner, build_sampler, obb2xyxy, rbbox2result
import numpy as np
import pdb

@ROTATED_DETECTORS.register_module()
class OrientedRCNN(RotatedTwoStageDetector):
    """Implementation of `Oriented R-CNN for Object Detection.`__

    __ https://openaccess.thecvf.com/content/ICCV2021/papers/Xie_Oriented_R-CNN_for_Object_Detection_ICCV_2021_paper.pdf  # noqa: E501, E261.
    """

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(OrientedRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmrotate/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 6).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def feats_extract(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        
        img = img.cuda()
        x = self.extract_feat(img)

        # RPN forward and loss
        if self.with_rpn:

            rpn_outs = self.rpn_head(x)
            # rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                        #   self.train_cfg.rpn)
            # rpn_losses = self.rpn_head.loss(
            #     *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            # losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)

            # proposal_cfg = self.train_cfg.get('rpn_proposal',
            #                                   self.test_cfg.rpn)
            # rpn_losses, proposal_list = self.rpn_head.forward_train(
            #     x,
            #     img_meta,
            #     gt_bboxes,
            #     gt_labels=None,
            #     gt_bboxes_ignore=gt_bboxes_ignore,
            #     proposal_cfg=proposal_cfg,
            #     **kwargs)
            
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)

            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i].cuda(),
                                                     gt_bboxes[i].cuda(),
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i].cuda())
                # import pdb; pdb.set_trace()
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i].cuda(),
                    gt_bboxes[i].cuda(),
                    gt_labels[i].cuda(),
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss

        # if self.with_bbox:
        # pdb.set_trace()
        rois = rbbox2roi([res.bboxes for res in sampling_results])
        # TODO: a more flexible way to decide which feature maps to use
        x_new = [t.cuda() for t in x]
        bbox_feats = self.roi_head.bbox_roi_extractor(x_new[:self.roi_head.bbox_roi_extractor.num_inputs], rois)
        
        # bbox_feats-->shape ==> num_boxes x 5
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        # cls_score, bbox_pred = self.bbox_head(bbox_feats)
        # aaa = self.classfier(bbox_feats)

        if self.roi_head.bbox_head.num_shared_fcs > 0:
            # already avg_pooled 
            # if self.with_avg_pool:
            #     x = self.avg_pool(x)
            bbox_feats = bbox_feats.view(bbox_feats.size(0), -1)
            for fc in self.roi_head.bbox_head.shared_fcs:
                # import pdb; pdb.set_trace()
                bbox_feats = self.roi_head.bbox_head.relu(fc(bbox_feats))

        #pdb.set_trace()
        bbox_targets = self.roi_head.bbox_head.get_targets(sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
        #pdb.set_trace()

        bg_inds = np.where(bbox_targets[0].data.cpu().numpy()==0)[0]
        fg_inds = np.where(bbox_targets[0].data.cpu().numpy()>0)[0]
        #bg_scores = cls_score[:, 0]
        #sorted_args = np.argsort(bg_scores.data.cpu().numpy(), kind='mergesort')[:len(fg_inds)*3]
        #selected_bg_inds = np.intersect1d(sorted_args, bg_inds)
        sub_neg_inds = np.random.permutation(bg_inds)[:int(2*len(fg_inds))]
        # 
        inds_to_select = np.concatenate((sub_neg_inds, fg_inds))
        return bbox_feats[inds_to_select], bbox_targets[0][inds_to_select], bbox_targets[2][inds_to_select]
        # return bbox_feats, bbox_targets[0], bbox_targets[2]