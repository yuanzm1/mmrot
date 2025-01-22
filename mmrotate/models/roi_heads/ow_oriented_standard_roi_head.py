# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import time
import pdb

from .VAE import DualVAE, loss_function_vae
from mmrotate.core import rbbox2roi, build_assigner, build_sampler, obb2xyxy, rbbox2result
from ..builder import ROTATED_HEADS, build_loss
from .rotate_standard_roi_head import RotatedStandardRoIHead

import heapq
import operator
import warnings
import cv2, math

import os
import logging
from torch import nn
from torch.nn import functional as F
import numpy as np
from mmdet.models.losses.smooth_l1_loss import SmoothL1Loss
from mmrotate.models.losses import ConvexGIoULoss
from mmrotate.core import obb2poly, obb2poly_np

# for adaptive clustering modules
from .adaptive_clustering import * 
from .offline import *
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import random
from collections import deque, defaultdict
import numpy as np
from datetime import datetime

class Store:
    def __init__(self, total_num_classes, items_per_class, shuffle=False):
        self.shuffle = shuffle
        self.items_per_class = items_per_class
        self.total_num_classes = total_num_classes
        self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]

    def add(self, items, class_ids):
        for idx, class_id in enumerate(class_ids):
            self.store[class_id].append(items[idx])

    def retrieve(self, class_id):
        if class_id != -1:
            items = []
            for item in self.store[class_id]:
                items.extend(list(item))
            if self.shuffle:
                random.shuffle(items)
            return items
        else:
            all_items = []
            for i in range(self.total_num_classes):
                items = []
                for item in self.store[i]:
                    items.append(list(item))
                all_items.append(items)
            return all_items

    def reset(self):
        self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]

    def __str__(self):
        s = self.__class__.__name__ + '('
        for idx, item in enumerate(self.store):
            s += '\n Class ' + str(idx) + ' --> ' + str(len(list(item))) + ' items'
        s = s + ' )'
        return s

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return sum([len(s) for s in self.store])

def per_class_accuracy(a, b):
    device = a.device
    num_classes = b.shape[1] - 4
    per_class_acc = torch.zeros(num_classes).to(device)
    per_class_count = torch.zeros(num_classes).to(device)
    for c in range(num_classes):
        # 找出 a 中第 c 类为最大值的样本索引
        indices_a = torch.argmax(a, dim=1) == c
        # 找出 b 中第 c 类为最大值的样本索引
        indices_b = torch.argmax(b, dim=1) == c
        # 找出在 a 中第 c 类为最大值的样本
        samples_a = a[indices_a]
        # 找出在 b 中第 c 类为最大值的样本
        samples_b = b[indices_a]
        if samples_a.numel() > 0:
            # 找出在 a 中第 c 类为最大值的样本在 b 中也是第 c 类为最大值的样本
            correct = torch.sum(torch.argmax(samples_b, dim=1) == c).to(device)
            # 计算该类的准确率
            acc = correct.float() / samples_a.shape[0]
            per_class_acc[c] += acc.to(device)
            per_class_count[c] += 1
    return per_class_acc, per_class_count

@ROTATED_HEADS.register_module()
class OwOrientedStandardRoIHead(RotatedStandardRoIHead):
    """Oriented RCNN roi head including one bbox head."""
    def __init__(self,text_super=None,*args, **kwargs):
        super(OwOrientedStandardRoIHead, self).__init__(*args, **kwargs)
        self.text_project = nn.Linear(1024, 1024)
        self.text_project2 = nn.ReLU()
        self.text_project3 = nn.Linear(1024, 1024)
        
        self.img_project = nn.Linear(1024, 1024)
        torch.nn.init.kaiming_normal_(self.img_project.weight)
        self.img_project2 = nn.ReLU()
        self.img_project3 = nn.Linear(1024, 1024)
        torch.nn.init.kaiming_normal_(self.img_project3.weight)
        
        self.text_super = text_super
        self.uk_score = []
        self.cls_trans_feature = defaultdict(list)
        self.accum_acc = torch.zeros(self.bbox_head.num_classes-1).to(device='cuda')
        self.accum_cnt = torch.zeros(self.bbox_head.num_classes-1).to(device='cuda')
        
        self.num_iter = 0
        self.mkdir = False
        self.dualvae = DualVAE(1024,1024,512,64,self.bbox_head.num_classes+3)
        
        # if self.

    def forward_dummy(self, x, proposals):
        """Dummy forward function.

        Args:
            x (list[Tensors]): list of multi-level img features.
            proposals (list[Tensors]): list of region proposals.

        Returns:
            list[Tensors]: list of region of interest.
        """
        outs = ()
        rois = rbbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        return outs

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        # 计数预测框数量
        self.count = 0
            
        # self.bbox_sampler_unknown = None # 增加第二个sampler
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.train_iter = 0
            self.num_classes = self.train_cfg.num_classes
            if self.bbox_assigner.unknown:
                self.enable_clustering = True

                # 储存原型特征向量
                # 先定义一个原型特征的储存位置
                root_dir = '/home/yuanzm/mmrotate/work_dirs/'
                os.makedirs(root_dir, exist_ok=True)
                self.feature_store_save_loc = os.path.join(root_dir, 'feat.pt')

                self.means = [None for _ in range(self.train_cfg.num_classes + self.train_cfg.num_unk_proto - 1)]
                
                # 增加一个unknown_mean
                self.unknown_means = [None for _ in range(self.train_cfg.num_unk_proto)]
                
                # 增加一个自适应改进的原型特征的初始化
                self.new_means_adaptive =  [None for _ in range(self.train_cfg.num_classes + self.train_cfg.num_unk_proto - 1)]
                
                self.margin = 10.0
                self.hingeloss = nn.HingeEmbeddingLoss(2)

                # if os.path.isfile(self.feature_store_save_loc):
                #     logging.getLogger(__name__).info('Trying to load feature store from ' + self.feature_store_save_loc)
                #     self.feature_store = torch.load(self.feature_store_save_loc)
                # else:
                logging.getLogger(__name__).info('Feature store not found in ' +
                                                self.feature_store_save_loc + '. Creating new feature store.')
                
                self.clustering_items_per_class = self.train_cfg.assigner['items_per_class'] # 初始化空的聚类原型向量数量
                
                # 保存
                # self.feature_store = Store(self.train_cfg.num_classes, self.clustering_items_per_class)  #原版
                self.feature_store = Store(self.train_cfg.num_classes + self.train_cfg.num_unk_proto - 1, self.clustering_items_per_class)
                    

                self.ogiou_loss = ConvexGIoULoss(loss_weight=1.0) # 增加自监督旋转归一化损失
                
                # self.random_seed = 4
                # random.seed(self.random_seed) # 增加一个随机种子
                
                if self.train_cfg.assigner['text_super']:
                    self.text_feature = torch.load(self.train_cfg.assigner['text_super']).transpose(0, 1)
                    a = 0
                    # self.text_project = nn.Linear(1024, 1024)
                    # self.text_project2 = nn.ReLU()
                    # self.text_project3 = nn.Linear(1024, 1024)
                    self.text_loss = build_loss(dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0))
                    

            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)
            
            # 增加第二个sampler
            # self.bbox_sampler_unknown = build_sampler(
            #     self.train_cfg.sampler, context=self)
                    # 增加一个contra的特征层
            if self.bbox_assigner.unknown_linear:
                self.nnlinear1 = nn.Linear(1024, 1024)
                torch.nn.init.kaiming_normal_(self.nnlinear1.weight)
                self.nnlinear2 = nn.ReLU()
                self.nnlinear3 = nn.Linear(1024, 1024)
                torch.nn.init.kaiming_normal_(self.nnlinear3.weight)

    def obtain_unknown(self, assign_result, bboxes, unknown_num, gt_bbox, img_metas):
        matched_label_ss = torch.where(assign_result.labels != -1) # 被选中为正样本的候选框
    
        pred_objectness_score_ss = bboxes[:, -1].clone() # 候选框的得分
        pred_objectness_score_ss[matched_label_ss] = -1 # 删除正样本的得分
        sorted_indices = list(zip(*heapq.nlargest(unknown_num , enumerate(pred_objectness_score_ss), 
                                                  key=operator.itemgetter(1))))[0] # 选择最高的top个unknown类样本
        sorted_indices = [i for i in sorted_indices]
        
        #_, sorted_indices = torch.topk(pred_objectness_score_ss, unknown_num)
        assign_result.labels[sorted_indices] = self.num_classes - 1 #-2 将最后一类作为unknown类试试 15
        assign_result.gt_inds[sorted_indices] = -1 # 分配真值标签框的index

        return assign_result, sorted_indices

    def obtain_contra_feats(self, trans_feature, sampling_results):

        # 这里也得更新 
        batch_size = len(sampling_results)
        # num_unknown = self.train_cfg.assigner['unknown_num']
        rois_num_per_batch = int(trans_feature.shape[0]/ batch_size)

        pos_feature = torch.zeros(1, trans_feature.shape[1]).to('cuda') 
        pos_gt_labels = torch.zeros(1).to('cuda')
        
        # 记录粗糙正样本的位置
        pos_index = []
        
        for i in range(batch_size):
            feature_single = trans_feature[i * rois_num_per_batch: (i + 1) * rois_num_per_batch, :]
            pos_single = feature_single[:sampling_results[i].pos_gt_labels.shape[0], :]
            
            pos_index.extend([j + i * rois_num_per_batch for j in range(sampling_results[i].pos_gt_labels.shape[0])])
            
            pos_gt_labels_single = sampling_results[i].pos_gt_labels
            # assert pos_gt_labels_single.shape[0] == pos_single.shape[0], "有bug"
            pos_feature = torch.cat((pos_feature, pos_single), dim=0)
            pos_gt_labels = torch.cat((pos_gt_labels, pos_gt_labels_single))

        input_features = pos_feature[1:, :]
        proposal_labels = pos_gt_labels[1:]

        return input_features, proposal_labels, torch.tensor(pos_index)

    def visualize(self, img_metas, sample_result):
        img_name = img_metas['filename']
        img = cv2.imread(img_name)
        img2 = cv2.resize(img, (img_metas['img_shape'][1], img_metas['img_shape'][0]))
        # a = cv2.namedWindow("show")
        # cv2.imshow(a, img2)
        # cv2.waitKey(0)

        if img_metas['flip_direction'] == 'vertical':
            img2 = cv2.flip(img2, 0)
        if img_metas['flip_direction'] == 'horizontal':
            img2 = cv2.flip(img2, 1)
        if img_metas['flip_direction'] == 'diagonal':
            img2 = cv2.flip(img2, -1)
            
        # if rotate:
        #     img2 = cv2.rotate(img2, cv2.ROTATE_180)
        
        proposal_1 = sample_result.info['pos_bboxes']
        labels = sample_result.info['pos_assigned_gt_inds']
        for i,  boxes_j in enumerate(proposal_1):
            # pdb.set_trace()
            boxes_j = obb2poly(boxes_j.unsqueeze(0), version=self.bbox_head.bbox_coder.angle_range)
            boxes_j = boxes_j.detach().cpu().numpy()
            # if boxes_j[-1] > 0.5:
            # quads = cv2.boxPoints(((int(boxes_j[0]),int(boxes_j[1])), (int(boxes_j[2]), int(boxes_j[3])), boxes_j[4] * 180 / math.pi)).reshape((1, 8))
            quads = boxes_j

            pts = np.array([quads.reshape((4, 2))], dtype=np.int32)
            if labels[i].item() == self.num_classes - 1:
                cv2.drawContours(img2, pts, 0, color=(255, 0, 0),
                            thickness=2)
            else:
                cv2.drawContours(img2, pts, 0, color=(0, 255, 0),
                            thickness=2)            
        
        img_num = img_metas['ori_filename'].split('.')[0]
        # cv2.imwrite(f"/media/disk2/guochen/visualize/proposal_{img_num}.jpg", img2)
        plt.imshow(img2)
        plt.show()
        
    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task. Always
                set to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        time1 = time.time()
        # assign gts and sample proposals
        if self.with_bbox:

            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            
            ori_gtbox = copy.deepcopy(gt_bboxes)
            ori_gtlbl = copy.deepcopy(gt_labels)
            target_bboxes = gt_bboxes
            target_bboxes_ignore = gt_bboxes_ignore
            # 删掉unknown类得到baseline
            for i in range(len(gt_labels)):
                where_unknown_classes = torch.where(gt_labels[i] != self.num_classes - 1)[0]
                target_bboxes[i] = target_bboxes[i][where_unknown_classes, :]
                gt_labels[i] = gt_labels[i][where_unknown_classes]
    
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], target_bboxes[i], target_bboxes_ignore[i],
                    gt_labels[i])
                
                # 自动给unknown类别打标签
                if self.train_cfg.assigner['unknown']:
                    time4 = time.time()
                    # assign_result, sorted_index = self.obtain_unknown_new(assign_result, proposal_list[i], 
                    #                                                       self.train_cfg.assigner['unknown_num'], target_bboxes[i], img_metas[i])
                    assign_result, sorted_index = self.obtain_unknown(assign_result, proposal_list[i], 
                                                                          self.train_cfg.assigner['unknown_num'], target_bboxes[i], img_metas[i])
                    # time5 = time.time()
                    # print("obtain_unknown", time5-time4)
                    # 更新unknown类别的选取方式——通过selective search的监督
                    
                    # 增加伪标签真值结果
                    pseudo_proposal_unknown = proposal_list[i][sorted_index, :-1]
                    pseudo_label_unknown = torch.ones(len(sorted_index)).int().to('cuda') * (self.num_classes - 1)
                   
                    target_bboxes[i] = torch.cat((target_bboxes[i], pseudo_proposal_unknown))
                    # gt_bboxes[i] = torch.cat((gt_bboxes[i], pseudo_proposal_unknown))
                    gt_labels[i] = torch.cat((gt_labels[i], pseudo_label_unknown))
                                    
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    target_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                
                if target_bboxes[i].numel() == 0:
                    sampling_result.pos_gt_bboxes = target_bboxes[i].new(
                        (0, target_bboxes[0].size(-1))).zero_()
                else:
                    sampling_result.pos_gt_bboxes = \
                        target_bboxes[i][sampling_result.pos_assigned_gt_inds, :]
                        
                # 可视化proposal的位置
                # self.visualize(img_metas[i], sampling_result)               

                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
        #pdb.set_trace()
        if not bbox_results['loss_bbox']['loss_bbox'] < 10:
            # has_nan = torch.isnan(bbox_targets[pos_inds.type(torch.bool)]).any()
            warnings.warn("This is a warning message", UserWarning)
            bbox_results['loss_bbox']['loss_bbox'] = torch.zeros(1).to('cuda')
            if False:
                for k in range(len(img_metas)):
                    img_meta = img_metas[k]
                    img_name = img_meta['filename']
                    img = cv2.imread(img_name)
                    img2 = cv2.resize(img, (img_meta['img_shape'][1], img_meta['img_shape'][0]))

                    if img_meta['flip_direction'] == 'vertical':
                        img2 = cv2.flip(img2, 0)
                    if img_meta['flip_direction'] == 'horizontal':
                        img2 = cv2.flip(img2, 1)
                    if img_meta['flip_direction'] == 'diagonal':
                        img2 = cv2.flip(img2, -1)
                    img3 = img2.copy()
                        
                    # sample_result = sampling_results[k]
                    # proposal_1 = sample_result.info['pos_bboxes']
                    # labels = sample_result.info['pos_assigned_gt_inds']
                    # for i,  boxes_j in enumerate(proposal_1):
                    #     boxes_j = boxes_j.detach().cpu().numpy()
                    #     # if boxes_j[-1] > 0.5:
                    #     quads = cv2.boxPoints(((int(boxes_j[0]),int(boxes_j[1])), (int(boxes_j[2]), int(boxes_j[3])), boxes_j[4] * 180 / math.pi)).reshape((1, 8))
                    #     pts = np.array([quads.reshape((4, 2))], dtype=np.int32)
                    #     if labels[i].item() == self.num_classes - 1:
                    #         cv2.drawContours(img2, pts, 0, color=(255, 0, 0),
                    #                     thickness=2)
                    #     else:
                    #         cv2.drawContours(img2, pts, 0, color=(0, 255, 0),
                    #                     thickness=2)   
                    for i,  boxes_j in enumerate(gt_bboxes[k]):
                        boxes_j = boxes_j.detach().cpu().numpy()
                        # if boxes_j[-1] > 0.5:
                        quads = cv2.boxPoints(((int(boxes_j[0]),int(boxes_j[1])), (int(boxes_j[2]), int(boxes_j[3])), boxes_j[4] * 180 / math.pi)).reshape((1, 8))
                        pts = np.array([quads.reshape((4, 2))], dtype=np.int32)
                        if gt_labels[k][i].item() == self.num_classes - 1:
                            cv2.drawContours(img3, pts, 0, color=(255, 0, 0),
                                        thickness=2)
                        else:
                            cv2.drawContours(img3, pts, 0, color=(0, 255, 0),
                                        thickness=2)   
                    for i,  boxes_j in enumerate(ori_gtbox[k]):
                        boxes_j = boxes_j.detach().cpu().numpy()
                        # if boxes_j[-1] > 0.5:
                        quads = cv2.boxPoints(((int(boxes_j[0]),int(boxes_j[1])), (int(boxes_j[2]), int(boxes_j[3])), boxes_j[4] * 180 / math.pi)).reshape((1, 8))
                        pts = np.array([quads.reshape((4, 2))], dtype=np.int32)
                        if ori_gtlbl[k][i].item() == self.num_classes - 1:
                            cv2.drawContours(img3, pts, 0, color=(0, 0, 255),
                                        thickness=2)                   
                    legend_elements = [
                        plt.Line2D([0], [0], marker='s', color=(1, 1, 1, 0), label='pseudo unknown', markerfacecolor='b', markersize=12),
                        plt.Line2D([0], [0], marker='s', color=(1, 1, 1, 0), label='known', markerfacecolor='g', markersize=12),
                        plt.Line2D([0], [0], marker='s', color=(1, 1, 1, 0), label='true unknown', markerfacecolor='r', markersize=12)
                    ]
                    # red_patch = mpatches.Patch(color='red', label='pseudo unknown')
                    # blue_patch = mpatches.Patch(color='blue', label='known')
                    # green_patch = mpatches.Patch(color='green', label='true unknown')
                    # plt.legend(handles=legend_elements, loc='upper right',  framealpha=0.1)
                    if not self.mkdir:
                        current_datetime = datetime.now()
                        formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
                        dir_path = f"/home/yuanzm/mmrotate/problem_pics/{formatted_datetime}"
                        os.mkdir(dir_path)
                        fig, ax = plt.subplots()
                        # 添加 legend 到 Axes 对象
                        ax.legend(handles=legend_elements, loc='center')
                        # 调整 Axes 对象的范围，确保 legend 可见
                        ax.set_xlim(-0.5, 0.5)
                        ax.set_ylim(-0.5, 0.5)
                        # 保存图像
                        plt.savefig(os.path.join(dir_path,'legend_only.png'))
                        self.mkdir = dir_path
                    cv2.imwrite(os.path.join(self.mkdir,img_meta['ori_filename']), img3)
                    
                    # plt.subplot(121)
                    # plt.imshow(img2)
                    # plt.subplot(122)
                    # plt.imshow(img3)
                    # plt.show()
                # self.visualize(img_metas[i],sampling_results[i])
        losses.update(bbox_results['loss_bbox'])
                
        # time3 = time.time()
        # print("_bbox_forward_train:", time3-time2, "assign:", time2-time1)        
                
        return losses
    
    def _bbox_forward_attention(self, x, rois):
        """add prototype attention for feature"""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
            
        # a = self.adaptive_proposal_feature

        # bbox_feats的维度为：batch * num_of_proposals, 7, 7 需要将这个bbox feat变成向量？
        cls_score, bbox_pred, trans_feature = self.bbox_head(bbox_feats, trans=True)
        # cls_score, bbox_pred  = self.bbox_head(bbox_feats)

        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, trans_feature=trans_feature)
        
        return bbox_results
    
    def update_feature_store(self, features, proposals):
        """更新储存的类别原型特征"""
        # cat(..., dim=0) concatenates over all images in the batch

        now_iter = self.train_iter
        gt_classes = proposals.int() # torch.cat([p.gt_classes for p in proposals])
        # print(gt_classes)
        self.feature_store.add(features, gt_classes)

        # 保存原型feature的结果 HRSC是8892 结束, dota是44000结束
        if now_iter == 44000: # and self.feature_store_is_stored is False: # and comm.is_main_process():
            logging.getLogger(__name__).info('Saving image store at iteration ' + str(now_iter) + ' to ' + self.feature_store_save_loc)
            torch.save(self.feature_store, self.feature_store_save_loc)

        return features, proposals

    def clstr_loss_l2_cdist_new(self, input_features, proposals):
        """
        Get the foreground input_features, generate distributions for the class,
        get probability of each feature from each distribution;
        Compute loss: if belonging to a class -> likelihood should be higher
                      else -> lower
        :param input_features:
        :param proposals:
        :return:
        
        修改对比聚类损失, 实现对unknown类别目标的自动筛选
        """
        gt_classes = proposals # torch.cat([p.gt_classes for p in proposals])
        fg_features = input_features
        classes = gt_classes.int()

        # fg_features = F.normalize(fg_features, dim=0)
        # fg_features = self.ae_model.encoder(fg_features)

        all_means = self.means
        for item in all_means:
            if item != None:
                length = 1024 # item.shape (改成1024以去除bug)
                break

        for i, item in enumerate(all_means):
            if item == None:
                # all_means[i] = torch.zeros((length))
                all_means[i] = torch.zeros((1024))
                
        # 针对unknown类进行改进
        labels = []
        
        # 默认1000开始启动
        distances = torch.cdist(fg_features, torch.stack(all_means).cuda(), p=self.margin) # num_of_fg * num_of_classes

        for index, feature in enumerate(fg_features):
            for cls_index, mu in enumerate(self.means):
                if mu is not None and feature is not None:
                    if  classes[index] ==  cls_index:
                        labels.append(1)
                    else:
                        labels.append(-1)
                else:
                    labels.append(0)

        loss = self.hingeloss(distances, torch.tensor(labels).reshape((-1, self.num_classes + self.train_cfg.num_unk_proto - 1)).cuda())

        return loss
    
    def normalize(self, x, axis=-1):
        x = 1. *x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
        return x

    def get_adaptive_clustering_loss(self, input_features, proposals):
        """计算对比聚类损失"""
        if not self.enable_clustering:
            return 0
        
        now_iter = self.train_iter

        c_loss = torch.zeros(1).to('cuda')
        if now_iter == self.bbox_assigner.start_iter:
            items = self.feature_store.retrieve(-1)
            for index, item in enumerate(items):
                if len(item) == 0:
                    self.means[index] = None
                else:
                    mu = torch.tensor(item).mean(dim=0)
                    self.means[index] = mu
            # c_loss = self.clstr_loss_l2_cdist(input_features, proposals) # 尝试对loss进行更新
            c_loss = self.clstr_loss_l2_cdist_new(input_features, proposals) # 尝试对loss进行更新
            
            # Freeze the parameters when clustering starts
            # for param in self.ae_model.parameters():
            #     param.requires_grad = False
        elif now_iter > self.bbox_assigner.start_iter:
        # if now_iter > 0:
            if now_iter % self.bbox_assigner.update_mu_iter == 0:
            # if now_iter % self.bbox_assigner.update_mu_iter != 0:
                
                # Compute new MUs
                items = self.feature_store.retrieve(-1)
                
                new_means = [None for _ in range(self.num_classes + self.train_cfg.num_unk_proto - 1)]
                new_means_adaptive = [None for _ in range(self.num_classes + self.train_cfg.num_unk_proto - 1)]
                new_means_2 = torch.zeros([self.train_cfg.num_unk_proto, 1024])
                new_means_3 = torch.zeros([self.train_cfg.num_unk_proto, 1024])
                
                for index, item in enumerate(items):
                    if len(item) == 0:
                        new_means[index] = None
                        new_means_adaptive[index] = None
                        
                        if index >= self.num_classes - 1:
                            new_means_3[index - self.num_classes + 1] = torch.ones(1024) * random.random() * 10
                    else:
                        new_means[index] = torch.tensor(item).mean(dim=0)
                        new_means_adaptive[index] = torch.tensor(item).mean(dim=0)
                        if index >= self.num_classes - 1:
                            new_means_3[index - self.num_classes + 1] = torch.tensor(item).mean(dim=0)
                # 对items进行聚类获得更新的means，更新的means需要跟原本的对应
                
                        # ###        
                # 如果使用k-means呢，这样的话需要对每一次的mean作重新定义， k-means的类别无固定（如何解决）
                
                probs = items[self.num_classes - 1 :] # 将其放在cpu上使用array等进行计算
                # probs = probs.
                
                probs_new = torch.zeros((1, 1024))
                for index, item in enumerate(probs):
                    if len(item) > 0:
                        probs_new = torch.cat((probs_new, torch.tensor(item)))

                if probs_new.shape[0] > self.train_cfg['num_unk_proto']:
                
                    # kmeans = KMeans(n_clusters=self.train_cfg['num_unk_proto'],
                    #                 random_state=4).fit(probs_new[1: , :]).labels_
                    
                    if all(element is not None for element in self.means[self.num_classes - 1 :]):
                        kmeans = KMeans(n_clusters=self.train_cfg['num_unk_proto'],
                                    random_state=4, init=torch.stack(self.means[self.num_classes - 1 :]), n_init=1).fit(probs_new[1: , :]).labels_
                    else:
                        kmeans = KMeans(n_clusters=self.train_cfg['num_unk_proto'],
                                    random_state=4).fit(probs_new[1: , :]).labels_
 
                
                    # 进行坐标重新对齐
                    for i in range(self.train_cfg['num_unk_proto']):

                        new_means_2[i] = probs_new[np.where(kmeans == i)[0], :].mean(dim=0)
                        
                        

                    # 计算距离进行重排序
                    distances = torch.cdist(new_means_3, new_means_2, p=self.margin).detach() # num_of_fg * num_of_classes
                    indices = linear_sum_assignment(distances)[1] # [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                    
                    # 对means向量进行对齐
                    for i in range(self.train_cfg['num_unk_proto']):
                        new_means_adaptive[self.num_classes + i - 1] = new_means_2[indices[i]]

                    self.new_means_adaptive = new_means_adaptive  # [self.num_classes - 1 : ]  # 更新自适应原型特征
                    
                # Update the MUs
                for i, mean in enumerate(self.means):
                    if(mean) is not None and self.new_means_adaptive[i] is not None:
                        self.means[i] = self.bbox_assigner.momentum * mean + \
                                        (1 - self.bbox_assigner.momentum) * self.new_means_adaptive[i]
                                        
                
                # 分配新未知类别，判断mean是否为有意义
                for j, unknown_mean in enumerate(self.means[self.num_classes - 1 : ]):
                    if unknown_mean is not None:
                        self.unknown_means[j] = unknown_mean
                    else:
                        self.unknown_means[j] =  self.new_means_adaptive[j + self.num_classes - 1]
                a = self.normalize(input_features[torch.where(proposals >= self.num_classes - 1)]).cuda().detach()
                b = self.normalize(torch.stack(self.unknown_means)).cuda().detach()
                    
        
                cos = 1 - torch.mm(a, b.permute(1, 0))
                if cos.shape[0] > 0:

                    # cos2 = cos[:, self.train_cfg['num_classes'] - 1:] # proposal对于对于最后几类的距离

                    cos_max, cos_id_max = torch.max(cos, dim=1) # 得到与潜在类别特征相似性最高的unknown类别
                    # cos_id_max_new = gt_classes_before_assign.clone().detach()

                    # 为新unknown类赋类别的值
                    j_count = 0
                    for j in range(proposals.shape[0]):
                        if proposals[j] >= self.num_classes - 1: 
                            proposals[j] = cos_id_max[j_count] + self.num_classes - 1
                            j_count += 1
                            # cos_id_max_new[i] = self.train_cfg['num_classes'] + cos_id_max[i] - 1
                else:
                    print('***************')
                                        
            else:
                a = self.normalize(input_features[torch.where(proposals >= self.num_classes - 1)]).cuda().detach()
                b = self.normalize(torch.stack(self.unknown_means)).cuda().detach()
                cos = 1 - torch.mm(a, b.permute(1, 0))
                if cos.shape[0] > 0:

                    # cos2 = cos[:, self.train_cfg['num_classes'] - 1:] # proposal对于对于最后几类的距离

                    cos_max, cos_id_max = torch.max(cos, dim=1) # 得到与潜在类别特征相似性最高的unknown类别
                    # cos_id_max_new = gt_classes_before_assign.clone().detach()

                    # 为新unknown类赋类别的值
                    j_count = 0
                    for j in range(proposals.shape[0]):
                        if proposals[j] >= self.num_classes - 1: 
                            proposals[j] = cos_id_max[j_count] + self.num_classes - 1
                            j_count += 1
                            # cos_id_max_new[i] = self.train_cfg['num_classes'] + cos_id_max[i] - 1
                else:
                    print('***************')

            # c_loss = self.clstr_loss_l2_cdist(input_features, proposals)
            c_loss = self.clstr_loss_l2_cdist_new(input_features, proposals)
            # print(c_loss)


        # loss_weight_contra 改变
        loss_weight_contra = self.bbox_assigner.loss_weight_contra

        return c_loss * loss_weight_contra   

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training.

        Args:
            x (list[Tensor]): list of multi-level img features.
            sampling_results (list[Tensor]): list of sampling results.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        rois = rbbox2roi([res.bboxes for res in sampling_results])
        # 对feature进行增强
        bbox_results = self._bbox_forward_attention(x, rois)
        # bbox_results 里有所有pos， unknown和neg的feats特征
        trans_feature = bbox_results['trans_feature']       

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                bbox_results['bbox_pred'], rois,
                                *bbox_targets)
        
        # if not losses['loss_bbox'] < 10:
        #     has_nan = torch.isnan(bbox_targets[pos_inds.type(torch.bool)]).any()
        #     # print(has_nan)    
        #     losses['loss_bbox'] = torch.zeros(1).to('cuda')
        
        # Ours adaptively unknown classes method
        if self.train_cfg.assigner['unknown_contra']:
            # 将pos和unknown类别的feature找出来，从而进行对比聚类
            input_features, proposal_labels, pos_ind = self.obtain_contra_feats(trans_feature, sampling_results)
            
            if self.train_cfg.assigner['unknown_linear']:
                input_features = self.nnlinear1(input_features)
                input_features = self.nnlinear2(input_features)
                input_features = self.nnlinear3(input_features)

            input_features, proposal_labels = self.update_feature_store(input_features, proposal_labels)
            # input_features, proposal_labels, distances_mask = self.update_feature_store_new(input_features, proposal_labels)
            
            # 加上对比聚类的loss
            # loss_contra = self.get_clustering_loss(input_features, proposal_labels)
            loss_contra = self.get_adaptive_clustering_loss(input_features, proposal_labels)
            
        if self.train_cfg.assigner['text_super']:
            # class_name = ['plane', 'ship', 'storage_tank', 'baseball_diamond', 'basketball_court', 'ground_track_field', 'harbor', 'bridge', 'large_vehicle', 'small_vehicle', 'roundabout',\
            # 'helicopter', 'tennis_court',  'soccer_ball_field',  'swimming_pool']
            text_features = self.text_feature.to('cuda').transpose(0, 1)
            input_features, proposal_labels, pos_ind = self.obtain_contra_feats(trans_feature, sampling_results)
            
            # 保存对齐后的text feature
            # path = 'text_supervised_feature/text_feature_align_text0.02_mlp_from_ckpt.pth'
            # if self.train_iter % 1000 == 0:
            #     torch.save(text_feature_align, path)
                
                # a = torch.load(path)
                
            # unknown feature的对齐 用处好像不大？
            # uk_feature = torch.mean(text_feature_align[self.num_classes:,:],dim=0).unsqueeze(0)
            # text_feature_align = text_feature_align[:self.num_classes - 1, :]
            # text_feature_align = torch.cat((text_feature_align, uk_feature), dim=0)
            # proposal_labels = proposal_labels.long()
            text_feature_align = self.text_project(text_features)
            text_feature_align = self.text_project3(self.text_project2(text_feature_align))           
            text_feature_align = text_feature_align[:self.num_classes - 1, :]
            keep = torch.where(proposal_labels != self.num_classes - 1)[0]
            input_features = input_features[keep]
            proposal_labels = proposal_labels[keep].long()
            input_features = F.normalize(input_features, dim=1)
            text_feature_align = F.normalize(text_feature_align, dim=1)
            res = torch.matmul(input_features, text_feature_align.transpose(-1, -2))
            _, a, _, _ = bbox_targets
            a = a[:proposal_labels.shape[0]]
            # 出现nan情况
            if res.shape[0] > 0:
                loss_text = self.text_loss(
                        res,
                        proposal_labels,
                        a,
                        avg_factor=proposal_labels.shape[0],
                        reduction_override=None) * 0.02
            # if res.shape[0] > 0 and (not (0 <= loss_text <=10)):
            #     print(res)
            # loss_text = self.clstr_loss_l2_cdist_new(input_features, proposals)
            
            def cal_loss_vae(input_features,proposal_labels, device):
                x_visual = input_features
                x_text = torch.index_select(text_features, 0, proposal_labels)
                x_visual_recon, mean_visual, logvar_visual, x_text_recon, mean_text, logvar_text, out_v, out_s = self.dualvae(x_visual, x_text)
                loss_visual = loss_function_vae(x_visual_recon, x_visual, mean_visual, logvar_visual)
                loss_text = loss_function_vae(x_text_recon, x_text, mean_text, logvar_text)
                distance = torch.sqrt(torch.sum((mean_visual - mean_text) ** 2, dim=1) + \
                                        torch.sum((torch.sqrt(logvar_visual.exp()) - torch.sqrt(logvar_text.exp())) ** 2, dim=1))
                distance = distance.sum()       
                shuffle_classification_criterion = nn.NLLLoss()
                # cal_label = torch.zeros(self.bbox_head.num_classes+3, device=device)
                cal_label = proposal_labels
                loss_v = shuffle_classification_criterion(out_v, cal_label)
                loss_s = shuffle_classification_criterion(out_s, cal_label)
                classification_loss = loss_v + loss_s
                loss = loss_visual + loss_text + distance*1 + classification_loss*1
                
                return loss / input_features.shape[0]
            
            # # 采用VAE进行特征对齐
            # keep = torch.where(proposal_labels != self.num_classes - 1)[0]
            # input_features = input_features[keep]
            # proposal_labels = proposal_labels[keep].long()    
            # res = torch.matmul(input_features, text_feature_align.transpose(-1, -2))
            # if res.shape[0] > 0:
            #     device = input_features.device
            #     loss_vae = cal_loss_vae(input_features,proposal_labels, device)
            #     # loss_vae /= input_features.shape[0]
                    
        if self.train_cfg.assigner['unknown_contra']:
            loss_bbox['loss_contra'] = loss_contra * 0.1
            
        if self.train_cfg.assigner['text_super'] and res.shape[0] > 0:
            loss_bbox['loss_text'] = loss_text
            # loss_bbox['loss_vae'] = loss_vae * 0.01
        self.train_iter += 1         # iter 数量增加
        bbox_results.update(loss_bbox=loss_bbox)

        return bbox_results

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains \
                the boxes of the corresponding image in a batch, each \
                tensor has the shape (num_boxes, 5) and last dimension \
                5 represent (cx, cy, w, h, a, score). Each Tensor \
                in the second list is the labels with shape (num_boxes, ). \
                The length of both lists should be equal to batch_size.
        """

        rois = rbbox2roi(proposals)
        bbox_results = self._bbox_forward_attention(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        cls_score2 = cls_score
        
        # # # DIORDataset 16+4
        # # # self.text_feature = torch.load(self.text_super).transpose(0, 1)
        # self.text_feature = torch.load("/mnt/disk2/yuanzm/weights/valid5-dior_aug.pth").transpose(0, 1)
        # text_feature = self.text_feature.to('cuda').transpose(0, 1)
        # # text_feature_align = text_feature
        # text_feature_align = self.text_project(text_feature)
        # text_feature_align = self.text_project3(self.text_project2(text_feature_align))
        # input_features = F.normalize(bbox_results['trans_feature'], dim=1)
        # text_feature_align = F.normalize(text_feature_align, dim=1)
        # res = torch.matmul(input_features, text_feature_align.transpose(-1, -2))
        # accs, counts = per_class_accuracy(cls_score, res)
        # self.accum_acc += accs
        # self.accum_cnt += counts
        # self.num_iter += 1
        # softmax_layer = nn.Softmax(dim = 1)
        # unknown_res = softmax_layer(res[:,16:])
        # # max_u_idx = torch.arange(cls_score.shape[0])[torch.argmax(cls_score, dim=1)==16]
        # # print(unknown_res[max_u_idx,:])
        # max_idx_res = torch.argmax(unknown_res, dim = 1)
        # cls_score2 = torch.zeros([cls_score.shape[0], 21]).to('cuda')
        # cls_score2[:,:16] =  cls_score[:,:16]
        # cls_score2[torch.arange(cls_score.shape[0]), max_idx_res + 16] = cls_score[:, 16]
        # cls_score2[:,-1] =  cls_score[:,17]
        # self.bbox_head.num_classes = 20

        # NWPUDataset 6+4
        self.text_feature = torch.load(self.text_super).transpose(0, 1) #valid5-dota_dsp
        text_feature = self.text_feature.to('cuda').transpose(0, 1)
        # cos sim测试方法
        text_feature_align = self.text_project(text_feature)
        text_feature_align = self.text_project3(self.text_project2(text_feature_align))
        input_features = F.normalize(bbox_results['trans_feature'], dim=1)
        text_feature_align = F.normalize(text_feature_align, dim=1)
        res = torch.matmul(input_features, text_feature_align.transpose(-1, -2))
        softmax_layer = nn.Softmax(dim = 1)
        unknown_res = softmax_layer(res[:,6:])
        # accs, counts = per_class_accuracy(cls_score, res)
        # self.accum_acc += accs
        # self.accum_cnt += counts
        # self.num_iter += 1
        max_idx_res = torch.argmax(unknown_res, dim = 1)
        cls_score2 = torch.zeros([cls_score.shape[0], 11]).to('cuda')
        cls_score2[:,:6] =  cls_score[:,:6]
        cls_score2[torch.arange(cls_score.shape[0]), max_idx_res + 6] = cls_score[:, 6]
        cls_score2[:,10] =  cls_score[:,7]
        self.bbox_head.num_classes = 10

        # # DOTADataset 11+4
        # self.text_feature = torch.load(self.text_super).transpose(0, 1) #valid5-dota_dsp
        # text_feature = self.text_feature.to('cuda').transpose(0, 1)
        # # # cos sim测试方法
        # # text_feature_align = self.text_project(text_feature)
        # # text_feature_align = self.text_project3(self.text_project2(text_feature_align))
        # # # text_feature_align = text_feature
        # # # input_features = self.img_project(bbox_results['trans_feature'])
        # # # input_features = self.img_project3(self.img_project2(input_features))
        # # input_features = F.normalize(bbox_results['trans_feature'], dim=1)
        # # text_feature_align = F.normalize(text_feature_align, dim=1)
        # # vae测试方法
        # self.dualvae.reparameterize_with_noise = False
        # input_features, text_feature_align = self.dualvae(bbox_results['trans_feature'], text_feature)
        # self.dualvae.reparameterize_with_noise = True
        # res = torch.matmul(input_features, text_feature_align.transpose(-1, -2))
        # softmax_layer = nn.Softmax(dim = 1)
        # unknown_res = softmax_layer(res[:,11:])
        # # accs, counts = per_class_accuracy(cls_score, res)
        # # self.accum_acc += accs
        # # self.accum_cnt += counts
        # # self.num_iter += 1
        # # max_u_idx = torch.arange(cls_score.shape[0])[torch.argmax(cls_score, dim=1)==11]
        # # print(unknown_res[max_u_idx,:])
        # max_idx_res = torch.argmax(unknown_res, dim = 1)
        # range_scores = torch.max(unknown_res, dim=1).values - torch.min(unknown_res, dim=1).values 
        # # idx_list = torch.where(range_scores <= 0.015)[0]
        # # max_idx_res[idx_list] = -1
        # #pdb.set_trace()
        # cls_score2 = torch.zeros([cls_score.shape[0], 16]).to('cuda')
        # cls_score2[:,:11] =  cls_score[:,:11]
        # cls_score2[torch.arange(cls_score.shape[0]), max_idx_res + 11] = cls_score[:, 11]
        # cls_score2[:,15] =  cls_score[:,12]
        # self.bbox_head.num_classes = 15
        
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score2 = cls_score2.split(num_proposals_per_img, 0)
        
        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        # det_features = []
        # trans_feature = bbox_results['trans_feature'][:, None].expand(
        #     bbox_results['trans_feature'].size(0), self.bbox_head.num_classes, 1024).reshape(-1, 1024)
        for i in range(len(proposals)):
            det_bbox, det_label, ind, keep = self.bbox_head.get_bboxes(
                rois[i],
                cls_score2[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg,
                return_ind=True)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            # det_feature = trans_feature[ind][keep]
            # 利用torch.unique和torch.where进行分组操作
            # unique_labels = torch.unique(det_label)
            # for label in unique_labels:
            #     mask = det_label == label
            #     if len(self.cls_trans_feature[label.item()]) < 1000:
            #         self.cls_trans_feature[label.item()].extend(det_feature[mask])
        return det_bboxes, det_labels
    
    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposal_list (list[Tensors]): list of region proposals.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            rbbox2result(det_bboxes[i], det_labels[i],
                         self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]
        self.bbox_head.num_classes -= 3
        return bbox_results

