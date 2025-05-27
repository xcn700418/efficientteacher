#Copyright (c) 2023, Alibaba Group
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel
# from scipy.optimize import linear_sum_assignment
import math
from torch.nn import functional as F
from torch.autograd import Variable,Function
import numpy as np
from models.module.nanodet_utils import generate_anchors, make_anchors
# from loss.yolox_loss import pairwise_bbox_iou
from utils.general import xywh2xyxy, box_iou
from assigner import YOLOAnchorAssigner,TaskAlignedAssigner
from loss.gfocal_loss import BboxLoss, VarifocalLoss
import logging
LOGGER = logging.getLogger(__name__)
# from loss import BboxLoss
# from tal import dist2bbox
# from torch.cuda.amp import autocast

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

def xywh2xyxy(bboxes):
    '''Transform bbox(xywh) to box(xyxy).'''
    bboxes[..., 0] = bboxes[..., 0] - bboxes[..., 2] * 0.5
    bboxes[..., 1] = bboxes[..., 1] - bboxes[..., 3] * 0.5
    bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2]
    bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3]
    return bboxes



def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

# for label match type semi spervised training
class ComputeStudentMatchLoss():
    # Compute losses
    def __init__(self, model, cfg):
        super(ComputeStudentMatchLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        #h = model.hyp  # hyperparameters
        autobalance = cfg.Loss.autobalance
        cls_pw = cfg.Loss.cls_pw
        obj_pw = cfg.Loss.obj_pw
        label_smoothing = cfg.Loss.label_smoothing

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cls_pw], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([obj_pw], device=device))

        if cfg.SSOD.focal_loss > 0:
            BCEobj = FocalLoss(BCEobj)

        self.cp, self.cn = smooth_BCE(eps=label_smoothing)  # positive, negative BCE targets

        det = model.module.head if is_parallel(model) else model.head # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr,  self.autobalance = BCEcls, BCEobj, 1.0, autobalance
        self.box_w = cfg.SSOD.box_loss_weight
        self.obj_w = cfg.SSOD.obj_loss_weight
        # self.cls_w = cfg.SSOD.cls_loss_weight
        self.cls_w = cfg.SSOD.cls_loss_weight * cfg.Dataset.nc / 80. * 3. / det.nl
        self.anchor_t = cfg.Loss.anchor_t
        # self.ignore_thres = cfg.SSOD.ignore_thres
        self.ignore_thres_high = [cfg.SSOD.ignore_thres_high] * cfg.Dataset.nc
        self.ignore_thres_low = [cfg.SSOD.ignore_thres_low] * cfg.Dataset.nc
        self.uncertain_aug = cfg.SSOD.uncertain_aug
        self.use_ota = cfg.SSOD.use_ota
        self.ignore_obj = cfg.SSOD.ignore_obj
        self.pseudo_label_with_obj = cfg.SSOD.pseudo_label_with_obj
        self.pseudo_label_with_bbox = cfg.SSOD.pseudo_label_with_bbox
        self.pseudo_label_with_cls = cfg.SSOD.pseudo_label_with_cls
        self.num_keypoints = cfg.Dataset.np
        self.single_targets = False
        if not self.uncertain_aug:
            self.single_targets = True

        for k in 'na', 'nc', 'nl', 'anchors', 'stride':
            setattr(self, k, getattr(det, k))
        self.assigner = YOLOAnchorAssigner(self.na, self.nl, self.anchors, self.anchor_t, det.stride, \
            self.nc, self.num_keypoints, single_targets=self.single_targets, ota=self.use_ota)


    def select_targets(self, targets):
        '''
        targets: [batch, classes, x, y, x, y, conf, obj_conf, cls_conf]
        '''
        device = targets.device
        reliable_targets = []
        uncertain_targets = []
        uncertain_obj_targets = []
        uncertain_cls_targets = []
        for t in targets:
            #伪标签得分大于相应类别的阈值,标记为正样本
            t = np.array(t.cpu())
            # original logic
            if t[6] >= self.ignore_thres_high[int(t[1])]:
                    reliable_targets.append(t[:7])
            # if t[6] >= self.ignore_thres_high[int(t[1])]:
            #     # if t[7] >= self.ignore_thres_high[int(t[1])] and t[8] >= self.ignore_thres_high[int(t[1])]:
            #     if t[8] >= 0.99:
            #         reliable_targets.append(t[:7])
            #     else: #如果obj和cls中其中一个小于阈值, 标记为uncertain 
            #         if self.pseudo_label_with_obj:
            #             uncertain_targets.append(np.concatenate((t[:6], t[7:8])))
            #             if t[7] > 0.99:
            #                 uncertain_obj_targets.append(np.concatenate((t[:6], t[7:8])))
            #         else:
            #             uncertain_targets.append(t[:7])
            #伪标签低阈值和高阈值之间的，标记为不确定样本
            elif t[6] >= self.ignore_thres_low[int(t[1])]:
                if self.pseudo_label_with_obj:
                    uncertain_targets.append(np.concatenate((t[:6], t[7:8])))
                    #不确定样本里面obj特别高的，送出来修iou loss
                    if t[7] >= 0.99:
                        uncertain_obj_targets.append(np.concatenate((t[:6], t[7:8])))
                    #不确定样本里cls特别高的，送出来修cls loss
                    if t[8] >= 0.99:
                        uncertain_cls_targets.append(np.concatenate((t[:6], t[7:8])))
                else:
                    uncertain_targets.append(t[:7])

        reliable_targets = np.array(reliable_targets).astype(np.float32)
        reliable_targets= torch.from_numpy(reliable_targets).contiguous()
        reliable_targets = reliable_targets.to(device)
        if reliable_targets.shape[0] == 0:
            reliable_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6])[False].to(device)

        uncertain_targets = np.array(uncertain_targets).astype(np.float32)
        uncertain_targets= torch.from_numpy(uncertain_targets).contiguous()
        uncertain_targets= uncertain_targets.to(device)
        if uncertain_targets.shape[0] == 0:
            uncertain_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6])[False].to(device)

        uncertain_obj_targets = np.array(uncertain_obj_targets).astype(np.float32)
        uncertain_obj_targets= torch.from_numpy(uncertain_obj_targets).contiguous()
        uncertain_obj_targets= uncertain_obj_targets.to(device)
        if uncertain_obj_targets.shape[0] == 0:
            uncertain_obj_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6])[False].to(device)

        uncertain_cls_targets = np.array(uncertain_cls_targets).astype(np.float32)
        uncertain_cls_targets= torch.from_numpy(uncertain_cls_targets).contiguous()
        uncertain_cls_targets= uncertain_cls_targets.to(device)
        if uncertain_cls_targets.shape[0] == 0:
            uncertain_cls_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6])[False].to(device)
        return reliable_targets, uncertain_targets, uncertain_obj_targets, uncertain_cls_targets

    def default_loss(self, p, targets):
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)        
        if targets.shape[1] > 6:
            certain_targets, uc_targets, uc_obj_targets, uc_cls_targets = self.select_targets(targets)
            if self.uncertain_aug:
                tcls, tbox, indices, anchors  = self.assigner(p, certain_targets)  # targets
                _, _, uc_indices, _, uc_scores = self.assigner(p, uc_targets, with_pseudo_score=True)
                _, uc_tbox, uc_obj_indices, uc_anchors, _ = self.assigner(p, uc_obj_targets, with_pseudo_score=True)
                uc_tcls, _, uc_cls_indices, _, _ = self.assigner(p, uc_cls_targets, with_pseudo_score=True)
            else:
                tcls, tbox, indices, anchors  = self.assigner(p, certain_targets)  # targets
                _, _, uc_indices, _, uc_scores = self.assigner(p, uc_targets, with_pseudo_score=True)
                _, uc_tbox, uc_obj_indices, uc_anchors, _ = self.assigner(p, uc_obj_targets, with_pseudo_score=True)                
                uc_tcls, _, uc_cls_indices, _, _ = self.assigner(p, uc_cls_targets, with_pseudo_score=True)
        else:
            tcls, tbox, indices, anchors  = self.assigner(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            anchors_spec = anchors[i] 
            tbox_spec = tbox[i]
            n = b.shape[0]  # number of targets

            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors_spec
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox_spec, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
                # tobj[b, a, gj, gi] = 1.0

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

            if targets.shape[1] > 6:
                #uncertain label cal obj loss
                uc_b, uc_a, uc_gj, uc_gi = uc_indices[i]
                n = uc_b.shape[0]
                if n:
                    if self.ignore_obj:
                        tobj[uc_b, uc_a, uc_gj, uc_gi] = -1 #ignore region set -1
                    else:
                        tobj[uc_b, uc_a, uc_gj, uc_gi] = uc_scores[i].type(tobj.dtype)  #ignore region set -1

                if self.pseudo_label_with_bbox:
                    #uncertain label cal iou loss
                    uc_obj_b, uc_obj_a, uc_obj_gj, uc_obj_gi = uc_obj_indices[i]
                    n = uc_obj_b.shape[0]
                    uc_tbox_spec = uc_tbox[i]
                    anchors_spec = uc_anchors[i]
                    if n:
                        uc_ps = pi[uc_obj_b, uc_obj_a, uc_obj_gj, uc_obj_gi]
                        pxy = uc_ps[:, :2].sigmoid() * 2. - 0.5
                        pwh = (uc_ps[:, 2:4].sigmoid() * 2) ** 2 * anchors_spec
                        pbox = torch.cat((pxy, pwh), 1)  # predicted box
                        iou = bbox_iou(pbox.T, uc_tbox_spec, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                        lbox += (1.0 - iou).mean()  # iou loss
                        # tobj[uc_obj_b, uc_obj_a, uc_obj_gj, uc_obj_gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
                
                if self.pseudo_label_with_cls:
                    #uncertain label cal cls loss
                    uc_cls_b, uc_cls_a, uc_cls_gj, uc_cls_gi = uc_cls_indices[i]
                    n = uc_cls_b.shape[0]
                    if n:
                        uc_ps = pi[uc_cls_b, uc_cls_a, uc_cls_gj, uc_cls_gi]
                        if self.nc > 1:  # cls loss (only if multiple classes)
                            t = torch.full_like(uc_ps[:, 5:], self.cn, device=device)  # targets
                            t[range(n), uc_tcls[i]] = self.cp
                            lcls += self.BCEcls(uc_ps[:, 5:], t)  # BCE
            # filtering ignore region, only cal gradient on foreground and background
            valid_mask = tobj >= 0
            obji = self.BCEobj(pi[..., 4][valid_mask], tobj[valid_mask])
            lobj += obji * self.balance[i]  # obj loss


        lbox *= self.box_w
        lobj *= self.obj_w
        lcls *= self.cls_w
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        loss_dict = dict(ss_box = lbox, ss_obj = lobj, ss_cls = lcls)
        return loss * bs, loss_dict

    def __call__(self, p, targets):
        if self.use_ota == False:
            loss, loss_dict = self.default_loss(p, targets)
        else:
            loss, loss_dict = self.ota_loss(p, targets)
        return loss, loss_dict
    
    def ota_loss(self, p, targets):
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        if targets.shape[1] > 6:
            reliable_targets, uc_targets, uc_obj_targets, uc_cls_targets = self.select_targets(targets)
            bs, as_, gjs, gis, reliable_targets, anchors, tscores = self.assigner(p, reliable_targets, with_pseudo_scores=True)
            uc_bs, uc_as_, uc_gjs, uc_gis, uc_targets, uc_anchors, uc_tscores = self.assigner(p, uc_targets, with_pseudo_scores=True)
            pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p] 
    
            # Losses
            for i, pi in enumerate(p):  # layer index, layer predictions
                b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
                tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

                n = b.shape[0]  # number of targets
                if n:
                    ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                    # Regression
                    grid = torch.stack([gi, gj], dim=1)
                    pxy = ps[:, :2].sigmoid() * 2. - 0.5
                    #pxy = ps[:, :2].sigmoid() * 3. - 1.
                    pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                    pbox = torch.cat((pxy, pwh), 1)  # predicted box
                    selected_tbox = reliable_targets[i][:, 2:6] * pre_gen_gains[i]
                    selected_tbox[:, :2] -= grid
                    iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                    lbox += (1.0 - iou).mean()  # iou loss

                    # Objectness
                    tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                    # Classification
                    selected_tcls = reliable_targets[i][:, 1].long()
                    if self.nc > 1:  # cls loss (only if multiple classes)
                        t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                        t[range(n), selected_tcls] = self.cp
                        lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                uc_b, uc_a, uc_gj, uc_gi = uc_bs[i], uc_as_[i], uc_gjs[i], uc_gis[i]
                n = uc_b.shape[0]
                if n:
                    if self.ignore_obj:
                        tobj[uc_b, uc_a, uc_gj, uc_gi] = -1
                    else:
                        tobj[uc_b, uc_a, uc_gj, uc_gi] = uc_tscores[i].type(tobj.dtype) 
                valid_mask = tobj >= 0
                obji = self.BCEobj(pi[..., 4][valid_mask], tobj[valid_mask])
                lobj += obji * self.balance[i]  # obj loss
                if self.autobalance:
                    self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        else:
            bs, as_, gjs, gis, targets, anchors = self.assigner(p, targets)
            pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p] 
    
            # Losses
            for i, pi in enumerate(p):  # layer index, layer predictions
                b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
                tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

                n = b.shape[0]  # number of targets
                if n:
                    ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                    # Regression
                    grid = torch.stack([gi, gj], dim=1)
                    pxy = ps[:, :2].sigmoid() * 2. - 0.5
                    #pxy = ps[:, :2].sigmoid() * 3. - 1.
                    pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                    pbox = torch.cat((pxy, pwh), 1)  # predicted box
                    selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                    selected_tbox[:, :2] -= grid
                    iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                    lbox += (1.0 - iou).mean()  # iou loss

                    # Objectness
                    tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                    # Classification
                    selected_tcls = targets[i][:, 1].long()
                    if self.nc > 1:  # cls loss (only if multiple classes)
                        t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                        t[range(n), selected_tcls] = self.cp
                        lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                obji = self.BCEobj(pi[..., 4], tobj)
                lobj += obji * self.balance[i]  # obj loss
                if self.autobalance:
                    self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.box_w
        lobj *= self.obj_w
        lcls *= self.cls_w
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        loss_dict = dict(ss_box = lbox, ss_obj = lobj, ss_cls = lcls)
        return loss * bs, loss_dict



'''
SSOD loss for yolov8 is different with yolov5 model since it is an anchor free model
'''
class ComputeStudentMatchLossV8():
    # Compute losses
    def __init__(self, model, cfg):
        super(ComputeStudentMatchLossV8, self).__init__()
        device = next(model.parameters()).device  # get model device
        self.device = device
        # YOLOv8 hyper-parameters
        autobalance = cfg.Loss.autobalance
        cls_pw = cfg.Loss.cls_pw # cls positive weights
        label_smoothing = cfg.Loss.label_smoothing
        # Define criteria
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        #BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cls_pw], device=device))
        # BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([obj_pw], device=device))
        self.cp, self.cn = smooth_BCE(eps=label_smoothing)  # positive, negative BCE targets

        det = model.module.head if is_parallel(model) else model.head # Detect() module
        self.stride = det.stride
        self.nc = det.nc
        self.no = det.no
        self.reg_max = det.reg_max
        self.use_dfl = self.reg_max > 1
        self.use_gfl = cfg.Loss.use_gfl
        self.fpn_strides = cfg.Model.Head.strides
        self.grid_cell_size = cfg.Loss.grid_cell_size
        self.grid_cell_offset = cfg.Loss.grid_cell_offset
        # self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        # self.ssi = list(self.stride).index(16) if autobalance else 0  # stride 16 index
        # self.BCEcls, self.gr,  self.autobalance = BCEcls, 1.0, autobalance
        self.ori_img_size = cfg.Dataset.img_size

        for k in 'nc', 'nl', 'stride':
            setattr(self, k, getattr(det, k))
            
        # self.assigner = YOLOAnchorFreeAssigner(self.na, self.nl, self.anchors, self.anchor_t, det.stride, \
        #     self.nc, self.num_keypoints, single_targets=self.single_targets, ota=self.use_ota)
        self.assigner = TaskAlignedAssigner(topk=13, num_classes=self.nc, alpha=1.0, beta=6.0, eps=1e-9)
        self.bbox_loss = BboxLoss(self.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max), requires_grad=False)
        self.varifocal_loss = VarifocalLoss().cuda()
        # SSOD hyper-parameters
        self.box_w = cfg.SSOD.box_loss_weight
        self.obj_w = cfg.SSOD.obj_loss_weight
        self.dfl_w = cfg.SSOD.dfl_loss_weight
        self.loss_weight = {'class': cfg.Loss.qfl_loss_weight, 'iou': cfg.Loss.box_loss_weight, 'dfl':cfg.Loss.dfl_loss_weight} 
        # self.cls_w = cfg.SSOD.cls_loss_weight
        self.cls_w = cfg.SSOD.cls_loss_weight * cfg.Dataset.nc / 80. * 3. / det.nl
        self.anchor_t = cfg.Loss.anchor_t
        # self.ignore_thres = cfg.SSOD.ignore_thres
        self.ignore_thres_high = [cfg.SSOD.ignore_thres_high] * cfg.Dataset.nc
        self.ignore_thres_low = [cfg.SSOD.ignore_thres_low] * cfg.Dataset.nc
        self.uncertain_aug = cfg.SSOD.uncertain_aug
        self.use_ota = cfg.SSOD.use_ota
        self.ignore_obj = cfg.SSOD.ignore_obj
        self.pseudo_label_with_obj = cfg.SSOD.pseudo_label_with_obj
        self.pseudo_label_with_bbox = cfg.SSOD.pseudo_label_with_bbox
        self.pseudo_label_with_cls = cfg.SSOD.pseudo_label_with_cls

        self.single_targets = False
        if not self.uncertain_aug:
            self.single_targets = True


    def preprocess(self, targets, batch_size, scale_tensor):
        targets_list = np.zeros((batch_size, 1, 5)).tolist()
        for i, item in enumerate(targets.cpu().numpy().tolist()):
            targets_list[int(item[0])].append(item[1:])
        max_len = max((len(l) for l in targets_list))
        num_gts = 0
        for l in targets_list:
            num_gts += len(l)
        targets = torch.from_numpy(np.array(list(map(lambda l:l + [[-1,0,0,0,0]]*(max_len - len(l)), targets_list)))[:,1:,:]).to(targets.device)
        batch_target = targets[:, :, 1:5].mul_(scale_tensor)
        targets[..., 1:] = xywh2xyxy(batch_target)
        return targets, num_gts
    
    def select_preprocess(self, targets, batch_size, scale_tensor):
        targets_list = np.zeros((batch_size, 1, 5)).tolist()
        for i, item in enumerate(targets.cpu().numpy().tolist()):
            targets_list[int(item[0])].append(item[1:6])
        max_len = max((len(l) for l in targets_list))
        num_gts = 0
        for l in targets_list:
            num_gts += len(l)
        # targets = torch.from_numpy(np.array(list(map(lambda l:l + [[-1,0,0,0,0]]*(max_len - len(l)), targets_list)))[:,1:,:]).to(targets.device)
        targets = torch.from_numpy(np.array(list(map(lambda l:l + [[-1,0,0,0,0]]*(max_len - len(l)), targets_list)))[:,1:,:]).to(targets.device)
        batch_target = targets[:, :, 1:5].mul_(scale_tensor)
        # batch_target = targets[:, 2:6].mul_(scale_tensor)
        targets[..., 1:] = xywh2xyxy(batch_target)
        # targets[..., 2:] = xywh2xyxy(batch_target)
        return targets, num_gts

    
    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            batch_size, n_anchors, _ = pred_dist.shape
            pred_dist = F.softmax(pred_dist.view(batch_size, n_anchors, 4, self.reg_max), dim=-1).matmul(self.proj.to(pred_dist.device))
        return dist2bbox(pred_dist, anchor_points)
    
    def select_targets(self, targets):
        '''
        targets: [image, num, classes, x, y, x, y, conf, obj_conf, cls_conf]
        '''
        '''
        targets: [image, num, classes, x, y, x, y, conf, obj_conf, cls_conf]
        '''
        # DISABLE obj loss in YOLOv8
        device = targets.device
        reliable_targets = []
        uncertain_targets = []
        uncertain_obj_targets = []
        uncertain_cls_targets = []
        
        for t in targets:
            #伪标签得分大于相应类别的阈值,标记为正样本
            t = np.array(t.cpu())
            # original logic
            if t[6] >= self.ignore_thres_high[int(t[1])]:
                    reliable_targets.append(t[:7])

            #伪标签低阈值和高阈值之间的，标记为不确定样本
            elif t[6] >= self.ignore_thres_low[int(t[1])]:
                if self.pseudo_label_with_obj:
                    uncertain_targets.append(np.concatenate((t[:6], t[8:9])))
                    #不确定样本里面obj特别高的，送出来修iou loss
                    # if t[7] >= 0.99:
                    #     uncertain_obj_targets.append(np.concatenate((t[:6], t[7:8])))
                    #不确定样本里cls特别高的，送出来修cls loss
                    if t[8] >= 0.99:
                        uncertain_cls_targets.append(np.concatenate((t[:6], t[8:9])))
                else:
                    uncertain_targets.append(t[:7])

        reliable_targets = np.array(reliable_targets).astype(np.float32)
        reliable_targets= torch.from_numpy(reliable_targets).contiguous()
        reliable_targets = reliable_targets.to(device)
        if reliable_targets.shape[0] == 0:
            reliable_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])[False].to(device)

        uncertain_targets = np.array(uncertain_targets).astype(np.float32)
        uncertain_targets= torch.from_numpy(uncertain_targets).contiguous()
        uncertain_targets= uncertain_targets.to(device)
        if uncertain_targets.shape[0] == 0:
            uncertain_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])[False].to(device)

        uncertain_obj_targets = np.array(uncertain_obj_targets).astype(np.float32)
        uncertain_obj_targets= torch.from_numpy(uncertain_obj_targets).contiguous()
        uncertain_obj_targets= uncertain_obj_targets.to(device)
        if uncertain_obj_targets.shape[0] == 0:
            uncertain_obj_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])[False].to(device)

        uncertain_cls_targets = np.array(uncertain_cls_targets).astype(np.float32)
        uncertain_cls_targets= torch.from_numpy(uncertain_cls_targets).contiguous()
        uncertain_cls_targets= uncertain_cls_targets.to(device)
        if uncertain_cls_targets.shape[0] == 0:
            uncertain_cls_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])[False].to(device)
        # return reliable_targets, uncertain_targets, uncertain_cls_targets
        return reliable_targets, uncertain_targets, uncertain_obj_targets, uncertain_cls_targets


    def build_loss(self, p, targets):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        device = targets.device
        loss_cls, loss_iou, loss_dfl = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

        feats, pred_scores, pred_distri = p
        pred_scores = pred_scores.float()
        pred_distri = pred_distri.float()

        anchors, anchor_points, n_anchors_list, stride_tensor = \
            generate_anchors(feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset, device=feats[0].device)

        assert pred_scores.type() == pred_distri.type()
        batch_size = pred_scores.shape[0]

        # Pboxes
        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self.bbox_decode(anchor_points_s, pred_distri)  # xyxy, (b, h*w, 4)
        gt_bboxes_scale = torch.full((1,4), self.ori_img_size).type_as(pred_scores)


        if targets.shape[1] > 6:
            certain_targets, uc_targets, uc_obj_targets, uc_cls_targets = self.select_targets(targets)
            # targets: [image, num, classes, x, y, x, y, conf, obj_conf, cls_conf]
            # CERTAIN TARGETS
            # LOGGER.info(f"SSOD TARGETS : {certain_targets.shape}")
            c_targets, c_num_gts = self.select_preprocess(certain_targets, batch_size, gt_bboxes_scale)
            c_targets_labels = c_targets[:,:,:1]
            c_targets_bbox = c_targets[:,:,1:]
            c_targets_mask_gt = (c_targets_bbox.sum(-1, keepdim=True) > 0).float()
            LOGGER.info(f"c_num_gts: {c_num_gts}")

            c_tcls, c_tbox, c_tscores, c_fg_masks, indices, anchors = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(c_targets_bbox.dtype),
                anchor_points * stride_tensor,
                c_targets_labels,
                c_targets_bbox,
                c_targets_mask_gt,
                with_pseudo_score = True
            )

            # UNCERTAIN TARGETS
            uc_targets, uc_num_gts = self.select_preprocess(uc_targets, batch_size, gt_bboxes_scale)
            uc_targets_labels = uc_targets[:,:,:1]
            uc_targets_bbox = uc_targets[:,:,1:]
            uc_targets_mask_gt = (uc_targets_bbox.sum(-1, keepdim=True) > 0).float()

            _, _, uc_tscores , uc_fg_masks, uc_indices, _ = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(uc_targets_bbox.dtype),
                anchor_points * stride_tensor,
                uc_targets_labels,
                uc_targets_bbox,
                uc_targets_mask_gt,
                with_pseudo_score = True
            )
            LOGGER.info(f"uc_num_gts: {uc_num_gts}")

            # UNCERTAIN OBJ TARGETS
            uc_obj_targets, uc_obj_num_gts = self.select_preprocess(uc_obj_targets, batch_size, gt_bboxes_scale)
            uc_obj_targets_labels = uc_obj_targets[:,:,:1]
            uc_obj_targets_bbox = uc_obj_targets[:,:,1:]
            uc_obj_targets_mask_gt = (uc_obj_targets_bbox.sum(-1, keepdim=True) > 0).float()

            _, uc_tbox, _ , _, uc_obj_indices, uc_anchors = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(uc_obj_targets_bbox.dtype),
                anchor_points * stride_tensor,
                uc_obj_targets_labels,
                uc_obj_targets_bbox,
                uc_obj_targets_mask_gt,
                with_pseudo_score = True
            )
            LOGGER.info(f"uc_obj_num_gts: {uc_obj_num_gts}")

            # UNCERTAIN CLS TARGETS
            uc_cls_targets, uc_cls_num_gts = self.select_preprocess(uc_cls_targets, batch_size, gt_bboxes_scale)
            uc_cls_targets_labels = uc_cls_targets[:,:,:1]
            uc_cls_targets_bbox = uc_cls_targets[:,:,1:]
            uc_cls_targets_mask_gt = (uc_cls_targets_bbox.sum(-1, keepdim=True) > 0).float()

            uc_tcls, _, _ , _, uc_cls_indices, _ = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(uc_cls_targets_bbox.dtype),
                anchor_points * stride_tensor,
                uc_cls_targets_labels,
                uc_cls_targets_bbox,
                uc_cls_targets_mask_gt,
                with_pseudo_score = True
            )
            LOGGER.info(f"uc_cls_num_gts: {uc_cls_num_gts}")
            # torch.cuda.empty_cache()
            # calculate losses and combine
            # CLS LOSS
            if self.use_gfl:
                c_one_hot_label = F.one_hot(c_tcls.long(), self.nc + 1)[..., :-1]
                c_loss_cls = self.varifocal_loss(pred_scores, c_tscores, c_one_hot_label)
                uc_one_hot_label = F.one_hot(uc_tcls.long(), self.nc + 1)[..., :-1]
                uc_loss_cls = self.varifocal_loss(pred_scores, uc_tscores, uc_one_hot_label)
            else:
                # c_loss_cls = (F.binary_cross_entropy_with_logits(pred_scores.float(), c_tscores.float(), reduction='none')).sum()
                c_loss_cls = self.bce(pred_scores, c_tscores.float()).sum()
                if uc_num_gts > 0:
                    # uc_loss_cls = (F.binary_cross_entropy_with_logits(pred_scores.float(), uc_tscores.float(), reduction='none')).sum()
                    uc_loss_cls = self.bce(pred_scores, uc_tscores.float()).sum()
                else:
                    uc_loss_cls = 0
            
            torch.cuda.empty_cache()
            c_target_scores_sum = max(c_tscores.sum(), 1)
            uc_target_scores_sum = max(uc_tscores.sum(), 1)
            LOGGER.info(f"check c_loss_cls {c_loss_cls}")   
            LOGGER.info(f"check uc_loss_cls {uc_loss_cls}")  
            if c_target_scores_sum > 0:
                c_loss_cls /= c_target_scores_sum
            if uc_target_scores_sum > 0:
                uc_loss_cls /= uc_target_scores_sum
            loss_cls = 0.5*c_loss_cls + 0.5*uc_loss_cls # cls
            # BBOX & DFL LOSS

            c_loss_iou, c_loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points_s, c_tbox,
                                            c_tscores, c_target_scores_sum, c_fg_masks)
            uc_loss_iou, uc_loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points_s, uc_tbox,
                                            uc_tscores, uc_target_scores_sum, uc_fg_masks)
            loss_iou = 0.5*c_loss_iou + 0.5*uc_loss_iou # bbox/iou
            loss_dfl = 0.5*c_loss_dfl + 0.5*uc_loss_dfl # dfl
            LOGGER.info(f"check loss_iou {loss_iou}")  
            LOGGER.info(f"check loss_dfl {loss_dfl}")  

            loss = torch.zeros(1, device=feats[0].device) + self.loss_weight['class'] * loss_cls + \
                self.loss_weight['iou'] * loss_iou + \
                self.loss_weight['dfl'] * loss_dfl
            c_num_fg = torch.sum(c_fg_masks)/max(c_num_gts, 1)
            uc_num_fg = torch.sum(uc_fg_masks)/max(uc_num_gts, 1)
            loss_dict = dict(loss_iou = self.loss_weight['iou']*loss_iou, loss_dfl = self.loss_weight['dfl'] * loss_dfl,\
                loss_cls = self.loss_weight['class'] * loss_cls, loss=loss, num_fg = c_num_fg+uc_num_fg)
            return loss, loss_dict
        
        else:
            '''
            [0.00000e+00, 1.00000e+00, 9.49814e-01, 4.08333e-02, 1.00369e-01, 8.16665e-02],
            [0.00000e+00, 1.90000e+01, 1.84126e-02, 5.50206e-01, 3.68252e-02, 4.24350e-01],
            [0.00000e+00, 0.00000e+00, 5.27865e-01, 6.42639e-01, 9.44267e-01, 3.78133e-01],
            '''
            targets, num_gts = self.preprocess(targets,batch_size, gt_bboxes_scale)
            targets_labels = targets[:,:,:1]
            targets_bbox = targets[:,:,1:]
            target_mask_gt = (targets_bbox.sum(-1, keepdim=True) > 0).float()

            tcls, tbox, tscores ,fg_masks, indices, anchors = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(targets_bbox.dtype),
                anchor_points * stride_tensor,
                targets_labels,
                targets_bbox,
                target_mask_gt,
            )
            #torch.cuda.empty_cache()
            tbox /= stride_tensor
            # CALCULATE LOSSES box, cls, dfl
            # Bbox and DFL
            # cls loss
            if self.use_gfl:
                target_labels = torch.where(fg_masks > 0, tcls, torch.full_like(tcls, self.nc))
                one_hot_label = F.one_hot(tcls.long(), self.nc + 1)[..., :-1]
                loss_cls = self.varifocal_loss(pred_scores, tscores, one_hot_label)
            else:
                # is_in_range = torch.logical_and(pred_scores >= 0, pred_scores <= 1).all(dim=2)
                # LOGGER.info(f"is_in_range:{is_in_range}") 
                # loss_cls = (F.binary_cross_entropy_with_logits(pred_scores.float(), tscores.float(), reduction='none')).sum()
                loss_cls = self.bce(pred_scores, tscores.float()).sum()
            
            target_scores_sum = max(tscores.sum(), 1)
            LOGGER.info(f"check value {target_scores_sum}") # 1
            LOGGER.info(f"check value {loss_cls}")   # 300
            if target_scores_sum > 0:
                loss_cls /= target_scores_sum

            # bbox & dfl loss
            loss_iou, loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points_s, tbox,
                                            tscores, target_scores_sum, fg_masks)

            loss = torch.zeros(1, device=feats[0].device) + self.loss_weight['class'] * loss_cls + \
                self.loss_weight['iou'] * loss_iou + \
                self.loss_weight['dfl'] * loss_dfl
            loss_dict = dict(loss_iou = self.loss_weight['iou']*loss_iou, loss_dfl=self.loss_weight['dfl'] * loss_dfl,\
                loss_cls=self.loss_weight['class'] * loss_cls, loss=loss, num_fg=torch.sum(fg_masks)/max(num_gts, 1))
            return loss, loss_dict
    
    def __call__(self, p, targets):
        if self.use_ota == False:
            loss, loss_dict = self.build_loss(p, targets)
        # else:
        #     loss, loss_dict = self.ota_loss(p, targets)
        return loss, loss_dict