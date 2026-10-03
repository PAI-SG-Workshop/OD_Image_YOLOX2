"""
YOLOv8 builder (Image task only).

Reference: YOLOv8 has no paper. The Ultralytics release (2023) is the original; the Legacy
YOLOv8 project (yjh0410 style) follows it module by module and is copied here.

Algorithm (this folder, Legacy modules):
    backbone : C2f ("ELAN-CSP") stages, 3x3 stem          yolov8_backbone.py / yolov8_basic.py
    neck     : SPPF on P5                                   yolov8_neck.py
    PaFPN    : C2f blocks, no 1x1 reduce layers             yolov8_pafpn.py
    head     : decoupled, no objectness branch              yolov8_head.py
               reg width max(16, C3/4, 4*reg_max), cls width max(C3, min(nc, 100))
    pred     : DFL box (reg_max 16 bins per side, ltrb distances) yolov8_pred.py
    assigner : Task-Aligned Assigner (top-k 10, alpha 0.5, beta 6) matcher.py
    loss     : BCE(soft targets) 0.5 + CIoU 7.5 + DFL 1.5  loss.py

Shared with the other anchor-free detectors: detectors/anchor_free (targets, raw output, logging).
Raw objectness = 1 (no objectness branch), so the common score = class probability.

Delta vs v5: C3 -> C2f, no FPN reduce layers, decoupled obj-free head, DFL, TAL, loss.
Not used (common protocol): EMA, Mosaic / MixUp, close-mosaic, warm-up, BN eps 1e-3 / momentum 0.03.
"""

import torch.nn as nn

from detectors.yolov8.yolov8_backbone import build_backbone
from detectors.yolov8.yolov8_neck import build_neck
from detectors.yolov8.yolov8_pafpn import build_fpn

VARIANT = "yolov8_n"     # same scale as v5 / YOLOX (width 0.25, depth 0.34)


class YOLOv8Body(nn.Module):
    """Backbone + SPPF + PaFPN (reused by YOLOX2, whose Legacy modules are identical)."""

    def __init__(self, cfg):
        super().__init__()
        self.backbone, feats_dim = build_backbone(cfg)
        self.neck = build_neck(cfg, feats_dim[-1], feats_dim[-1])
        feats_dim[-1] = self.neck.out_dim
        self.fpn = build_fpn(cfg, feats_dim)
        self.out_dim = self.fpn.out_dim

    def forward(self, x):
        feats = self.backbone(x)
        feats[-1] = self.neck(feats[-1])
        return self.fpn(feats)
