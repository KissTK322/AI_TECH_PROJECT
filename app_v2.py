import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QHBoxLayout, QCheckBox, QMessageBox, QProgressBar, QDialog
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.ops import box_iou
from torchvision.ops import boxes as box_ops
from torchvision.transforms import functional as TVF
from pycocotools.coco import COCO

CONF_TH = 0.25          
IOU_TH = 0.25           

def to_qimage(img_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)


def draw_detection_boxes(img_bgr, boxes_xyxy, labels, scores, id2name, color=(0, 255, 0)):
    out = img_bgr.copy()
    for (x1, y1, x2, y2), lab, sc in zip(boxes_xyxy, labels, scores):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        name = id2name.get(int(lab), str(lab))
        text = f"{name} {float(sc):.2f}"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_top = max(0, y1 - th - 8)
        cv2.rectangle(out, (x1, y_top), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            out,
            text,
            (x1 + 3, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2
        )
    return out


def compare_prediction_with_gt(pred_boxes, pred_catids, gt_boxes, gt_catids, iou_thr=0.25):

    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 0, 0, 0, []
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes), []
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0, []

    pb = torch.tensor(pred_boxes, dtype=torch.float32)
    gb = torch.tensor(gt_boxes, dtype=torch.float32)
    ious = box_iou(pb, gb).numpy()

    used_g = set()
    matches = []
    tp_cls = 0
    fp = 0

    order = np.argsort(-np.max(ious, axis=1))
    for pi in order:
        gi = int(np.argmax(ious[pi]))
        best = float(ious[pi, gi])

        if best >= iou_thr and gi not in used_g:
            used_g.add(gi)
            ok_cls = (int(pred_catids[pi]) == int(gt_catids[gi]))
            matches.append((pi, gi, best, ok_cls))
            if ok_cls:
                tp_cls += 1
            else:
                fp += 1
        else:
            fp += 1

    fn = len(gt_boxes) - tp_cls
    return tp_cls, fp, fn, matches


def ensure_coco_fields(coco):
    changed = False
    for a in coco.dataset.get("annotations", []):
        if "iscrowd" not in a:
            a["iscrowd"] = 0
            changed = True
        if "area" not in a and "bbox" in a:
            a["area"] = float(a["bbox"][2] * a["bbox"][3])
            changed = True
    if changed:
        coco.createIndex()
    return coco


def normalize_int_mapping(d):
    if d is None:
        return None
    out = {}
    for k, v in d.items():
        out[int(k)] = int(v)
    return out


def encode_box_targets(proposals, gt, stds=(0.1, 0.1, 0.2, 0.2)):
    px1, py1, px2, py2 = proposals.unbind(dim=1)
    gx1, gy1, gx2, gy2 = gt.unbind(dim=1)

    pw = (px2 - px1).clamp(min=1e-6)
    ph = (py2 - py1).clamp(min=1e-6)
    px = px1 + 0.5 * pw
    py = py1 + 0.5 * ph

    gw = (gx2 - gx1).clamp(min=1e-6)
    gh = (gy2 - gy1).clamp(min=1e-6)
    gx = gx1 + 0.5 * gw
    gy = gy1 + 0.5 * gh

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)

    wx, wy, ww, wh = stds
    return torch.stack([dx / wx, dy / wy, dw / ww, dh / wh], dim=1)


def decode_box_deltas(proposals, deltas, stds=(0.1, 0.1, 0.2, 0.2), clip_val=math.log(1000.0 / 16)):
    px1, py1, px2, py2 = proposals.unbind(dim=1)
    pw = (px2 - px1).clamp(min=1e-6)
    ph = (py2 - py1).clamp(min=1e-6)
    px = px1 + 0.5 * pw
    py = py1 + 0.5 * ph

    wx, wy, ww, wh = stds
    dx = deltas[:, 0] * wx
    dy = deltas[:, 1] * wy
    dw = (deltas[:, 2] * ww).clamp(max=clip_val)
    dh = (deltas[:, 3] * wh).clamp(max=clip_val)

    gx = dx * pw + px
    gy = dy * ph + py
    gw = torch.exp(dw) * pw
    gh = torch.exp(dh) * ph

    x1 = gx - 0.5 * gw
    y1 = gy - 0.5 * gh
    x2 = gx + 0.5 * gw
    y2 = gy + 0.5 * gh
    return torch.stack([x1, y1, x2, y2], dim=1)


class TwoFCHead(nn.Module):
    def __init__(self, in_channels, fc_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class ClassAgnosticPredictor(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_dim, num_classes)  
        self.bbox_pred = nn.Linear(in_dim, 4)             

    def forward(self, x):
        return self.cls_score(x), self.bbox_pred(x)


def sample_training_labels(labels, batch_size=256, pos_fraction=0.25):
    pos_idx = torch.nonzero(labels > 0).squeeze(1)
    neg_idx = torch.nonzero(labels == 0).squeeze(1)

    num_pos = min(int(batch_size * pos_fraction), pos_idx.numel())
    num_neg = min(batch_size - num_pos, neg_idx.numel())

    if pos_idx.numel() > 0:
        pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=labels.device)[:num_pos]]
    if neg_idx.numel() > 0:
        neg_idx = neg_idx[torch.randperm(neg_idx.numel(), device=labels.device)[:num_neg]]

    keep = torch.cat([pos_idx, neg_idx], dim=0)
    return keep


def agnostic_fastrcnn_loss(class_logits, box_regression, labels, regression_targets, beta=1.0):
    loss_cls = F.cross_entropy(class_logits, labels)
    pos = torch.nonzero(labels > 0).squeeze(1)

    if pos.numel() == 0:
        loss_box = box_regression.sum() * 0.0
    else:
        loss_box = F.smooth_l1_loss(
            box_regression[pos],
            regression_targets[pos],
            beta=beta,
            reduction="sum"
        ) / labels.numel()

    return loss_cls, loss_box


class CascadeRoIHeads(nn.Module):
    def __init__(
        self,
        box_roi_pool,
        num_classes,
        feat_channels,
        num_stages=3,
        stage_iou_thr=(0.5, 0.6, 0.7),
        stage_bbox_stds=((0.1, 0.1, 0.2, 0.2), (0.05, 0.05, 0.1, 0.1), (0.033, 0.033, 0.067, 0.067)),
        stage_loss_weights=(1.0, 0.5, 0.25),
        batch_size_per_image=128,
        positive_fraction=0.25,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
        reg_beta=1.0,
        add_gt_as_proposals=True,
    ):
        super().__init__()
        self.box_roi_pool = box_roi_pool
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.stage_iou_thr = stage_iou_thr
        self.stage_bbox_stds = stage_bbox_stds
        self.stage_loss_weights = stage_loss_weights

        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.reg_beta = reg_beta
        self.add_gt_as_proposals = add_gt_as_proposals

        out_size = self.box_roi_pool.output_size
        if isinstance(out_size, int):
            out_size = (out_size, out_size)
        pool_h, pool_w = out_size

        in_dim = feat_channels * pool_h * pool_w

        self.box_heads = nn.ModuleList([TwoFCHead(in_dim, 1024) for _ in range(num_stages)])
        self.box_predictors = nn.ModuleList([ClassAgnosticPredictor(1024, num_classes) for _ in range(num_stages)])

    def has_mask(self):
        return False

    @torch.no_grad()
    def _match_and_label(self, proposals, gt_boxes, gt_labels, thr):
        device = proposals.device
        N = proposals.shape[0]

        if gt_boxes.numel() == 0:
            matched_idx = torch.full((N,), -1, dtype=torch.int64, device=device)
            labels = torch.zeros((N,), dtype=torch.int64, device=device)
            return matched_idx, labels

        ious = box_iou(gt_boxes, proposals)  # [G, N]
        max_iou, argmax = ious.max(dim=0)
        matched_idx = argmax
        labels = gt_labels[matched_idx].clone()
        labels[max_iou < thr] = 0
        return matched_idx, labels

    def _refine(self, proposals, deltas, stds, image_shape):
        boxes = decode_box_deltas(proposals, deltas, stds=stds)
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
        return boxes

    def forward(self, features, proposals, image_shapes, targets=None):
        device = proposals[0].device

        if targets is not None:
            losses = {}
            cur_props = proposals

            for s in range(self.num_stages):
                thr = self.stage_iou_thr[s]
                stds = self.stage_bbox_stds[s]
                w = self.stage_loss_weights[s]

                sampled_props, sampled_labels, sampled_reg_targets = [], [], []
                per_image_counts = []

                for props_i, tgt_i, img_shape in zip(cur_props, targets, image_shapes):
                    gt_boxes = tgt_i["boxes"]
                    gt_labels = tgt_i["labels"]

                    if self.add_gt_as_proposals and gt_boxes.numel() > 0:
                        props_i = torch.cat([props_i, gt_boxes], dim=0)

                    matched_idx, labels = self._match_and_label(props_i, gt_boxes, gt_labels, thr=thr)
                    keep = sample_training_labels(labels, self.batch_size_per_image, self.positive_fraction)

                    props_keep = props_i[keep]
                    labels_keep = labels[keep]

                    reg_t = torch.zeros((props_keep.shape[0], 4), dtype=torch.float32, device=device)
                    pos = torch.nonzero(labels_keep > 0).squeeze(1)
                    if pos.numel() > 0 and gt_boxes.numel() > 0:
                        gt_pos = gt_boxes[matched_idx[keep][pos]]
                        reg_t[pos] = encode_box_targets(props_keep[pos], gt_pos, stds=stds)

                    sampled_props.append(props_keep)
                    sampled_labels.append(labels_keep)
                    sampled_reg_targets.append(reg_t)
                    per_image_counts.append(props_keep.shape[0])

                box_feats = self.box_roi_pool(features, sampled_props, image_shapes)
                box_feats = self.box_heads[s](box_feats)
                class_logits, box_regression = self.box_predictors[s](box_feats)

                labels_cat = torch.cat(sampled_labels, dim=0)
                reg_targets_cat = torch.cat(sampled_reg_targets, dim=0)

                loss_cls, loss_box = agnostic_fastrcnn_loss(
                    class_logits, box_regression, labels_cat, reg_targets_cat, beta=self.reg_beta
                )
                losses[f"loss_cls_s{s+1}"] = loss_cls * w
                losses[f"loss_box_s{s+1}"] = loss_box * w

                with torch.no_grad():
                    splits = torch.split(box_regression, per_image_counts, dim=0)
                    next_props = []
                    for props_i, deltas_i, img_shape in zip(sampled_props, splits, image_shapes):
                        refined = self._refine(props_i, deltas_i, stds=stds, image_shape=img_shape)
                        next_props.append(refined.detach())
                    cur_props = next_props

            dets = []
            for _ in image_shapes:
                dets.append({
                    "boxes": torch.zeros((0, 4), device=device),
                    "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                    "scores": torch.zeros((0,), device=device),
                })
            return dets, losses

        else:
            cur_props = proposals
            last_logits = None

            for s in range(self.num_stages):
                stds = self.stage_bbox_stds[s]

                box_feats = self.box_roi_pool(features, cur_props, image_shapes)
                box_feats = self.box_heads[s](box_feats)
                class_logits, box_regression = self.box_predictors[s](box_feats)

                counts = [p.shape[0] for p in cur_props]
                logits_split = torch.split(class_logits, counts, dim=0)
                deltas_split = torch.split(box_regression, counts, dim=0)

                next_props = []
                for props_i, deltas_i, img_shape in zip(cur_props, deltas_split, image_shapes):
                    refined = self._refine(props_i, deltas_i, stds=stds, image_shape=img_shape)
                    next_props.append(refined)

                cur_props = next_props
                last_logits = logits_split

            detections = []
            for boxes_i, logits_i, img_shape in zip(cur_props, last_logits, image_shapes):
                boxes_i = box_ops.clip_boxes_to_image(boxes_i, img_shape)
                probs = F.softmax(logits_i, dim=1)

                all_boxes, all_scores, all_labels = [], [], []
                for c in range(1, self.num_classes):
                    sc = probs[:, c]
                    keep = torch.nonzero(sc > self.score_thresh).squeeze(1)
                    if keep.numel() == 0:
                        continue
                    all_boxes.append(boxes_i[keep])
                    all_scores.append(sc[keep])
                    all_labels.append(torch.full((keep.numel(),), c, dtype=torch.int64, device=boxes_i.device))

                if len(all_boxes) == 0:
                    detections.append({
                        "boxes": torch.zeros((0, 4), device=boxes_i.device),
                        "scores": torch.zeros((0,), device=boxes_i.device),
                        "labels": torch.zeros((0,), dtype=torch.int64, device=boxes_i.device),
                    })
                    continue

                boxes_cat = torch.cat(all_boxes, dim=0)
                scores_cat = torch.cat(all_scores, dim=0)
                labels_cat = torch.cat(all_labels, dim=0)

                keep_small = box_ops.remove_small_boxes(boxes_cat, min_size=1.0)
                boxes_cat = boxes_cat[keep_small]
                scores_cat = scores_cat[keep_small]
                labels_cat = labels_cat[keep_small]

                keep_nms = box_ops.batched_nms(boxes_cat, scores_cat, labels_cat, self.nms_thresh)
                keep_nms = keep_nms[: self.detections_per_img]

                detections.append({
                    "boxes": boxes_cat[keep_nms],
                    "scores": scores_cat[keep_nms],
                    "labels": labels_cat[keep_nms],
                })

            return detections, {}


def build_cascade_detector(num_classes, min_size=600, max_size=1000, cascade_cfg=None):
    if cascade_cfg is None:
        cascade_cfg = {}

    num_stages = int(cascade_cfg.get("stages", 3))
    stage_iou_thr = tuple(cascade_cfg.get("iou_thr", (0.5, 0.6, 0.7)))
    stage_bbox_stds = tuple(cascade_cfg.get(
        "bbox_stds",
        ((0.1, 0.1, 0.2, 0.2), (0.05, 0.05, 0.1, 0.1), (0.033, 0.033, 0.067, 0.067))
    ))
    stage_loss_weights = tuple(cascade_cfg.get("loss_w", (1.0, 0.5, 0.25)))

    base = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        min_size=min_size,
        max_size=max_size
    )

    rpn = base.rpn
    if hasattr(rpn, "_pre_nms_top_n") and isinstance(rpn._pre_nms_top_n, dict):
        rpn._pre_nms_top_n["training"] = 1000
        rpn._pre_nms_top_n["testing"] = 1000
    if hasattr(rpn, "_post_nms_top_n") and isinstance(rpn._post_nms_top_n, dict):
        rpn._post_nms_top_n["training"] = 500
        rpn._post_nms_top_n["testing"] = 500

    feat_channels = getattr(base.backbone, "out_channels", 256)

    base.roi_heads = CascadeRoIHeads(
        box_roi_pool=base.roi_heads.box_roi_pool,
        num_classes=num_classes,
        feat_channels=feat_channels,
        num_stages=num_stages,
        stage_iou_thr=stage_iou_thr,
        stage_bbox_stds=stage_bbox_stds,
        stage_loss_weights=stage_loss_weights,
        batch_size_per_image=128,
        positive_fraction=0.25,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
    )

    return base


def load_checkpoint_and_build_model(ckpt_path, coco):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and ("model_state_dict" in ckpt or "model_state" in ckpt):
        state = ckpt.get("model_state_dict", ckpt.get("model_state"))
        num_classes = int(ckpt.get("num_classes", len(coco.getCatIds()) + 1))
        min_size = int(ckpt.get("MIN_SIZE", 600))
        max_size = int(ckpt.get("MAX_SIZE", 1000))
        cascade_cfg = ckpt.get("cascade_cfg", {})
        contig_to_catid = normalize_int_mapping(ckpt.get("contig_to_catid", None))
        class_names = ckpt.get("CLASS_NAMES", None)
    else:

        state = ckpt
        num_classes = len(sorted(coco.getCatIds())) + 1
        min_size = 600
        max_size = 1000
        cascade_cfg = {}
        contig_to_catid = None
        class_names = None

    cats_sorted = sorted(coco.loadCats(coco.getCatIds()), key=lambda c: c["id"])

    if contig_to_catid is None:

        contig_to_catid = {i + 1: c["id"] for i, c in enumerate(cats_sorted)}

    model = build_cascade_detector(
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        cascade_cfg=cascade_cfg
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    return model, contig_to_catid, class_names, min_size, max_size

class PreviewDialog(QDialog):
    def __init__(self, title: str, qpix: QPixmap, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(950, 850)

        layout = QVBoxLayout()
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setPixmap(qpix)
        layout.addWidget(lbl)
        self.setLayout(layout)

class CascadeReviewApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cascade FRCNN Review App")
        self.resize(1500, 900)

        self.model_path = ""
        self.img_dir = ""
        self.gt_json = ""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.coco = None

        self.catid_to_name = {}
        self.contig_to_catid = {}
        self.fname2id = {}

        self.results = {}
        self.order_all = []
        self.order_wrong = []
        self.show_wrong_only = False
        self.idx = 0
        self.summary = {}

        self.min_size = None
        self.max_size = None

        root = QWidget()
        self.setCentralWidget(root)

        top = QHBoxLayout()
        self.btn_model = QPushButton("Choose model checkpoint")
        self.btn_imgs = QPushButton("Choose Image Folder")
        self.btn_gt = QPushButton("Choose GT COCO json")
        self.btn_run = QPushButton("Run review on all images")
        top.addWidget(self.btn_model)
        top.addWidget(self.btn_imgs)
        top.addWidget(self.btn_gt)
        top.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("Ready")

        mid = QHBoxLayout()
        self.lbl_pred = QLabel("Predicted")
        self.lbl_gt = QLabel("Ground Truth")
        self.lbl_pred.setAlignment(Qt.AlignCenter)
        self.lbl_gt.setAlignment(Qt.AlignCenter)
        self.lbl_pred.setMinimumSize(650, 460)
        self.lbl_gt.setMinimumSize(650, 460)
        mid.addWidget(self.lbl_pred, 1)
        mid.addWidget(self.lbl_gt, 1)

        # nav
        nav = QHBoxLayout()
        self.btn_prev = QPushButton("Prev")
        self.btn_next = QPushButton("Next")
        self.chk_wrong = QCheckBox("Show only wrong images")
        self.btn_csv = QPushButton("Export CSV summary")
        self.btn_save = QPushButton("Save wrong cases to fail_cases/")
        self.btn_cm = QPushButton("Open confusion matrix")
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        nav.addWidget(self.chk_wrong)
        nav.addWidget(self.btn_csv)
        nav.addWidget(self.btn_save)
        nav.addWidget(self.btn_cm)

        self.btn_cm.setEnabled(False)

        self.lbl_summary = QLabel("Summary: run the evaluation first")
        self.lbl_summary.setWordWrap(True)

        self.lbl_info = QLabel("Select a checkpoint, image folder, and COCO JSON file, then run the review.")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.progress)
        layout.addLayout(mid)
        layout.addLayout(nav)
        layout.addWidget(self.lbl_summary)
        layout.addWidget(self.lbl_info)
        root.setLayout(layout)

        self.btn_model.clicked.connect(self.pick_model)
        self.btn_imgs.clicked.connect(self.pick_image_folder)
        self.btn_gt.clicked.connect(self.pick_gt_json)
        self.btn_run.clicked.connect(self.run_all_predictions)
        self.btn_prev.clicked.connect(self.show_prev)
        self.btn_next.clicked.connect(self.show_next)
        self.chk_wrong.stateChanged.connect(self.toggle_wrong_filter)
        self.btn_csv.clicked.connect(self.export_csv)
        self.btn_save.clicked.connect(self.save_fail_cases)
        self.btn_cm.clicked.connect(self.show_confusion_matrix)

        self.refresh_path_info()

    def show_message(self, text, title="Info"):
        QMessageBox.information(self, title, text)

    def refresh_path_info(self):
        self.lbl_info.setText(
            f"Checkpoint: {self.model_path or '(not selected)'}\n"
            f"Images: {self.img_dir or '(not selected)'}\n"
            f"GT: {self.gt_json or '(not selected)'}\n"
            f"Device: {self.device}\n"
        )

    def pick_model(self):
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Select Cascade/FRCNN checkpoint",
            "",
            "PyTorch checkpoint (*.pt *.pth);;All files (*.*)"
        )
        if p:
            self.model_path = p
            self.refresh_path_info()

    def pick_image_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select image folder")
        if d:
            self.img_dir = d
            self.refresh_path_info()

    def pick_gt_json(self):
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Select COCO GT json",
            "",
            "JSON (*.json);;All files (*.*)"
        )
        if p:
            self.gt_json = p
            self.refresh_path_info()

    def load_resources(self):
        if not self.model_path or not Path(self.model_path).is_file():
            self.show_message("Please choose a model checkpoint (.pt or .pth).", "Error")
            return False

        if not self.img_dir or not Path(self.img_dir).is_dir():
            self.show_message("Please choose an image folder.", "Error")
            return False

        if not self.gt_json or not Path(self.gt_json).is_file():
            self.show_message("Please choose a COCO ground-truth JSON file.", "Error")
            return False

        self.coco = COCO(self.gt_json)
        self.coco = ensure_coco_fields(self.coco)

        cats = self.coco.loadCats(self.coco.getCatIds())
        cats_sorted = sorted(cats, key=lambda c: c["id"])
        self.catid_to_name = {c["id"]: c["name"] for c in cats_sorted}

        self.model, self.contig_to_catid, _, self.min_size, self.max_size = load_checkpoint_and_build_model(
            self.model_path,
            self.coco
        )
        self.model.to(self.device).eval()

        self.fname2id = {im["file_name"]: im["id"] for im in self.coco.dataset["images"]}
        return True

    def get_gt_for_image(self, fname):
        img_id = self.fname2id.get(fname, None)
        if img_id is None:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        boxes, cats = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
            cats.append(a["category_id"])

        return np.array(boxes, np.float32), np.array(cats, np.int64)

    def run_inference_on_image(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_t = TVF.to_tensor(rgb).to(self.device)

        with torch.no_grad():
            out = self.model([img_t])[0]

        if "boxes" not in out or out["boxes"].numel() == 0:
            pb = np.zeros((0, 4), np.float32)
            ps = np.zeros((0,), np.float32)
            pc = np.zeros((0,), np.int64)
            return pb, ps, pc

        pb = out["boxes"].detach().cpu().numpy().astype(np.float32)
        ps = out["scores"].detach().cpu().numpy().astype(np.float32)
        pl = out["labels"].detach().cpu().numpy().astype(np.int64)

        keep = ps >= CONF_TH
        pb = pb[keep]
        ps = ps[keep]
        pl = pl[keep]

        pc = np.array(
            [self.contig_to_catid.get(int(lbl), int(lbl)) for lbl in pl],
            dtype=np.int64
        )

        return pb, ps, pc

    def get_active_order(self):
        return self.order_wrong if self.show_wrong_only else self.order_all

    def build_detail_text(self, rec):
        pred_names = [self.catid_to_name.get(int(c), str(int(c))) for c in rec["pred_catids"]]
        gt_names = [self.catid_to_name.get(int(c), str(int(c))) for c in rec["gt_catids"]]

        lines = []
        lines.append("Pred classes: " + (", ".join(pred_names) if pred_names else "(none)"))
        lines.append("GT classes: " + (", ".join(gt_names) if gt_names else "(none)"))

        if len(rec["matches"]) == 0:
            if len(pred_names) == 0 and len(gt_names) == 0:
                lines.append("Per-object compare: no objects in both prediction and ground truth")
            else:
                lines.append("Per-object compare: no matched boxes")
            return "\n".join(lines)

        lines.append("Per-object compare:")

        used_p = set()
        used_g = set()

        for k, (pi, gi, iou, ok_cls) in enumerate(rec["matches"], start=1):
            used_p.add(pi)
            used_g.add(gi)

            pred_name = self.catid_to_name.get(int(rec["pred_catids"][pi]), str(int(rec["pred_catids"][pi])))
            gt_name = self.catid_to_name.get(int(rec["gt_catids"][gi]), str(int(rec["gt_catids"][gi])))

            status = "OK" if ok_cls else "WRONG CLASS"
            lines.append(f"  {k}. Pred = {pred_name} | GT = {gt_name} | IoU = {iou:.2f} | {status}")

        extra_preds = []
        for pi in range(len(rec["pred_catids"])):
            if pi not in used_p:
                pred_name = self.catid_to_name.get(int(rec["pred_catids"][pi]), str(int(rec["pred_catids"][pi])))
                extra_preds.append(pred_name)
        if extra_preds:
            lines.append("Unmatched predictions: " + ", ".join(extra_preds))

        missed_gts = []
        for gi in range(len(rec["gt_catids"])):
            if gi not in used_g:
                gt_name = self.catid_to_name.get(int(rec["gt_catids"][gi]), str(int(rec["gt_catids"][gi])))
                missed_gts.append(gt_name)
        if missed_gts:
            lines.append("Missed ground truths: " + ", ".join(missed_gts))

        return "\n".join(lines)

    def run_all_predictions(self):
        if not self.load_resources():
            return

        images = self.coco.dataset["images"]
        n = len(images)

        self.results = {}
        self.order_all = []
        self.order_wrong = []
        self.idx = 0
        self.btn_cm.setEnabled(False)

        self.progress.setRange(0, n)
        self.progress.setValue(0)

        TP = FP = FN = 0
        TP_img = FP_img = FN_img = TN_img = 0

        for b in [self.btn_run, self.btn_prev, self.btn_next, self.btn_csv, self.btn_save, self.btn_cm]:
            b.setEnabled(False)

        for i, im in enumerate(images, start=1):
            fname = im["file_name"]
            fpath = str(Path(self.img_dir) / fname)

            self.progress.setValue(i)
            self.progress.setFormat(f"Predicting {i}/{n}: {fname}")
            QApplication.processEvents()

            if not Path(fpath).is_file():
                rec = {
                    "file": fpath,
                    "pred_boxes": np.zeros((0, 4), np.float32),
                    "pred_scores": np.zeros((0,), np.float32),
                    "pred_catids": np.zeros((0,), np.int64),
                    "gt_boxes": np.zeros((0, 4), np.float32),
                    "gt_catids": np.zeros((0,), np.int64),
                    "tp": 0, "fp": 0, "fn": 0, "wrong": True, "matches": []
                }
                self.results[fname] = rec
                continue

            img = cv2.imread(fpath)
            if img is None:
                rec = {
                    "file": fpath,
                    "pred_boxes": np.zeros((0, 4), np.float32),
                    "pred_scores": np.zeros((0,), np.float32),
                    "pred_catids": np.zeros((0,), np.int64),
                    "gt_boxes": np.zeros((0, 4), np.float32),
                    "gt_catids": np.zeros((0,), np.int64),
                    "tp": 0, "fp": 0, "fn": 0, "wrong": True, "matches": []
                }
                self.results[fname] = rec
                continue

            pb, ps, pc = self.run_inference_on_image(img)
            gb, gc = self.get_gt_for_image(fname)

            tp, fp, fn, matches = compare_prediction_with_gt(pb, pc, gb, gc, iou_thr=IOU_TH)
            wrong = (fp > 0) or (fn > 0)

            TP += tp
            FP += fp
            FN += fn

            has_gt = (len(gc) > 0)
            has_pred = (len(pc) > 0)

            if (not has_gt) and (not has_pred):
                TN_img += 1
            elif (not has_gt) and has_pred:
                FP_img += 1
            elif has_gt and (tp == 0):
                FN_img += 1
            else:
                TP_img += 1

            rec = {
                "file": fpath,
                "pred_boxes": pb,
                "pred_scores": ps,
                "pred_catids": pc,
                "gt_boxes": gb,
                "gt_catids": gc,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "wrong": wrong,
                "matches": matches
            }
            self.results[fname] = rec

        self.order_all = list(self.results.keys())
        self.order_wrong = [k for k, v in self.results.items() if v["wrong"]]

        correct_img = len(self.order_all) - len(self.order_wrong)
        acc_img = correct_img / max(len(self.order_all), 1)

        self.summary = {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TP_img": TP_img,
            "FP_img": FP_img,
            "FN_img": FN_img,
            "TN_img": TN_img,
            "wrong_images": len(self.order_wrong),
            "total_images": len(self.order_all),
            "image_accuracy": acc_img
        }

        self.lbl_summary.setText(
            "SUMMARY (locked)\n"
            f"Object-level: TP={TP}  FP={FP}  FN={FN}\n"
            f"Image-level: TP={TP_img}  FP={FP_img}  FN={FN_img}  TN={TN_img}\n"
            f"Wrong images={len(self.order_wrong)} / {len(self.order_all)}   (image-acc={acc_img*100:.2f}%)\n"
            f"post-filter conf={CONF_TH}, match iou={IOU_TH}\n"
            f"device={self.device}"
        )

        self.progress.setFormat("Finished")

        for b in [self.btn_run, self.btn_prev, self.btn_next, self.btn_csv, self.btn_save, self.btn_cm]:
            b.setEnabled(True)
        self.btn_cm.setEnabled(True)

        self.idx = 0
        self.refresh_preview()

        self.show_message(
            f"Done. total={len(self.order_all)} | wrong={len(self.order_wrong)} | image-acc={acc_img*100:.2f}%",
            "Finished"
        )

    def refresh_preview(self):
        order = self.get_active_order()
        if not order:
            self.lbl_pred.setText("No images to show")
            self.lbl_gt.setText("No images to show")
            return

        fname = order[self.idx % len(order)]
        rec = self.results[fname]

        img_bgr = cv2.imread(rec["file"])
        if img_bgr is None:
            self.lbl_info.setText(f"Cannot read: {rec['file']}")
            return

        vis_pred = draw_detection_boxes(
            img_bgr,
            rec["pred_boxes"],
            rec["pred_catids"],
            rec["pred_scores"],
            self.catid_to_name,
            color=(0, 255, 0)
        )

        vis_gt = draw_detection_boxes(
            img_bgr,
            rec["gt_boxes"],
            rec["gt_catids"],
            np.ones((len(rec["gt_catids"]),), np.float32),
            self.catid_to_name,
            color=(0, 0, 255)
        )

        qp = QPixmap.fromImage(to_qimage(vis_pred)).scaled(
            self.lbl_pred.width(),
            self.lbl_pred.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        qg = QPixmap.fromImage(to_qimage(vis_gt)).scaled(
            self.lbl_gt.width(),
            self.lbl_gt.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.lbl_pred.setPixmap(qp)
        self.lbl_gt.setPixmap(qg)

        class_detail = self.build_detail_text(rec)

        self.lbl_info.setText(
            f"{fname}\n"
            f"TP={rec['tp']} FP={rec['fp']} FN={rec['fn']} wrong={rec['wrong']} "
            f"{class_detail}\n\n"
            f"Checkpoint: {self.model_path}\n"
            f"Images: {self.img_dir}\n"
            f"GT: {self.gt_json}\n"
            f"Device: {self.device}\n"
            f"MIN_SIZE={self.min_size}, MAX_SIZE={self.max_size}"
        )

    def show_prev(self):
        order = self.get_active_order()
        if not order:
            return
        self.idx = (self.idx - 1) % len(order)
        self.refresh_preview()

    def show_next(self):
        order = self.get_active_order()
        if not order:
            return
        self.idx = (self.idx + 1) % len(order)
        self.refresh_preview()

    def toggle_wrong_filter(self, _):
        self.show_wrong_only = self.chk_wrong.isChecked()
        self.idx = 0
        self.refresh_preview()

    def export_csv(self):
        if not self.results:
            self.show_message("Run prediction first.", "Error")
            return

        default = str(Path(self.img_dir) / "summary_cascade_frcnn.csv")
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV",
            default,
            "CSV (*.csv)"
        )
        if not out_path:
            return

        rows = []
        for fname, rec in self.results.items():
            pred_names = [self.catid_to_name.get(int(c), str(int(c))) for c in rec["pred_catids"]]
            gt_names = [self.catid_to_name.get(int(c), str(int(c))) for c in rec["gt_catids"]]

            rows.append({
                "file_name": fname,
                "tp": rec["tp"],
                "fp": rec["fp"],
                "fn": rec["fn"],
                "wrong": rec["wrong"],
                "num_pred": int(len(rec["pred_catids"])),
                "num_gt": int(len(rec["gt_catids"])),
                "pred_class_names": " | ".join(pred_names),
                "gt_class_names": " | ".join(gt_names),
                "detail": self.build_detail_text(rec).replace("\n", " || ")
            })

        pd.DataFrame(rows).to_csv(out_path, index=False)
        self.show_message(f"Saved: {out_path}", "CSV")

    def save_fail_cases(self):
        if not self.results:
            self.show_message("Run prediction first.", "Error")
            return

        out_dir = Path(self.img_dir) / "fail_cases"
        out_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for fname, rec in self.results.items():
            if not rec["wrong"]:
                continue

            img_bgr = cv2.imread(rec["file"])
            if img_bgr is None:
                continue

            vis = img_bgr.copy()
            vis = draw_detection_boxes(
                vis,
                rec["gt_boxes"],
                rec["gt_catids"],
                np.ones((len(rec["gt_catids"]),), np.float32),
                self.catid_to_name,
                color=(0, 0, 255)
            )
            vis = draw_detection_boxes(
                vis,
                rec["pred_boxes"],
                rec["pred_catids"],
                rec["pred_scores"],
                self.catid_to_name,
                color=(0, 255, 0)
            )

            out_path = out_dir / fname
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), vis)
            saved += 1

        self.show_message(f"Saved {saved} images to: {out_dir}", "Done")

    def show_confusion_matrix(self):
        if not self.results or not self.coco:
            self.show_message("Run prediction first.", "Error")
            return

        cats = self.coco.loadCats(self.coco.getCatIds())
        cats_sorted = sorted(cats, key=lambda c: c["id"])
        cat_ids = [c["id"] for c in cats_sorted]
        cat_names = [c["name"] for c in cats_sorted]

        bg_id = -1
        labels = cat_ids + [bg_id]
        names = cat_names + ["__background__"]
        idx_of = {cid: i for i, cid in enumerate(labels)}

        K = len(labels)
        cm = np.zeros((K, K), dtype=np.int64)

        for fname, rec in self.results.items():
            pb = rec["pred_boxes"]
            pc = rec["pred_catids"]
            gb = rec["gt_boxes"]
            gc = rec["gt_catids"]

            if len(pb) == 0 and len(gb) == 0:
                continue

            if len(gb) == 0 and len(pb) > 0:
                for pred_c in pc:
                    if int(pred_c) in idx_of:
                        cm[idx_of[bg_id], idx_of[int(pred_c)]] += 1
                continue

            if len(pb) == 0 and len(gb) > 0:
                for gt_c in gc:
                    if int(gt_c) in idx_of:
                        cm[idx_of[int(gt_c)], idx_of[bg_id]] += 1
                continue

            pb_t = torch.tensor(pb, dtype=torch.float32)
            gb_t = torch.tensor(gb, dtype=torch.float32)
            ious = box_iou(pb_t, gb_t).numpy()

            used_g = set()
            used_p = set()

            order = np.argsort(-np.max(ious, axis=1))
            for pi in order:
                gi = int(np.argmax(ious[pi]))
                best = float(ious[pi, gi])

                if best >= IOU_TH and gi not in used_g:
                    used_g.add(gi)
                    used_p.add(pi)

                    gt_c = int(gc[gi])
                    pred_c = int(pc[pi])

                    if gt_c in idx_of and pred_c in idx_of:
                        cm[idx_of[gt_c], idx_of[pred_c]] += 1

            for pi in range(len(pb)):
                if pi not in used_p:
                    pred_c = int(pc[pi])
                    if pred_c in idx_of:
                        cm[idx_of[bg_id], idx_of[pred_c]] += 1

            for gi in range(len(gb)):
                if gi not in used_g:
                    gt_c = int(gc[gi])
                    if gt_c in idx_of:
                        cm[idx_of[gt_c], idx_of[bg_id]] += 1

        fig, ax = plt.subplots(figsize=(10, 9))
        ax.imshow(cm, interpolation="nearest")
        ax.set_title(f"Detection Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")

        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels(names, rotation=90, fontsize=8)
        ax.set_yticklabels(names, fontsize=8)

        for i in range(K):
            for j in range(K):
                v = cm[i, j]
                if v != 0:
                    ax.text(j, i, str(v), ha="center", va="center", fontsize=7)

        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        plt.close(fig)

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        qimg = to_qimage(img_bgr)
        pix = QPixmap.fromImage(qimg).scaled(900, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        dlg = PreviewDialog("Confusion Matrix", pix, parent=self)
        dlg.exec_()


def main():
    app = QApplication(sys.argv)
    w = CascadeReviewApp()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()    bytes_per_line = c * w
    return QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)


def draw_detection_boxes(image_bgr, boxes_xyxy, labels, scores, label_map, color=(0, 255, 0)):
    canvas = image_bgr.copy()

    for (x1, y1, x2, y2), label_id, score in zip(boxes_xyxy, labels, scores):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        label_name = label_map.get(int(label_id), str(label_id))
        tag = f"{label_name} {float(score):.2f}"
        (text_w, text_h), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        top = max(0, y1 - text_h - 8)
        cv2.rectangle(canvas, (x1, top), (x1 + text_w + 6, y1), color, -1)
        cv2.putText(
            canvas,
            tag,
            (x1 + 3, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    return canvas


def compare_prediction_with_gt(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thr=0.25):
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 0, 0, 0, []

    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes), []

    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0, []

    pred_tensor = torch.tensor(pred_boxes, dtype=torch.float32)
    gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32)
    iou_table = box_iou(pred_tensor, gt_tensor).numpy()

    used_gt = set()
    matches = []
    true_positive = 0
    false_positive = 0

    candidate_order = np.argsort(-np.max(iou_table, axis=1))
    for pred_idx in candidate_order:
        gt_idx = int(np.argmax(iou_table[pred_idx]))
        best_iou = float(iou_table[pred_idx, gt_idx])

        if best_iou >= iou_thr and gt_idx not in used_gt:
            used_gt.add(gt_idx)
            class_ok = int(pred_labels[pred_idx]) == int(gt_labels[gt_idx])
            matches.append((pred_idx, gt_idx, best_iou, class_ok))
            if class_ok:
                true_positive += 1
            else:
                false_positive += 1
        else:
            false_positive += 1

    false_negative = len(gt_boxes) - true_positive
    return true_positive, false_positive, false_negative, matches


class ImagePopup(QDialog):
    def __init__(self, title: str, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(950, 850)

        layout = QVBoxLayout()
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setPixmap(pixmap)

        layout.addWidget(image_label)
        self.setLayout(layout)


class DetectionReviewApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cascade FRCNN Review App")
        self.resize(1500, 900)

        self.model_path = ""
        self.image_dir = ""
        self.gt_json_path = ""

        self.model = None
        self.coco = None
        self.catid_to_name = {}
        self.model_cls_to_catid = {}
        self.filename_to_img_id = {}

        self.records = {}
        self.all_image_names = []
        self.wrong_image_names = []
        self.show_wrong_only = False
        self.current_index = 0
        self.summary = {}

        self._build_ui()
        self._connect_signals()
        self.refresh_path_info()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        top_bar = QHBoxLayout()
        self.btn_model = QPushButton("Choose Cascade FRCNN Weights")
        self.btn_images = QPushButton("Choose Image Folder")
        self.btn_gt = QPushButton("Choose GT COCO JSON")
        self.btn_run = QPushButton("Run All Predictions")

        for widget in [self.btn_model, self.btn_images, self.btn_gt, self.btn_run]:
            top_bar.addWidget(widget)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("Idle")

        preview_row = QHBoxLayout()
        self.pred_label = QLabel("Prediction")
        self.gt_label = QLabel("Ground Truth")

        for label in [self.pred_label, self.gt_label]:
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(650, 460)

        preview_row.addWidget(self.pred_label, 1)
        preview_row.addWidget(self.gt_label, 1)

        action_row = QHBoxLayout()
        self.btn_prev = QPushButton("Prev")
        self.btn_next = QPushButton("Next")
        self.chk_wrong = QCheckBox("Show only wrong images")
        self.btn_export = QPushButton("Export CSV")
        self.btn_save_fail = QPushButton("Save fail cases")
        self.btn_confusion = QPushButton("Show confusion matrix")

        for widget in [
            self.btn_prev,
            self.btn_next,
            self.chk_wrong,
            self.btn_export,
            self.btn_save_fail,
            self.btn_confusion,
        ]:
            action_row.addWidget(widget)

        self.btn_confusion.setEnabled(False)

        self.summary_label = QLabel("SUMMARY: run evaluation first")
        self.summary_label.setWordWrap(True)

        self.info_label = QLabel("Pick a model, image folder, and GT json.")
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        page = QVBoxLayout()
        page.addLayout(top_bar)
        page.addWidget(self.progress)
        page.addLayout(preview_row)
        page.addLayout(action_row)
        page.addWidget(self.summary_label)
        page.addWidget(self.info_label)

        root.setLayout(page)

    def _connect_signals(self):
        self.btn_model.clicked.connect(self.pick_model)
        self.btn_images.clicked.connect(self.pick_image_folder)
        self.btn_gt.clicked.connect(self.pick_gt_json)
        self.btn_run.clicked.connect(self.run_all_predictions)

        self.btn_prev.clicked.connect(self.show_prev)
        self.btn_next.clicked.connect(self.show_next)
        self.chk_wrong.stateChanged.connect(self.toggle_wrong_filter)
        self.btn_export.clicked.connect(self.export_csv)
        self.btn_save_fail.clicked.connect(self.save_fail_cases)
        self.btn_confusion.clicked.connect(self.show_confusion_matrix)

    def show_message(self, text, title="Info"):
        QMessageBox.information(self, title, text)

    def refresh_path_info(self):
        self.info_label.setText(
            f"Model: {self.model_path or '(not selected)'}\n"
            f"Images: {self.image_dir or '(not selected)'}\n"
            f"GT: {self.gt_json_path or '(not selected)'}\n"
        )

    def pick_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select casecade_frcnn_best.pt",
            "",
            "PyTorch weights (*.pt);;All files (*.*)",
        )
        if path:
            self.model_path = path
            self.refresh_path_info()

    def pick_image_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select image folder")
        if path:
            self.image_dir = path
            self.refresh_path_info()

    def pick_gt_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select COCO GT json",
            "",
            "JSON (*.json);;All files (*.*)",
        )
        if path:
            self.gt_json_path = path
            self.refresh_path_info()

    def load_resources(self):
        if not self.model_path or not Path(self.model_path).is_file():
            self.show_message("Please choose casecade_frcnn_best.pt", "Error")
            return False
        if not self.image_dir or not Path(self.image_dir).is_dir():
            self.show_message("Please choose image folder", "Error")
            return False
        if not self.gt_json_path or not Path(self.gt_json_path).is_file():
            self.show_message("Please choose GT COCO json", "Error")
            return False

        self.model = YOLO(self.model_path)
        self.coco = COCO(self.gt_json_path)

        categories = self.coco.loadCats(self.coco.getCatIds())
        self.catid_to_name = {item["id"]: item["name"] for item in categories}

        categories_sorted = sorted(categories, key=lambda item: item["id"])
        self.model_cls_to_catid = {idx: item["id"] for idx, item in enumerate(categories_sorted)}

        self.filename_to_img_id = {
            item["file_name"]: item["id"] for item in self.coco.dataset["images"]
        }
        return True

    def get_gt_for_image(self, file_name):
        image_id = self.filename_to_img_id.get(file_name)
        if image_id is None:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)

        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        annotations = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        return np.array(boxes, np.float32), np.array(labels, np.int64)

    def current_list(self):
        return self.wrong_image_names if self.show_wrong_only else self.all_image_names

    def build_detail_text(self, record):
        pred_names = [self.catid_to_name.get(int(c), str(int(c))) for c in record["pred_labels"]]
        gt_names = [self.catid_to_name.get(int(c), str(int(c))) for c in record["gt_labels"]]

        lines = [
            "Pred classes: " + (", ".join(pred_names) if pred_names else "(none)"),
            "GT classes: " + (", ".join(gt_names) if gt_names else "(none)"),
        ]

        if len(record["matches"]) == 0:
            if len(pred_names) == 0 and len(gt_names) == 0:
                lines.append("Per-object compare: no objects in either prediction or ground truth")
            else:
                lines.append("Per-object compare: no matched boxes")
            return "\n".join(lines)

        lines.append("Per-object compare:")

        used_pred = set()
        used_gt = set()

        for idx, (pred_i, gt_i, iou, ok_cls) in enumerate(record["matches"], start=1):
            used_pred.add(pred_i)
            used_gt.add(gt_i)

            pred_name = self.catid_to_name.get(int(record["pred_labels"][pred_i]), str(int(record["pred_labels"][pred_i])))
            gt_name = self.catid_to_name.get(int(record["gt_labels"][gt_i]), str(int(record["gt_labels"][gt_i])))
            status = "OK" if ok_cls else "WRONG CLASS"

            lines.append(
                f"  {idx}. Pred = {pred_name} | GT = {gt_name} | IoU = {iou:.2f} | {status}"
            )

        extra_preds = []
        for pred_i in range(len(record["pred_labels"])):
            if pred_i not in used_pred:
                extra_preds.append(
                    self.catid_to_name.get(int(record["pred_labels"][pred_i]), str(int(record["pred_labels"][pred_i])))
                )
        if extra_preds:
            lines.append("Unmatched predictions: " + ", ".join(extra_preds))

        missed_gt = []
        for gt_i in range(len(record["gt_labels"])):
            if gt_i not in used_gt:
                missed_gt.append(
                    self.catid_to_name.get(int(record["gt_labels"][gt_i]), str(int(record["gt_labels"][gt_i])))
                )
        if missed_gt:
            lines.append("Missed ground truths: " + ", ".join(missed_gt))

        return "\n".join(lines)

    def run_all_predictions(self):
        if not self.load_resources():
            return

        image_entries = self.coco.dataset["images"]
        total_images = len(image_entries)

        self.records = {}
        self.all_image_names = []
        self.wrong_image_names = []
        self.current_index = 0
        self.btn_confusion.setEnabled(False)

        self.progress.setRange(0, total_images)
        self.progress.setValue(0)

        obj_tp = obj_fp = obj_fn = 0
        img_tp = img_fp = img_fn = img_tn = 0

        buttons_to_disable = [
            self.btn_run,
            self.btn_prev,
            self.btn_next,
            self.btn_export,
            self.btn_save_fail,
            self.btn_confusion,
        ]
        for button in buttons_to_disable:
            button.setEnabled(False)

        for step, image_info in enumerate(image_entries, start=1):
            file_name = image_info["file_name"]
            image_path = str(Path(self.image_dir) / file_name)

            self.progress.setValue(step)
            self.progress.setFormat(f"Predicting {step}/{total_images}: {file_name}")
            QApplication.processEvents()

            record = self._run_single_image(image_path, file_name)
            self.records[file_name] = record

            obj_tp += record["tp"]
            obj_fp += record["fp"]
            obj_fn += record["fn"]

            has_gt = len(record["gt_labels"]) > 0
            has_pred = len(record["pred_labels"]) > 0

            if not has_gt and not has_pred:
                img_tn += 1
            elif not has_gt and has_pred:
                img_fp += 1
            elif has_gt and record["tp"] == 0:
                img_fn += 1
            else:
                img_tp += 1

        self.all_image_names = list(self.records.keys())
        self.wrong_image_names = [name for name, rec in self.records.items() if rec["wrong"]]

        num_correct = len(self.all_image_names) - len(self.wrong_image_names)
        img_acc = num_correct / max(len(self.all_image_names), 1)

        self.summary = {
            "TP": obj_tp,
            "FP": obj_fp,
            "FN": obj_fn,
            "TP_img": img_tp,
            "FP_img": img_fp,
            "FN_img": img_fn,
            "TN_img": img_tn,
            "wrong_images": len(self.wrong_image_names),
            "total_images": len(self.all_image_names),
            "image_accuracy": img_acc,
        }

        self.summary_label.setText(
            "SUMMARY\n"
            f"Object-level: TP={obj_tp}  FP={obj_fp}  FN={obj_fn}\n"
            f"Image-level: TP={img_tp}  FP={img_fp}  FN={img_fn}  TN={img_tn}\n"
            f"Wrong images={len(self.wrong_image_names)} / {len(self.all_image_names)}   "
            f"(image-acc={img_acc * 100:.2f}%)"
        )

        self.progress.setFormat("Done")

        for button in buttons_to_disable:
            button.setEnabled(True)
        self.btn_confusion.setEnabled(True)

        self.current_index = 0
        self.refresh_preview()
        self.show_message(
            f"Done. total={len(self.all_image_names)} | wrong={len(self.wrong_image_names)} | "
            f"image-acc={img_acc * 100:.2f}%",
            "Finished",
        )

    def _run_single_image(self, image_path, file_name):
        empty_record = {
            "file": image_path,
            "pred_boxes": np.zeros((0, 4), np.float32),
            "pred_scores": np.zeros((0,), np.float32),
            "pred_labels": np.zeros((0,), np.int64),
            "gt_boxes": np.zeros((0, 4), np.float32),
            "gt_labels": np.zeros((0,), np.int64),
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "wrong": True,
            "matches": [],
        }

        if not Path(image_path).is_file():
            return empty_record

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            return empty_record

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        prediction = self.model.predict(
            source=image_rgb,
            conf=CONF_TH,
            iou=IOU_TH,
            verbose=False,
        )[0]

        if prediction.boxes is None or len(prediction.boxes) == 0:
            pred_boxes = np.zeros((0, 4), np.float32)
            pred_scores = np.zeros((0,), np.float32)
            pred_labels = np.zeros((0,), np.int64)
        else:
            pred_boxes = prediction.boxes.xyxy.cpu().numpy().astype(np.float32)
            pred_scores = prediction.boxes.conf.cpu().numpy().astype(np.float32)
            raw_labels = prediction.boxes.cls.cpu().numpy().astype(np.int64)
            pred_labels = np.array(
                [self.model_cls_to_catid[int(label)] for label in raw_labels],
                dtype=np.int64,
            )

        gt_boxes, gt_labels = self.get_gt_for_image(file_name)
        tp, fp, fn, matches = compare_prediction_with_gt(
            pred_boxes,
            pred_labels,
            gt_boxes,
            gt_labels,
            iou_thr=IOU_TH,
        )

        return {
            "file": image_path,
            "pred_boxes": pred_boxes,
            "pred_scores": pred_scores,
            "pred_labels": pred_labels,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "wrong": (fp > 0) or (fn > 0),
            "matches": matches,
        }

    def refresh_preview(self):
        image_names = self.current_list()
        if not image_names:
            self.pred_label.setText("No images to show")
            self.gt_label.setText("No images to show")
            return

        file_name = image_names[self.current_index % len(image_names)]
        record = self.records[file_name]

        image_bgr = cv2.imread(record["file"])
        if image_bgr is None:
            self.info_label.setText(f"Cannot read: {record['file']}")
            return

        pred_vis = draw_detection_boxes(
            image_bgr,
            record["pred_boxes"],
            record["pred_labels"],
            record["pred_scores"],
            self.catid_to_name,
            color=(0, 255, 0),
        )
        gt_vis = draw_detection_boxes(
            image_bgr,
            record["gt_boxes"],
            record["gt_labels"],
            np.ones((len(record["gt_labels"]),), np.float32),
            self.catid_to_name,
            color=(0, 0, 255),
        )

        pred_pix = QPixmap.fromImage(to_qimage(pred_vis)).scaled(
            self.pred_label.width(),
            self.pred_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        gt_pix = QPixmap.fromImage(to_qimage(gt_vis)).scaled(
            self.gt_label.width(),
            self.gt_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.pred_label.setPixmap(pred_pix)
        self.gt_label.setPixmap(gt_pix)

        detail_text = self.build_detail_text(record)
        self.info_label.setText(
            f"{file_name}\n"
            f"TP={record['tp']} FP={record['fp']} FN={record['fn']} wrong={record['wrong']}  |  "
            f"conf={CONF_TH}, iou={IOU_TH}\n\n"
            f"{detail_text}\n\n"
            f"Model: {self.model_path}\n"
            f"Images: {self.image_dir}\n"
            f"GT: {self.gt_json_path}"
        )

    def show_prev(self):
        image_names = self.current_list()
        if not image_names:
            return
        self.current_index = (self.current_index - 1) % len(image_names)
        self.refresh_preview()

    def show_next(self):
        image_names = self.current_list()
        if not image_names:
            return
        self.current_index = (self.current_index + 1) % len(image_names)
        self.refresh_preview()

    def toggle_wrong_filter(self, _):
        self.show_wrong_only = self.chk_wrong.isChecked()
        self.current_index = 0
        self.refresh_preview()

    def export_csv(self):
        if not self.records:
            self.show_message("Run prediction first.", "Error")
            return

        default_path = str(Path(self.image_dir) / "summary_cascade_frcnn.csv")
        out_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", default_path, "CSV (*.csv)")
        if not out_path:
            return

        rows = []
        for file_name, record in self.records.items():
            pred_names = [self.catid_to_name.get(int(c), str(int(c))) for c in record["pred_labels"]]
            gt_names = [self.catid_to_name.get(int(c), str(int(c))) for c in record["gt_labels"]]

            rows.append(
                {
                    "file_name": file_name,
                    "tp": record["tp"],
                    "fp": record["fp"],
                    "fn": record["fn"],
                    "wrong": record["wrong"],
                    "num_pred": int(len(record["pred_labels"])),
                    "num_gt": int(len(record["gt_labels"])),
                    "pred_class_names": " | ".join(pred_names),
                    "gt_class_names": " | ".join(gt_names),
                    "detail": self.build_detail_text(record).replace("\n", " || "),
                }
            )

        pd.DataFrame(rows).to_csv(out_path, index=False)
        self.show_message(f"Saved: {out_path}", "CSV")

    def save_fail_cases(self):
        if not self.records:
            self.show_message("Run prediction first.", "Error")
            return

        out_dir = Path(self.image_dir) / "fail_cases"
        out_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0
        for file_name, record in self.records.items():
            if not record["wrong"]:
                continue

            image_bgr = cv2.imread(record["file"])
            if image_bgr is None:
                continue

            merged_view = image_bgr.copy()
            merged_view = draw_detection_boxes(
                merged_view,
                record["gt_boxes"],
                record["gt_labels"],
                np.ones((len(record["gt_labels"]),), np.float32),
                self.catid_to_name,
                color=(0, 0, 255),
            )
            merged_view = draw_detection_boxes(
                merged_view,
                record["pred_boxes"],
                record["pred_labels"],
                record["pred_scores"],
                self.catid_to_name,
                color=(0, 255, 0),
            )

            save_path = out_dir / file_name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), merged_view)
            saved_count += 1

        self.show_message(f"Saved {saved_count} images to: {out_dir}", "Done")

    def show_confusion_matrix(self):
        if not self.records or not self.coco:
            self.show_message("Run prediction first.", "Error")
            return

        categories = self.coco.loadCats(self.coco.getCatIds())
        categories_sorted = sorted(categories, key=lambda item: item["id"])
        category_ids = [item["id"] for item in categories_sorted]
        category_names = [item["name"] for item in categories_sorted]

        bg_id = -1
        labels = category_ids + [bg_id]
        label_names = category_names + ["__background__"]
        index_map = {cat_id: idx for idx, cat_id in enumerate(labels)}

        size = len(labels)
        cm = np.zeros((size, size), dtype=np.int64)

        for record in self.records.values():
            pred_boxes = record["pred_boxes"]
            pred_labels = record["pred_labels"]
            gt_boxes = record["gt_boxes"]
            gt_labels = record["gt_labels"]

            if len(pred_boxes) == 0 and len(gt_boxes) == 0:
                continue

            if len(gt_boxes) == 0 and len(pred_boxes) > 0:
                for pred_label in pred_labels:
                    if int(pred_label) in index_map:
                        cm[index_map[bg_id], index_map[int(pred_label)]] += 1
                continue

            if len(pred_boxes) == 0 and len(gt_boxes) > 0:
                for gt_label in gt_labels:
                    if int(gt_label) in index_map:
                        cm[index_map[int(gt_label)], index_map[bg_id]] += 1
                continue

            pred_tensor = torch.tensor(pred_boxes, dtype=torch.float32)
            gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32)
            iou_table = box_iou(pred_tensor, gt_tensor).numpy()

            used_gt = set()
            used_pred = set()

            candidate_order = np.argsort(-np.max(iou_table, axis=1))
            for pred_idx in candidate_order:
                gt_idx = int(np.argmax(iou_table[pred_idx]))
                best_iou = float(iou_table[pred_idx, gt_idx])

                if best_iou >= IOU_TH and gt_idx not in used_gt:
                    used_gt.add(gt_idx)
                    used_pred.add(pred_idx)

                    gt_label = int(gt_labels[gt_idx])
                    pred_label = int(pred_labels[pred_idx])
                    if gt_label in index_map and pred_label in index_map:
                        cm[index_map[gt_label], index_map[pred_label]] += 1

            for pred_idx in range(len(pred_boxes)):
                if pred_idx not in used_pred:
                    pred_label = int(pred_labels[pred_idx])
                    if pred_label in index_map:
                        cm[index_map[bg_id], index_map[pred_label]] += 1

            for gt_idx in range(len(gt_boxes)):
                if gt_idx not in used_gt:
                    gt_label = int(gt_labels[gt_idx])
                    if gt_label in index_map:
                        cm[index_map[gt_label], index_map[bg_id]] += 1

        fig, ax = plt.subplots(figsize=(10, 9))
        ax.imshow(cm, interpolation="nearest")
        ax.set_title("Detection Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")

        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        ax.set_xticklabels(label_names, rotation=90, fontsize=8)
        ax.set_yticklabels(label_names, fontsize=8)

        for row in range(size):
            for col in range(size):
                value = cm[row, col]
                if value != 0:
                    ax.text(col, row, str(value), ha="center", va="center", fontsize=7)

        fig.tight_layout()
        fig.canvas.draw()

        width, height = fig.canvas.get_width_height()
        buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
        plt.close(fig)

        image_bgr = cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR)
        pixmap = QPixmap.fromImage(to_qimage(image_bgr)).scaled(
            900,
            800,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        dialog = ImagePopup("Confusion Matrix", pixmap, parent=self)
        dialog.exec_()


def main():
    app = QApplication(sys.argv)
    window = DetectionReviewApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
